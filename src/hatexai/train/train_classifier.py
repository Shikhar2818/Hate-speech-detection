from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from hatexai.data.dataset import MultimodalHateDataset, collate_batch
from hatexai.eval.evaluate import evaluate_model
from hatexai.models.multimodal import MultimodalHateDetector
from hatexai.utils import ensure_dir, resolve_device, set_seed


def compute_class_weights(labels: list[int], num_labels: int = 2) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 (so the loss scale is
    unchanged from unweighted cross-entropy).

    This is the imbalance handling the paper describes; the linear baseline uses
    `class_weight='balanced'`, and the transformer must match it for a fair
    comparison (PROJECT_AUDIT.md section 3.4).
    """
    counts = [0] * num_labels
    for label in labels:
        counts[int(label)] += 1
    total = sum(counts)
    raw = [(total / (num_labels * c)) if c > 0 else 0.0 for c in counts]
    # Normalise to mean 1 so the loss scale stays comparable to unweighted CE.
    mean = sum(raw) / len(raw)
    if mean > 0:
        raw = [w / mean for w in raw]
    return torch.tensor(raw, dtype=torch.float32)


def train_classifier(config: dict[str, Any]) -> MultimodalHateDetector:
    set_seed(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    model = MultimodalHateDetector(config).to(device)
    train_set = MultimodalHateDataset(
        config["data"]["train_path"],
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        max_length=config["data"]["max_text_length"],
    )
    val_set = MultimodalHateDataset(
        config["data"]["val_path"],
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        max_length=config["data"]["max_text_length"],
    )
    train_loader = DataLoader(train_set, batch_size=config["train"]["batch_size"], shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=config["train"]["batch_size"], collate_fn=collate_batch)

    class_weights = compute_class_weights([record.label for record in train_set.records]).to(device)
    print("class weights:", class_weights.tolist())

    optimizer = AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), weight_decay=float(config["train"]["weight_decay"]))
    total_steps = max(1, len(train_loader) * int(config["train"]["epochs"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config["train"]["warmup_ratio"])),
        num_training_steps=total_steps,
    )

    ckpt_dir = ensure_dir(Path(config["output_dir"]) / "checkpoints")
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_state = None
    patience = int(config["train"].get("early_stopping_patience", 2))
    epochs_without_improvement = 0

    model.train()
    step = 0
    for epoch in range(int(config["train"]["epochs"])):
        progress = tqdm(train_loader, desc=f"epoch {epoch+1}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                class_weights=class_weights,
            )
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
            optimizer.step()
            scheduler.step()
            step += 1
            if step % int(config["train"]["log_every"]) == 0:
                progress.set_postfix(loss=float(loss.item()))

        metrics = evaluate_model(model, val_loader, device)
        record = {"epoch": epoch + 1, "train_loss": float(loss.item()), "val": metrics}
        history.append(record)
        print(f"validation epoch {epoch+1}: {metrics}")

        f1 = metrics.get("f1_macro")
        if f1 is not None and f1 > best_f1:
            best_f1 = float(f1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping at epoch {epoch+1} (best val macro-F1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_state": model.state_dict(), "config": config}, ckpt_dir / "hate_detector.pt")
    (ckpt_dir / "hate_detector_history.json").write_text(
        json.dumps({"history": history, "best_val_f1_macro": best_f1}, indent=2),
        encoding="utf-8",
    )
    return model
