from __future__ import annotations

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

    optimizer = AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), weight_decay=float(config["train"]["weight_decay"]))
    total_steps = max(1, len(train_loader) * int(config["train"]["epochs"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config["train"]["warmup_ratio"])),
        num_training_steps=total_steps,
    )

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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
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
        print(f"validation epoch {epoch+1}: {metrics}")

    ckpt_dir = ensure_dir(Path(config["output_dir"]) / "checkpoints")
    torch.save({"model_state": model.state_dict(), "config": config}, ckpt_dir / "hate_detector.pt")
    return model
