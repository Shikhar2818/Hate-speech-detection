"""Train a text-only XLM-R benchmark without changing the multimodal trainer."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from transformers import get_linear_schedule_with_warmup

from hatexai.baselines.linear import load_text_split
from hatexai.data.public_sources import file_digest
from hatexai.eval.metrics import compute_classification_metrics
from hatexai.models.xlm_roberta import XLMClassifier
from hatexai.train.train_classifier import compute_class_weights
from hatexai.utils import ensure_dir, resolve_device, set_seed, write_json


def train_transformer(config):
    settings = config["transformer"]
    out = Path(settings["checkpoint"])
    if out.exists():
        raise FileExistsError(f"Select a fresh checkpoint directory: {out}")
    set_seed(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    batch_size = int(settings["batch_size"])
    epochs = int(settings["epochs"])
    patience = int(settings["early_stopping_patience"])
    max_length = int(settings["max_length"])
    if min(batch_size, epochs, patience) < 1 or not 4 <= max_length <= 512:
        raise ValueError("Use positive training settings and a token limit between 4 and 512")
    classifier = XLMClassifier(settings["base_model"], device, max_length, batch_size, training=True)
    model = classifier.model
    layers = model.base_model.encoder.layer
    last = int(settings.get("train_encoder_layers", len(layers)))
    if not 0 <= last <= len(layers):
        raise ValueError("Invalid number of trainable encoder layers")
    if last < len(layers):
        model.base_model.requires_grad_(False)
        for layer in layers[len(layers) - last:]:
            layer.requires_grad_(True)
    parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    train, labels = load_text_split(config["data"]["train_path"])
    val, val_labels = load_text_split(config["data"]["val_path"])
    weights = compute_class_weights(labels).to(device)
    steps = epochs * ((len(train) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.06), steps)
    rng = np.random.default_rng(config["seed"])
    best, history = -1.0, []
    with TemporaryDirectory(dir=ensure_dir(out.parent)) as staging:
        checkpoint = Path(staging) / "best"
        for epoch in range(epochs):
            model.train()
            if last < len(layers):
                model.base_model.eval()
                for layer in layers[len(layers) - last:]:
                    layer.train()
            order = rng.permutation(len(train))
            loss_total = 0.0
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                x = classifier.encode([train[i] for i in batch])
                y = torch.tensor([labels[i] for i in batch], device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(**x).logits
                loss = cross_entropy(logits, y, weight=weights)
                if not torch.isfinite(loss):
                    raise ValueError("Classifier loss became non-finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                loss_total += loss.item() * len(batch)
                if start % (50 * batch_size) == 0:
                    print(f"Epoch {epoch + 1}, {start}/{len(train)} posts, loss {loss.item():.4f}", flush=True)
            model.eval()
            probs = classifier.predict_proba(val)
            metrics = compute_classification_metrics(val_labels, probs.argmax(1), probs[:, 1])
            f1 = metrics["f1_macro"]
            history.append({"epoch": epoch + 1, "loss": loss_total / len(train), "validation": metrics})
            print(f"Epoch {epoch + 1}: validation macro F1 {f1:.4f}", flush=True)
            if f1 > best:
                best, best_epoch = f1, epoch
                model.save_pretrained(checkpoint)
                classifier.tokenizer.save_pretrained(checkpoint)
            elif epoch - best_epoch >= patience:
                break
        write_json(checkpoint / "training.json", {
            "settings": settings, "history": history, "best_epoch": best_epoch + 1,
            "seed": config["seed"], "device": str(device), "train_rows": len(train),
            "torch": torch.__version__, "trainable_parameters": sum(x.numel() for x in parameters),
            "manifest_sha256": file_digest(Path(config["data"]["manifest_path"])),
        })
        checkpoint.rename(out.resolve())
    return XLMClassifier(out, device, max_length, batch_size)
