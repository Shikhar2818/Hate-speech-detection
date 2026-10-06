from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hatexai.data.dataset import MultimodalHateDataset, collate_batch
from hatexai.eval.metrics import compute_classification_metrics, group_fairness
from hatexai.utils import ensure_dir, write_jsonl


@torch.no_grad()
def evaluate_model(model, dataloader, device: torch.device) -> dict[str, Any]:
    model.eval()
    y_true, y_pred, y_prob, languages = [], [], [], []
    for batch in tqdm(dataloader, desc="eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)["logits"]
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs[:, 1].cpu().tolist())
        languages.extend(batch["language"])
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    metrics["fairness_by_language"] = group_fairness(y_true, y_pred, languages)
    return metrics


def run_evaluation(config: dict[str, Any], model, tokenizer, image_processor, device: torch.device) -> dict[str, Any]:
    dataset = MultimodalHateDataset(
        config["data"]["test_path"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config["data"]["max_text_length"],
    )
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], collate_fn=collate_batch)
    metrics = evaluate_model(model, loader, device)
    out_dir = ensure_dir(Path(config["output_dir"]) / "eval")
    write_jsonl(out_dir / "metrics.jsonl", [metrics])
    return metrics
