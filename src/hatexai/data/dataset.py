from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from hatexai.data.preprocess import extract_text_features
from hatexai.utils import read_jsonl


LABEL_MAP = {"non_hate": 0, "hate": 1, 0: 0, 1: 1, "0": 0, "1": 1}


@dataclass
class HateExample:
    id: str
    text: str
    label: int
    language: str
    code_mixed: bool
    context: str
    image_path: str | None
    chosen_explanation: str
    rejected_explanation: str


class MultimodalHateDataset(Dataset):
    """Dataset for x_i = {T_i, V_i, C_i} with optional preference explanations."""

    def __init__(self, path: str | Path, tokenizer=None, image_processor=None, max_length: int = 256):
        self.records = [self._to_example(item) for item in read_jsonl(path)]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.records[index]
        item: dict[str, Any] = {
            "id": example.id,
            "text": example.text,
            "label": torch.tensor(example.label, dtype=torch.long),
            "language": example.language,
            "code_mixed": example.code_mixed,
            "context": example.context,
            "chosen_explanation": example.chosen_explanation,
            "rejected_explanation": example.rejected_explanation,
        }
        fused = self._fuse_text(example)
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                fused,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)
        else:
            item["fused_text"] = fused

        if self.image_processor is not None:
            image = self._load_image(example.image_path)
            pixels = self.image_processor(images=image, return_tensors="pt")
            item["pixel_values"] = pixels["pixel_values"].squeeze(0)
        return item

    def _fuse_text(self, example: HateExample) -> str:
        parts = [f"Text: {example.text}"]
        if example.context:
            parts.append(f"Context: {example.context}")
        parts.append(f"Language: {example.language}")
        parts.append(f"CodeMixed: {int(example.code_mixed)}")
        return " [SEP] ".join(parts)

    def _load_image(self, path: str | None) -> Image.Image:
        if path and Path(path).exists():
            return Image.open(path).convert("RGB")
        return Image.new("RGB", (224, 224), color=(240, 240, 240))

    @staticmethod
    def _to_example(item: dict[str, Any]) -> HateExample:
        feats = extract_text_features(item["text"])
        return HateExample(
            id=str(item.get("id", "")),
            text=feats.normalized,
            label=int(LABEL_MAP[item["label"]]),
            language=item.get("language", feats.language),
            code_mixed=bool(item.get("code_mixed", feats.code_mixed)),
            context=item.get("context", ""),
            image_path=item.get("image_path"),
            chosen_explanation=item.get("chosen_explanation", ""),
            rejected_explanation=item.get("rejected_explanation", ""),
        )


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [row[key] for row in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out
