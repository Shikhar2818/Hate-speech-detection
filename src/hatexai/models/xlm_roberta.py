"""Local sequence classifier shared by training, evaluation and serving."""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class XLMClassifier:
    def __init__(self, path, device, max_length, batch_size=16, *, training=False):
        options = {"local_files_only": not training}
        self.tokenizer = AutoTokenizer.from_pretrained(path, **options)
        if training:
            options.update(num_labels=2, id2label={0: "non_hate", 1: "hate"},
                           label2id={"non_hate": 0, "hate": 1})
        self.model = AutoModelForSequenceClassification.from_pretrained(path, **options).to(device).eval()
        if self.model.config.id2label != {0: "non_hate", 1: "hate"}:
            raise ValueError("Classifier checkpoint has incompatible label order")
        self.device, self.max_length, self.batch_size = device, max_length, batch_size

    def encode(self, texts, **options):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                              return_tensors="pt", **options).to(self.device)

    @torch.inference_mode()
    def predict_proba(self, texts):
        scores = [self.model(**self.encode(texts[i:i + self.batch_size])).logits.float().softmax(-1)
                  .cpu().numpy() for i in range(0, len(texts), self.batch_size)]
        return np.concatenate(scores) if scores else np.empty((0, 2))
