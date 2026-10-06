from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hatexai.xai.attributions import (
    Explanation,
    attention_attribution,
    counterfactual_hints,
    format_rationale,
    lime_attribution,
    top_k_tokens,
)


class HateExplainer:
    def __init__(self, model, tokenizer, device: torch.device, config: dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config["xai"]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])["logits"]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def explain(self, text: str, context: str = "", method: str | None = None) -> Explanation:
        method = method or self.config["methods"][0]
        fused = f"Text: {text} [SEP] Context: {context or 'none'}"
        probs = self.predict_proba([fused])[0]
        label = int(np.argmax(probs))
        prob = float(probs[label])

        if method == "lime":
            tokens, scores = lime_attribution(fused, self.predict_proba, samples=int(self.config["lime_samples"]))
        else:
            encoded = self.tokenizer(fused, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            tokens, scores = attention_attribution(
                self.model, encoded["input_ids"], encoded["attention_mask"], self.tokenizer
            )

        evidence = top_k_tokens(tokens, scores, k=int(self.config["top_k_tokens"]))
        rationale = format_rationale(label, prob, evidence, method)
        hints = counterfactual_hints(text)
        rationale = rationale + " " + " ".join(hints)
        return Explanation(
            method=method,
            predicted_label=label,
            predicted_prob=prob,
            tokens=tokens,
            scores=scores,
            rationale=rationale,
            top_evidence=evidence,
        )
