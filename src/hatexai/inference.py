from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hatexai.baselines.linear import load_linear_baseline
from hatexai.data.preprocess import extract_text_features
from hatexai.reward.scoring import heuristic_reward
from hatexai.xai.attributions import (
    Explanation,
    counterfactual_hints,
    format_rationale,
    lime_attribution,
    top_k_tokens,
)


LABEL_NAMES = {0: "non_hate", 1: "hate"}


class LinearHateService:
    def __init__(self, model_path: str | Path):
        self.model = load_linear_baseline(model_path)

    def predict_proba_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(texts)

    def analyze(self, text: str, context: str = "") -> dict[str, Any]:
        feats = extract_text_features(text)
        fused = f"{feats.normalized} {context}".strip()
        probs = self.predict_proba_texts([fused])[0]
        label = int(np.argmax(probs))
        tokens, scores = lime_attribution(fused, self.predict_proba_texts, samples=120)
        evidence = top_k_tokens(tokens, scores, k=8)
        rationale = format_rationale(label, float(probs[label]), evidence, "lime")
        rationale = rationale + " " + " ".join(counterfactual_hints(feats.normalized))
        reward = heuristic_reward(feats.normalized, rationale, gold_label=label, predicted_label=label)
        explanation = Explanation(
            method="lime",
            predicted_label=label,
            predicted_prob=float(probs[label]),
            tokens=tokens,
            scores=scores,
            rationale=rationale,
            top_evidence=evidence,
        )
        return {
            "label": LABEL_NAMES[label],
            "label_id": label,
            "probability_hate": float(probs[1]),
            "probability_non_hate": float(probs[0]),
            "language": feats.language,
            "code_mixed": feats.code_mixed,
            "normalized_text": feats.normalized,
            "reward": reward,
            "explanation": explanation.__dict__,
        }
