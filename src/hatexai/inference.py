from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hatexai.baselines.linear import load_linear_baseline
from hatexai.data.preprocess import extract_text_features, fuse_text
from hatexai.data.public_sources import file_digest, load_json
from hatexai.eval.thresholds import decision_three_way
from hatexai.reward.scoring import heuristic_reward
from hatexai.xai.attributions import (
    format_rationale,
    lime_attribution,
    occlusion_attribution,
    top_k_tokens,
)
from hatexai.xai.faithfulness import comprehensiveness, sufficiency

LABEL_NAMES = {0: "non_hate", 1: "hate"}


def uncertainty_from_probs(probs: np.ndarray) -> dict[str, Any]:
    """A transparent uncertainty signal, not a calibrated probability.

    Combines the decision margin with predictive entropy. Both are cheap and
    explainable; neither is a substitute for a calibrated model, which is why
    the API reports calibration separately via /api/evaluation.
    """
    probs = np.clip(np.asarray(probs, dtype=float), 1e-9, 1.0)
    margin = float(abs(probs[1] - probs[0]))
    entropy = float(-(probs * np.log(probs)).sum() / np.log(len(probs)))
    return {
        "margin": margin,
        "normalized_entropy": entropy,
        "is_uncertain": bool(margin < 0.30 or entropy > 0.80),
    }


class LinearHateService:
    """Serving wrapper around the trained linear baseline.

    This is the only model with a checkpoint shipped in the repository, so it is
    the default backend. The class is model-agnostic in its explanation logic:
    any ``predict_proba`` callable can be substituted.
    """
    model_type = "linear-calibrated"
    model_name = "TF-IDF + logistic regression"
    methods = ("lime", "occlusion")
    generator = None
    generator_status = "not configured"

    def __init__(self, model_path: str | Path, lime_samples: int = 120,
                 decision_bounds: tuple[float, float] | None = None):
        self.model = load_linear_baseline(model_path)
        self.lime_samples = lime_samples
        self.model_path = str(model_path)
        self.decision_bounds = decision_bounds

    def predict_proba_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(texts)

    def _score(self, text: str, context: str) -> tuple[dict[str, Any], str]:
        feats = extract_text_features(text)
        fused = fuse_text(text, context, feats.language, feats.code_mixed)
        probs = self.predict_proba_texts([fused])[0]
        label = int(np.argmax(probs))
        result = {
            "label": LABEL_NAMES[label],
            "label_id": label,
            "probability_hate": float(probs[1]),
            "probability_non_hate": float(probs[0]),
            "confidence": float(probs[label]),
            "language": feats.language,
            "code_mixed": feats.code_mixed,
            "normalized_text": feats.normalized,
            "uncertainty": uncertainty_from_probs(probs),
        }
        if self.decision_bounds is not None:
            result["decision"] = decision_three_way(float(probs[1]), *self.decision_bounds)
        return result, fused

    def predict(self, text: str, context: str = "") -> dict[str, Any]:
        result, _ = self._score(text, context)
        return result

    def analyze(
        self,
        text: str,
        context: str = "",
        method: str = "lime",
        gold_label: int | None = None,
    ) -> dict[str, Any]:
        if method not in self.methods:
            raise ValueError(f"Explanation method {method} is unavailable for {self.model_name}")
        result, fused = self._score(text, context)
        label = result["label_id"]
        prob = result["confidence"]

        if method == "occlusion":
            tokens, scores = occlusion_attribution(fused, self.predict_proba_texts, target_class=1)
        else:
            method = "lime"
            tokens, scores = lime_attribution(fused, self.predict_proba_texts, samples=self.lime_samples)

        evidence_count = sum(abs(score) > 1e-8 for score in scores)
        evidence = top_k_tokens(tokens, scores, k=min(8, evidence_count))
        rationale = format_rationale(label, prob, evidence, method)

        faith = {"comprehensiveness": None, "sufficiency": None}
        if evidence_count:
            mask_options = {"target_class": label, "top_k": min(5, evidence_count)}
            faith = {
                "comprehensiveness": comprehensiveness(
                    fused, tokens, scores, self.predict_proba_texts, **mask_options,
                ),
                "sufficiency": sufficiency(
                    fused, tokens, scores, self.predict_proba_texts, **mask_options,
                ),
            }
        result["explanation"] = {
            "method": method,
            "predicted_label": label,
            "predicted_prob": prob,
            "tokens": tokens,
            "scores": [float(s) for s in scores],
            "top_evidence": evidence,
            "rationale": rationale,
            "faithfulness": faith,
            "disclaimer": (
                "Highlighted tokens are model-derived feature attributions, not proof of intent. "
                "The summary describes the binary model score. The review policy uses separate thresholds."
            ),
        }
        # Reward is only meaningful with an external label. Computing it against
        # the model's own prediction (the previous behaviour) pinned the
        # correctness term to 1.0 for every input — see PROJECT_AUDIT.md 3.1.
        if gold_label is not None:
            result["offline_reward"] = heuristic_reward(
                result["normalized_text"], rationale, gold_label=gold_label, predicted_label=label
            )
        self.add_generated_explanation(result, text, context)
        return result

    def add_generated_explanation(self, result, text, context):
        if self.generator is None:
            return
        explanation = result["explanation"]
        original = (text.casefold(), context.casefold())
        evidence = [token for token in explanation["top_evidence"]
                    if sum(char.isalnum() for char in token) >= 2
                    and any(token.casefold() in passage for passage in original)]
        try:
            explanation["generated_rationale"] = self.generator.grounded_rationale(
                text, context, result["label"], result["confidence"], result.get("decision") or result["label"], evidence,
            )
        except (RuntimeError, ValueError):
            explanation["generated_rationale"] = None


class TransformerHateService(LinearHateService):
    model_type = "xlm-roberta"
    model_name = "XLM-RoBERTa-base"
    methods = ("integrated_gradients", "shap", "lime", "occlusion")

    def __init__(self, classifier, bounds, xai):
        self.classifier, self.decision_bounds, self.xai = classifier, bounds, xai
        self.lime_samples = int(xai.get("lime_samples", 120))

    def predict_proba_texts(self, texts):
        return self.classifier.predict_proba(texts)

    def analyze(self, text, context="", method="integrated_gradients", gold_label=None):
        if method in ("lime", "occlusion"):
            return super().analyze(text, context, method, gold_label)
        from hatexai.xai.transformer import explain_tokens

        result, fused = self._score(text, context)
        label, confidence = result["label_id"], result["confidence"]
        tokens, scores, faith = explain_tokens(self.classifier, fused, label, method, self.xai)
        evidence_count = sum(abs(score) > 1e-8 for score in scores)
        evidence = top_k_tokens(tokens, scores, k=min(int(self.xai.get("top_k_tokens", 8)), evidence_count))
        result["explanation"] = {
            "method": method, "predicted_label": label, "predicted_prob": confidence,
            "tokens": tokens, "scores": scores, "top_evidence": evidence, "faithfulness": faith,
            "rationale": format_rationale(label, confidence, evidence, method),
            "disclaimer": "Attributions describe model scores. A human reviewer can assess intent and context.",
        }
        self.add_generated_explanation(result, text, context)
        return result


def build_service_from_config(config: dict[str, Any]) -> LinearHateService:
    """Load the configured baseline. The demo can bootstrap its checkpoint."""
    output = Path(config["output_dir"])
    ckpt = output / "checkpoints" / "linear_baseline.joblib"
    bounds = None
    if config["data"].get("manifest_path"):
        training = load_json(output / "eval" / "baseline_training.json")
        threshold = output / "eval" / "threshold.json"
        if (training["checkpoint_sha256"] != file_digest(ckpt)
                or training["threshold_sha256"] != file_digest(threshold)):
            raise ValueError("Benchmark checkpoint and threshold do not match the training record")
        policy = load_json(threshold)
        bounds = (float(policy["low"]), float(policy["high"]))
        if not 0 <= bounds[0] <= bounds[1] <= 1:
            raise ValueError("Benchmark threshold bounds must be ordered probabilities")
    elif not ckpt.exists():
        from hatexai.baselines.linear import train_linear_baseline

        train_linear_baseline(config)
    return LinearHateService(ckpt, lime_samples=int(config.get("xai", {}).get("lime_samples", 120)),
                             decision_bounds=bounds)
