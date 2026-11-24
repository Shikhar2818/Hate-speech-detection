from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from hatexai.data.preprocess import extract_text_features, fuse_text
from hatexai.eval.metrics import compute_classification_metrics
from hatexai.utils import ensure_dir, read_jsonl

LABEL_MAP = {"non_hate": 0, "hate": 1, 0: 0, 1: 1, "0": 0, "1": 1}

# Feature sets available for the CPU baseline. ``union`` combines word and char
# n-grams; ablation showed it to be the strongest on both cross-validation and
# the diagnostic probe set, at the cost of worse raw calibration — hence the
# optional Platt scaling wrapper.
FEATURE_SETS = ("char", "word", "union")


def load_text_split(
    path: str | Path, include_metadata: bool = False
) -> tuple[list[str], list[int]]:
    """Load a split using the same canonical fusion as the transformer model."""
    texts, labels = [], []
    for item in read_jsonl(path):
        language = item.get("language")
        code_mixed = item.get("code_mixed")
        if include_metadata and ("language" not in item or "code_mixed" not in item):
            feats = extract_text_features(item["text"])
            language = item.get("language", feats.language)
            code_mixed = item.get("code_mixed", feats.code_mixed)
        fused = fuse_text(
            item["text"],
            item.get("context", ""),
            language=language,
            code_mixed=bool(code_mixed),
            include_metadata=include_metadata,
        )
        texts.append(fused)
        labels.append(int(LABEL_MAP[item["label"]]))
    return texts, labels


def _make_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=3000, class_weight="balanced")


def build_linear_pipeline(feature_set: str = "union", calibrate: bool = False, cv=3):
    """Build the baseline pipeline.

    ``calibrate`` wraps the pipeline in Platt (sigmoid) scaling fitted by
    cross-validation. Calibration does not change ranking (AUROC is unchanged)
    but reduces Brier score and ECE, which is what the abstention band relies on.
    """
    if feature_set == "char":
        base = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000, analyzer="char_wb")),
                ("clf", _make_classifier()),
            ]
        )
    elif feature_set == "word":
        base = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                ("clf", _make_classifier()),
            ]
        )
    elif feature_set == "union":
        base = Pipeline(
            steps=[
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                            ("char", TfidfVectorizer(ngram_range=(1, 2), min_df=1, analyzer="char_wb", sublinear_tf=True)),
                        ]
                    ),
                ),
                ("clf", _make_classifier()),
            ]
        )
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unknown feature_set {feature_set!r}; expected one of {FEATURE_SETS}")

    if calibrate:
        return CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    return base


def baseline_settings(config: dict[str, Any]) -> dict[str, Any]:
    baseline_cfg = config.get("baseline", {}) or {}
    return {
        "feature_set": baseline_cfg.get("feature_set", "union"),
        "calibrate": bool(baseline_cfg.get("calibrate", True)),
        "calibration_cv": int(baseline_cfg.get("calibration_cv", 3)),
    }


def train_linear_baseline(config: dict[str, Any]):
    settings = baseline_settings(config)
    texts, labels = load_text_split(config["data"]["train_path"])
    if len(set(labels)) < 2:
        raise ValueError("training split must contain both classes")
    cv = min(settings["calibration_cv"], len(labels))
    if settings["calibrate"] and config["data"].get("manifest_path"):
        groups = [row["duplicate_group"] for row in read_jsonl(config["data"]["train_path"])]
        splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=config["seed"])
        cv = list(splitter.split(texts, labels, groups))
    model = build_linear_pipeline(
        feature_set=settings["feature_set"],
        calibrate=settings["calibrate"],
        cv=cv,
    )
    model.fit(texts, labels)

    val_texts, val_labels = load_text_split(config["data"]["val_path"])
    preds = model.predict(val_texts)
    probs = model.predict_proba(val_texts)[:, 1]
    print("linear baseline val:", compute_classification_metrics(val_labels, preds.tolist(), probs.tolist()))

    out = ensure_dir(Path(config["output_dir"]) / "checkpoints")
    joblib.dump({"pipeline": model, "settings": settings}, out / "linear_baseline.joblib")
    return model


def load_linear_baseline(path: str | Path):
    """Load a baseline checkpoint, tolerating the legacy bare-pipeline format."""
    payload = joblib.load(path)
    if isinstance(payload, dict) and "pipeline" in payload:
        return payload["pipeline"]
    return payload


def load_baseline_settings(path: str | Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if isinstance(payload, dict):
        return payload.get("settings", {})
    return {}
