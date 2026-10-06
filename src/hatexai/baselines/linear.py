from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hatexai.data.preprocess import extract_text_features
from hatexai.eval.metrics import compute_classification_metrics
from hatexai.utils import ensure_dir, read_jsonl

LABEL_MAP = {"non_hate": 0, "hate": 1, 0: 0, 1: 1, "0": 0, "1": 1}


def load_text_split(path: str | Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for item in read_jsonl(path):
        feats = extract_text_features(item["text"])
        context = item.get("context", "")
        texts.append(f"{feats.normalized} {context}".strip())
        labels.append(int(LABEL_MAP[item["label"]]))
    return texts, labels


def build_linear_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000, analyzer="char_wb")),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_linear_baseline(config: dict[str, Any]) -> Pipeline:
    texts, labels = load_text_split(config["data"]["train_path"])
    model = build_linear_pipeline()
    model.fit(texts, labels)

    val_texts, val_labels = load_text_split(config["data"]["val_path"])
    preds = model.predict(val_texts)
    probs = model.predict_proba(val_texts)[:, 1]
    print("linear baseline val:", compute_classification_metrics(val_labels, preds, probs.tolist()))

    out = ensure_dir(Path(config["output_dir"]) / "checkpoints")
    joblib.dump(model, out / "linear_baseline.joblib")
    return model


def load_linear_baseline(path: str | Path) -> Pipeline:
    return joblib.load(path)
