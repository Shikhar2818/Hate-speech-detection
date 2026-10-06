from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true: list[int], y_pred: list[int], y_prob: list[float] | None = None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auroc"] = float("nan")
    return metrics


def group_fairness(y_true: list[int], y_pred: list[int], groups: list[str]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for yt, yp, g in zip(y_true, y_pred, groups):
        buckets[g]["y_true"].append(yt)
        buckets[g]["y_pred"].append(yp)
    report = {}
    for group, values in buckets.items():
        report[group] = {
            "support": float(len(values["y_true"])),
            "f1_macro": float(f1_score(values["y_true"], values["y_pred"], average="macro", zero_division=0)),
            "false_positive_rate": _fpr(values["y_true"], values["y_pred"]),
        }
    return report


def _fpr(y_true: list[int], y_pred: list[int]) -> float:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    negatives = y_true_arr == 0
    if negatives.sum() == 0:
        return float("nan")
    return float(((y_pred_arr == 1) & negatives).sum() / negatives.sum())


def explanation_faithfulness(original_probs: list[float], masked_probs: list[float]) -> float:
    """Comprehensiveness: probability drop after masking top evidence."""
    diffs = np.array(original_probs) - np.array(masked_probs)
    return float(diffs.mean())
