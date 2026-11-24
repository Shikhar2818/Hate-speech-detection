from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Below this many examples a group's metrics are reported as unavailable rather
# than as a number that a reader might over-trust.
MIN_GROUP_SUPPORT = 5


def _finite(value: Any) -> float | None:
    """Coerce to a JSON-strict float; non-finite values become None.

    ``json.dumps`` emits the bare token ``NaN``/``Infinity`` for non-finite
    floats, which is invalid JSON (RFC 8259) and breaks ``JSON.parse`` in the
    browser. Every metric that leaves this module must pass through here.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> float | None:
    """Top-label / positive-class expected calibration error.

    Uses equal-width bins over the predicted P(y=1) and weights each bin's
    |confidence - accuracy| gap by its share of the data.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if y_true_arr.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = y_true_arr.size
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo == 0.0:
            mask = (y_prob_arr >= lo) & (y_prob_arr <= hi)
        else:
            mask = (y_prob_arr > lo) & (y_prob_arr <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(y_prob_arr[mask].mean())
        accuracy = float(y_true_arr[mask].mean())
        ece += (count / total) * abs(confidence - accuracy)
    return _finite(ece)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float] | None = None,
    labels: Iterable[int] = (0, 1),
) -> dict[str, Any]:
    """Full classification report.

    All values are JSON-strict: finite floats or ``None``. ``None`` means
    "cannot be computed on this data" (e.g. AUROC with a single class in
    ``y_true``) and is deliberately distinguishable from 0.0.
    """
    labels = list(labels)
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return {"n": 0, "unavailable": "empty split"}

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        str(label): {
            "precision": _finite(precision[i]),
            "recall": _finite(recall[i]),
            "f1": _finite(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics: dict[str, Any] = {
        "n": len(y_true),
        "accuracy": _finite(accuracy_score(y_true, y_pred)),
        "f1_macro": _finite(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": _finite(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": _finite(np.mean([precision[i] for i in range(len(labels))])),
        "recall": _finite(np.mean([recall[i] for i in range(len(labels))])),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.astype(int).tolist(),
            "tn": int(cm[0, 0]) if cm.shape == (2, 2) else None,
            "fp": int(cm[0, 1]) if cm.shape == (2, 2) else None,
            "fn": int(cm[1, 0]) if cm.shape == (2, 2) else None,
            "tp": int(cm[1, 1]) if cm.shape == (2, 2) else None,
        },
    }

    rates = error_rates(y_true, y_pred)
    metrics["false_positive_rate"] = rates["false_positive_rate"]
    metrics["false_negative_rate"] = rates["false_negative_rate"]

    if y_prob is not None and len(y_prob) == len(y_true):
        probs = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
        if len(set(y_true)) > 1:
            metrics["auroc"] = _finite(roc_auc_score(y_true, probs))
            metrics["brier"] = _finite(np.mean((probs - np.asarray(y_true, dtype=float)) ** 2))
            metrics["ece"] = expected_calibration_error(y_true, probs)
        else:
            metrics["auroc"] = None
            metrics["brier"] = None
            metrics["ece"] = None
    else:
        metrics["auroc"] = None
        metrics["brier"] = None
        metrics["ece"] = None
    return metrics


def error_rates(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | None]:
    """FPR = FP/(FP+TN), FNR = FN/(FN+TP). None when the denominator is 0."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    negatives = yt == 0
    positives = yt == 1
    fpr = None
    fnr = None
    if negatives.sum() > 0:
        fpr = _finite(((yp == 1) & negatives).sum() / negatives.sum())
    if positives.sum() > 0:
        fnr = _finite(((yp == 0) & positives).sum() / positives.sum())
    return {"false_positive_rate": fpr, "false_negative_rate": fnr}


def group_fairness(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[str],
    min_support: int = MIN_GROUP_SUPPORT,
) -> dict[str, dict[str, Any]]:
    """Per-group macro-F1, FPR and FNR.

    Groups with fewer than ``min_support`` examples are reported with their
    support and ``insufficient_support: True`` instead of metrics, so that a
    single-example group cannot masquerade as a fairness finding.
    """
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for yt, yp, g in zip(y_true, y_pred, groups):
        buckets[g]["y_true"].append(int(yt))
        buckets[g]["y_pred"].append(int(yp))

    report: dict[str, dict[str, Any]] = {}
    for group, values in buckets.items():
        support = len(values["y_true"])
        entry: dict[str, Any] = {"support": support}
        if support < min_support:
            entry.update(
                {
                    "insufficient_support": True,
                    "min_support": min_support,
                    "f1_macro": None,
                    "false_positive_rate": None,
                    "false_negative_rate": None,
                }
            )
        else:
            entry.update(
                {
                    "insufficient_support": False,
                    "f1_macro": _finite(
                        f1_score(values["y_true"], values["y_pred"], average="macro", zero_division=0)
                    ),
                    **error_rates(values["y_true"], values["y_pred"]),
                }
            )
        report[group] = entry
    return report


def explanation_faithfulness(original_probs: Sequence[float], masked_probs: Sequence[float]) -> float:
    """Comprehensiveness: mean probability drop after removing top evidence.

    Kept for backward compatibility; ``xai.faithfulness`` provides the fuller
    comprehensiveness/sufficiency pair.
    """
    diffs = np.array(original_probs, dtype=float) - np.array(masked_probs, dtype=float)
    return float(diffs.mean())
