from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from hatexai.eval.metrics import compute_classification_metrics
from hatexai.utils import read_jsonl

# A category with fewer than this many probes is listed but its metrics are
# flagged rather than presented as a robustness finding.
MIN_CATEGORY_SUPPORT = 3


def load_probes(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def evaluate_probe_set(
    probs_for: Callable[[str, str], float],
    probes: list[dict[str, Any]],
    threshold: float = 0.5,
    min_support: int = MIN_CATEGORY_SUPPORT,
) -> dict[str, Any]:
    """Score a diagnostic probe set and break results down by category.

    ``probs_for(text, context) -> P(hate)`` keeps this independent of any
    particular model. Results are separated into the unambiguous subset and the
    deliberately contested subset (reclaimed slurs, sarcasm), because scoring
    the two together hides exactly the cases moderation cares about.
    """
    predictions: list[dict[str, Any]] = []
    for probe in probes:
        prob = float(probs_for(probe["text"], probe.get("context", "")))
        pred = int(prob >= threshold)
        gold = int(probe["label"])
        predictions.append(
            {
                "id": probe.get("id"),
                "category": probe.get("category", "unknown"),
                "ambiguous": bool(probe.get("ambiguous", False)),
                "language": probe.get("language", "und"),
                "gold": gold,
                "pred": pred,
                "prob_hate": prob,
                "correct": pred == gold,
                "text": probe["text"],
            }
        )

    def subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"unavailable": "no probes in subset"}
        return compute_classification_metrics(
            [r["gold"] for r in rows], [r["pred"] for r in rows], [r["prob_hate"] for r in rows]
        )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_category[row["category"]].append(row)

    category_report: dict[str, Any] = {}
    for category, rows in sorted(by_category.items()):
        entry: dict[str, Any] = {"support": len(rows)}
        if len(rows) < min_support:
            entry["insufficient_support"] = True
            entry["accuracy"] = None
        else:
            entry["insufficient_support"] = False
            entry["accuracy"] = sum(r["correct"] for r in rows) / len(rows)
        entry["errors"] = [r["id"] for r in rows if not r["correct"]]
        category_report[category] = entry

    unambiguous = [r for r in predictions if not r["ambiguous"]]
    ambiguous = [r for r in predictions if r["ambiguous"]]

    return {
        "threshold": threshold,
        "n": len(predictions),
        "overall": subset_metrics(predictions),
        "unambiguous": subset_metrics(unambiguous),
        "ambiguous": subset_metrics(ambiguous),
        "by_category": category_report,
        "failures": [r for r in predictions if not r["correct"]],
    }
