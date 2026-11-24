from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hatexai.eval.metrics import compute_classification_metrics, error_rates

# Cost model for moderation. A false negative (hate left up) is treated as more
# costly than a false positive (benign post hidden) by default, but both are
# tunable and must be justified for a deployment, not assumed.
DEFAULT_FN_COST = 5.0
DEFAULT_FP_COST = 1.0


def decision_three_way(prob_hate: float, low: float, high: float) -> str:
    """Map a probability to hate / non_hate / needs_review.

    Scores strictly between low and high go to human review. The high cutoff
    takes precedence if both thresholds are equal.
    """
    if low > high:
        raise ValueError("low threshold must be <= high threshold")
    if prob_hate >= high:
        return "hate"
    if prob_hate <= low:
        return "non_hate"
    return "needs_review"


def apply_three_way(
    y_prob: Sequence[float], low: float, high: float
) -> list[str]:
    return [decision_three_way(float(p), low, high) for p in y_prob]


def select_operating_point(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
    review_band: float = 0.10,
    grid: int = 99,
    require_both_classes: bool = True,
) -> dict[str, Any]:
    """Pick a decision threshold from *validation* data (never the test set).

    Minimises `fn_cost * FN + fp_cost * FP` over a grid of thresholds, then
    widens a symmetric abstention band of width ``review_band`` around it.
    Returns the thresholds plus the validation cost they achieve.

    ``require_both_classes`` rejects operating points that collapse the whole
    validation set into a single predicted class. When the asymmetric costs
    dominate — a common failure on small or weakly-separated validation splits —
    the unconstrained optimum is "call everything hate", which is a useless
    deployment and would silently turn every downstream decision into a review
    queue. If no threshold predicts both classes, the constraint is relaxed and
    ``degenerate: true`` is reported rather than hidden.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if y_true_arr.size == 0:
        raise ValueError("cannot select a threshold from an empty validation split")

    n = int(y_true_arr.size)

    def cost_at(t: float) -> tuple[int, int, float, int]:
        pred = (y_prob_arr >= t).astype(int)
        fp = int(((pred == 1) & (y_true_arr == 0)).sum())
        fn = int(((pred == 0) & (y_true_arr == 1)).sum())
        return fp, fn, fn_cost * fn + fp_cost * fp, int(pred.sum())

    best = None
    best_feasible = None
    for t in np.linspace(0.0, 1.0, grid + 1):
        fp, fn, cost, n_positive = cost_at(float(t))
        if best is None or cost < best[0]:
            best = (cost, float(t), fn, fp)
        # Feasible = the threshold actually predicts some of each class rather
        # than collapsing the split to a single label.
        predicts_both = 0 < n_positive < n
        if predicts_both and (best_feasible is None or cost < best_feasible[0]):
            best_feasible = (cost, float(t), fn, fp)

    if require_both_classes and best_feasible is not None:
        cost, threshold, fn, fp = best_feasible
        degenerate = False
    else:
        cost, threshold, fn, fp = best
        degenerate = True

    low = max(0.0, threshold - review_band / 2.0)
    high = min(1.0, threshold + review_band / 2.0)
    return {
        "threshold": threshold,
        "low": low,
        "high": high,
        "review_band": review_band,
        "validation_cost": float(cost),
        "validation_false_positives": fp,
        "validation_false_negatives": fn,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "degenerate": degenerate,
        "note": (
            "Threshold minimises weighted validation error; it is chosen on validation data only."
            if not degenerate
            else "No threshold separated both classes on validation; the reported point is degenerate."
        ),
    }


def evaluate_three_way(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    low: float,
    high: float,
) -> dict[str, Any]:
    """Metrics for a three-way decision, counting abstentions explicitly.

    Abstentions are excluded from the strict hate/non-hate precision-recall
    figures and reported separately, because a review queue is not a silent
    error — but it is also not a correct automatic decision.
    """
    decisions = apply_three_way(y_prob, low, high)
    covered_idx = [i for i, d in enumerate(decisions) if d != "needs_review"]
    abstained = len(decisions) - len(covered_idx)
    out: dict[str, Any] = {
        "n": len(decisions),
        "abstained": abstained,
        "abstention_rate": abstained / len(decisions) if decisions else None,
        "decided": len(covered_idx),
    }
    if covered_idx:
        yt = [int(y_true[i]) for i in covered_idx]
        yp = [1 if decisions[i] == "hate" else 0 for i in covered_idx]
        prob = [float(y_prob[i]) for i in covered_idx]
        out["decided_metrics"] = compute_classification_metrics(yt, yp, prob)
    else:
        out["decided_metrics"] = {"unavailable": "all examples abstained"}
    out["rates"] = error_rates(y_true, [1 if d == "hate" else 0 for d in decisions])
    return out
