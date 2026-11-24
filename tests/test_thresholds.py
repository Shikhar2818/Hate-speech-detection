import pytest

from hatexai.eval.thresholds import (
    apply_three_way,
    decision_three_way,
    evaluate_three_way,
    select_operating_point,
)


def test_decision_bands():
    assert decision_three_way(0.9, 0.4, 0.6) == "hate"
    assert decision_three_way(0.1, 0.4, 0.6) == "non_hate"
    assert decision_three_way(0.5, 0.4, 0.6) == "needs_review"


def test_decision_rejects_inverted_bands():
    with pytest.raises(ValueError):
        decision_three_way(0.5, 0.9, 0.1)


def test_threshold_selection_avoids_degenerate_point():
    """High FN cost must not collapse to 'call everything hate'."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.35, 0.7, 0.8, 0.9]
    point = select_operating_point(y_true, y_prob, fn_cost=50.0, fp_cost=1.0, review_band=0.1)
    assert point["degenerate"] is False
    # The chosen threshold must actually split the split into both classes.
    preds = [p >= point["threshold"] for p in y_prob]
    assert any(preds) and not all(preds)


def test_apply_three_way_and_abstention_accounting():
    probs = [0.05, 0.45, 0.5, 0.55, 0.95]
    decisions = apply_three_way(probs, 0.4, 0.6)
    assert decisions == ["non_hate", "needs_review", "needs_review", "needs_review", "hate"]
    result = evaluate_three_way([0, 0, 1, 1, 1], probs, 0.4, 0.6)
    assert result["abstained"] == 3
    assert result["decided"] == 2
    assert result["decided_metrics"]["n"] == 2
