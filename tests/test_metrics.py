import json

import numpy as np

from hatexai.eval.metrics import (
    compute_classification_metrics,
    error_rates,
    expected_calibration_error,
    explanation_faithfulness,
    group_fairness,
)


def test_metrics_balanced():
    metrics = compute_classification_metrics([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["auroc"] == 1.0


def test_confusion_matrix_and_per_class():
    metrics = compute_classification_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.2, 0.6, 0.4, 0.9])
    cm = metrics["confusion_matrix"]
    assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (1, 1, 1, 1)
    assert metrics["per_class"]["1"]["recall"] == 0.5
    assert metrics["per_class"]["0"]["support"] == 2


def test_error_rates():
    rates = error_rates([0, 0, 1, 1], [1, 0, 1, 0])
    assert rates["false_positive_rate"] == 0.5
    assert rates["false_negative_rate"] == 0.5
    # No negatives -> FPR is unavailable, not zero.
    assert error_rates([1, 1], [1, 0])["false_positive_rate"] is None


def test_calibration_error_is_bounded_and_sensitive():
    well_calibrated = expected_calibration_error([0, 1, 0, 1], [0.05, 0.95, 0.05, 0.95], n_bins=5)
    badly_calibrated = expected_calibration_error([0, 0, 0, 0], [0.9, 0.9, 0.9, 0.9], n_bins=5)
    assert well_calibrated is not None and well_calibrated < 0.06
    assert badly_calibrated is not None and badly_calibrated > 0.8


def test_nan_is_never_emitted():
    """Metrics must stay strict-JSON even when AUROC is undefined."""
    metrics = compute_classification_metrics([1, 1], [1, 1], [0.9, 0.8])
    assert metrics["auroc"] is None
    assert metrics["brier"] is None
    # json.dumps with allow_nan=False would raise on a NaN; this must not.
    json.dumps(metrics, allow_nan=False)


def test_group_fairness_guards_small_groups():
    report = group_fairness([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], ["en"] * 7 + ["hi"] * 3, min_support=5)
    assert report["en"]["insufficient_support"] is False
    assert report["en"]["f1_macro"] == 1.0
    assert report["hi"]["insufficient_support"] is True
    assert report["hi"]["f1_macro"] is None


def test_faithfulness():
    value = explanation_faithfulness([0.9, 0.8], [0.4, 0.3])
    assert abs(value - 0.5) < 1e-6
