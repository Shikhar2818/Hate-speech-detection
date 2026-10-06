from hatexai.eval.metrics import compute_classification_metrics, explanation_faithfulness


def test_metrics_balanced():
    metrics = compute_classification_metrics([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_faithfulness():
    value = explanation_faithfulness([0.9, 0.8], [0.4, 0.3])
    assert abs(value - 0.5) < 1e-6
