import numpy as np
import pytest

from experiments.evaluate_public_baseline import (
    analyze_errors,
    evaluate_ablations,
    fit_feature_ablations,
    measure_faithfulness,
)
from hatexai.utils import write_jsonl
from hatexai.xai.faithfulness import sufficiency


def test_native_labels_keep_offensive_and_normal_errors_separate():
    rows = [
        {"label": label, "code_mixed": False,
         "provenance": {"dataset": "hatexplain", "source_label": source_label}}
        for source_label, label in [("offensive", 0), ("normal", 0), ("hatespeech", 1)]
    ]
    report = analyze_errors(rows, [1, 0, 0])
    assert {row["source_label"]: row["errors"] for row in report["native_labels"]} == {
        "hatespeech": 1, "normal": 0, "offensive": 1,
    }
    assert report["by_script_mix"]["mixed_script"]["support"] == 0
    with pytest.raises(ValueError):
        analyze_errors(rows, [1])


def test_faithfulness_uses_seeded_controls_and_skips_short_posts():
    def predict(texts):
        return np.array([[.1, .9] if "signal" in text else [.9, .1] for text in texts])

    texts = ["signal rests among unrelated words", "short"]
    report = measure_faithfulness(texts, [1, 0], predict, seed=42, top_k=1)
    assert report == measure_faithfulness(texts, [1, 0], predict, seed=42, top_k=1)
    assert report["n"] == report["excluded_at_most_top_k_tokens"] == 1
    assert report["mean_comprehensiveness"] == pytest.approx(.8)
    assert report["mean_sufficiency"] == pytest.approx(.9)


def test_sufficiency_keeps_evidence_in_original_order():
    def predict(texts):
        return np.array([[.1, .9] if text == "first second" else [.9, .1] for text in texts])

    assert sufficiency(
        "first irrelevant second", ["first", "irrelevant", "second"], [.5, 0, .8], predict, top_k=2,
    ) == pytest.approx(.9)


def test_feature_ablations_report_each_fixed_variant_on_holdout(tmp_path):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    write_jsonl(train, [
        {"text": "a finch flies", "label": 0}, {"text": "look a finch", "label": 0},
        {"text": "a big cat", "label": 1}, {"text": "look a cat", "label": 1},
    ])
    write_jsonl(test, [{"text": "a finch", "label": 0}, {"text": "a cat", "label": 1}])
    report = evaluate_ablations(fit_feature_ablations(train), test)
    assert report["n"] == 2
    assert [row["feature_set"] for row in report["rows"]] == ["char", "word", "union"]
    for row in report["rows"]:
        assert row["test_f1_macro"] == 1
        assert 0 <= row["test_ece"] <= 1
