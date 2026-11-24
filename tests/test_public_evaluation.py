import numpy as np
import pytest

from experiments.compare_context import compare
from experiments.compare_language_policy import choose_threshold, rates_at
from experiments.evaluate_public_baseline import (
    analyze_errors,
    evaluate_ablations,
    evaluate_targets,
    fit_feature_ablations,
    measure_faithfulness,
)
from hatexai.data.public_sources import file_digest
from hatexai.utils import write_json, write_jsonl
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
    grouped = measure_faithfulness(texts, [1, 0], predict, seed=42, top_k=1, labels=[0, 0])
    assert grouped["by_correctness"]["incorrect"]["n"] == 1
    assert grouped["by_correctness"]["correct"]["n"] == 0


def test_target_analysis_requires_annotator_agreement():
    records = [{"label": 0, "provenance": {"dataset": "hatexplain", "source_id": "post"}}]
    annotations = {"post": {"annotators": [
        {"target": ["Group A", "Group A", "Group B"]},
        {"target": ["Group A"]},
        {"target": ["None"]},
    ]}}
    result = evaluate_targets(records, [1], annotations)
    assert set(result) == {"Group A"}
    assert result["Group A"]["support"] == 1
    assert result["Group A"]["false_positive_rate"] is None


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


def test_context_comparison_scores_identical_holdout_rows(tmp_path):
    rows = [
        {"text": "Community picnic", "context": "Town hall agenda", "label": 0},
        {"text": "Sports schedule", "context": "Sports discussion", "label": 1},
    ]
    manifest = {"splits": {}}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][split] = {"sha256": file_digest(path)}
    write_json(tmp_path / "manifest.json", manifest)
    result = compare(tmp_path, tmp_path / "report.json")
    assert {row["n"] for row in result["test"].values()} == {2}
    assert set(result["test"]) == {"comment_only", "comment_and_parent"}


def test_threshold_search_can_reduce_the_gap_without_sacrificing_recall():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.51, 0.85])
    languages = np.array(["en", "en", "hi-en", "hi-en"])
    threshold = choose_threshold(labels, scores, languages)
    comparison = rates_at(labels, scores, languages, threshold)
    assert comparison["fpr_gap"] == 0
    assert all(group["false_negative_rate"] == 0 for group in comparison["by_cohort"].values())
