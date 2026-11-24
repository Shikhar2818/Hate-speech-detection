import json

import pytest

from hatexai.baselines.linear import build_linear_pipeline, load_text_split
from hatexai.data.preprocess import fuse_text, is_code_mixed, normalize_text
from hatexai.train.train_classifier import compute_class_weights
from hatexai.utils import read_jsonl, write_json, write_jsonl


def test_normalize_and_codemix():
    text = normalize_text("Hello @user see https://x.com and नमस्ते")
    assert "<USER>" in text
    assert "<URL>" in text
    assert is_code_mixed("Yeh movie zabardast thi") is False
    assert is_code_mixed("Yeh movie बहुत good thi") is True


def test_fuse_text_is_shared_and_metadata_is_opt_in():
    plain = fuse_text("Hello world", "reply thread")
    assert "reply thread" in plain
    assert "[LANG]" not in plain
    with_meta = fuse_text("Hello", language="en", code_mixed=True, include_metadata=True)
    assert "[LANG] en" in with_meta and "[CM] 1" in with_meta


def test_baseline_ablation_features_all_build():
    for feature_set in ("char", "word", "union"):
        pipeline = build_linear_pipeline(feature_set=feature_set)
        assert pipeline is not None
    with pytest.raises(ValueError):
        build_linear_pipeline(feature_set="nope")


def test_splits_use_identical_fusion_for_baseline_and_dataset():
    """The baseline must not leak Language/CodeMixed tokens the model does not get."""
    texts, _ = load_text_split("data/synthetic/train.jsonl")
    assert all("Language:" not in t and "CodeMixed:" not in t for t in texts)


def test_class_weights_are_inverse_frequency():
    weights = compute_class_weights([0, 0, 0, 1])
    # class 0 is 3x more frequent than class 1, so it must be weighted lower
    assert weights[0] < weights[1]
    assert abs(float(weights.mean()) - 1.0) < 1e-6


def test_write_jsonl_refuses_non_finite(tmp_path):
    path = tmp_path / "out.jsonl"
    with pytest.raises(ValueError):
        write_jsonl(path, [{"auroc": float("nan")}])


def test_write_json_roundtrip(tmp_path):
    path = tmp_path / "out.json"
    write_json(path, {"a": 1, "b": None})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": None}


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"x": 1}\n\n{"x": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"x": 1}, {"x": 2}]
