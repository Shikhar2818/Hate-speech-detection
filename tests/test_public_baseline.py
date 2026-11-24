from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import registry
from backend.main import create_app
from hatexai.baselines.linear import load_linear_baseline
from hatexai.data.public_sources import file_digest, load_json
from hatexai.eval.thresholds import decision_three_way
from hatexai.inference import build_service_from_config
from hatexai.utils import write_json, write_jsonl
from scripts.run_pipeline import run_public_baseline


@pytest.fixture
def benchmark_config(tmp_path):
    config = {
        "seed": 42,
        "output_dir": str(tmp_path / "artifacts"),
        "baseline": {"feature_set": "word", "calibrate": True, "calibration_cv": 3},
        "data": {"manifest_path": str(tmp_path / "manifest.json")},
    }
    manifest = {"splits": {}}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.jsonl"
        rows = [
            {"id": f"{split}-{i}", "text": f"{text} sample {i}", "language": "en",
             "label": i % 2, "duplicate_group": f"{split}-{i // 4}-{i % 2}"}
            for i, text in enumerate(["blue sky", "red sea"] * 12)
        ]
        write_jsonl(path, rows)
        config["data"][f"{split}_path"] = str(path)
        manifest["splits"][split] = {"file": path.name, "sha256": file_digest(path), "labels": {"0": 12, "1": 12}}
    write_json(config["data"]["manifest_path"], manifest)
    return config


def test_public_baseline_trains_and_records_metrics_and_policy(benchmark_config):
    results = run_public_baseline(benchmark_config)
    root = Path(benchmark_config["output_dir"])
    checkpoint = root / "checkpoints" / "linear_baseline.joblib"
    training = load_json(root / "eval" / "baseline_training.json")
    assert training["checkpoint_sha256"] == file_digest(checkpoint)
    assert training["threshold_sha256"] == file_digest(root / "eval" / "threshold.json")
    assert results["baseline_test"]["by_language"]["en"]["confusion_matrix"]["matrix"] == [[12, 0], [0, 12]]
    assert results["test_three_way"]["n"] == 24
    assert list(load_linear_baseline(checkpoint).predict(["blue sky", "red sea"])) == [0, 1]


def test_unverified_training_file_stops_before_fitting(benchmark_config):
    Path(benchmark_config["data"]["train_path"]).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="train.*checksum"):
        run_public_baseline(benchmark_config)
    assert not Path(benchmark_config["output_dir"]).exists()


def test_benchmark_checkpoint_policy_and_analytics_are_served(benchmark_config, tmp_path, monkeypatch):
    benchmark_config.update(benchmark_name="HateXplain", model_version="linear-hatexplain-test")
    results = run_public_baseline(benchmark_config)
    config_path = tmp_path / "serve.yaml"
    write_json(config_path, benchmark_config)
    monkeypatch.setenv("HATEXAI_CONFIG", str(config_path))
    monkeypatch.setattr(registry, "_config", None)
    monkeypatch.setattr(registry, "_service", None)

    with TestClient(create_app()) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] and health["model_version"] == "linear-hatexplain-test"
        model_info = client.get("/api/model-info").json()
        assert model_info["supported_languages"] == ["English"]
        assert "HateXplain" in model_info["dataset"]["shipped"]
        response = client.post("/api/explain", json={"text": "red sea", "method": "occlusion"})
        assert response.status_code == 200
        prediction = response.json()
        threshold = results["threshold"]
        assert prediction["decision"] == decision_three_way(
            prediction["probability_hate"], threshold["low"], threshold["high"],
        )
        assert prediction["explanation"]["tokens"]
        assert prediction["model"]["version"] == health["model_version"]
        evaluation = client.get("/api/evaluation").json()
        assert evaluation["available"]["baseline_test"] == results["baseline_test"]
        assert "ablation" in evaluation["missing"]

    threshold_path = Path(benchmark_config["output_dir"]) / "eval" / "threshold.json"
    write_json(threshold_path, {"low": 0, "high": 1})
    with pytest.raises(ValueError, match="do not match"):
        build_service_from_config(benchmark_config)
