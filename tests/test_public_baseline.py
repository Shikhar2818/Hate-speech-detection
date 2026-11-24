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


def test_missing_transformer_falls_back_and_reports_real_model(benchmark_config, tmp_path, monkeypatch):
    run_public_baseline(benchmark_config)
    benchmark_config["model_version"] = "local-baseline"
    benchmark_config["transformer"] = {
        "enabled": True, "checkpoint": str(tmp_path / "missing"), "evaluation_dir": str(tmp_path / "missing"),
        "model_version": "unavailable-transformer",
    }
    monkeypatch.setattr(registry, "_config", benchmark_config)
    monkeypatch.setattr(registry, "_service", None)
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["model_version"] == "local-baseline"
        info = client.get("/api/model-info").json()
        assert info["transformer_status"] == "baseline fallback"
        body = {"text": "blue sky", "method": "integrated_gradients"}
        assert client.post("/api/explain", json=body).status_code == 422
        assert client.post("/api/predict", json={"text": "blue sky", "classifier": "baseline"}).json()[
            "model"
        ]["type"] == "linear-calibrated"


def test_transformer_comparison_writes_same_split_metrics(benchmark_config, tmp_path, monkeypatch):
    from experiments import compare_transformer
    from hatexai.models import xlm_roberta
    from hatexai.utils import ensure_dir

    run_public_baseline(benchmark_config)
    presentation = {**benchmark_config, "model_version": "main-tfidf"}
    manifest_path = Path(benchmark_config["data"]["manifest_path"])
    checkpoint = ensure_dir(tmp_path / "transformer")
    write_json(checkpoint / "training.json", {
        "manifest_sha256": file_digest(manifest_path), "settings": {"max_length": 16},
    })
    write_json(checkpoint / "config.json", {})
    benchmark_config.update(device="cpu", transformer={
        "checkpoint": str(checkpoint), "evaluation_dir": str(tmp_path / "reports"),
        "max_length": 16, "batch_size": 4, "model_version": "research-xlmr", "enabled": True,
    })

    class TransformerFixture:
        def __init__(self, *args):
            self.model = load_linear_baseline(
                Path(benchmark_config["output_dir"]) / "checkpoints" / "linear_baseline.joblib",
            )

        def predict_proba(self, texts):
            return self.model.predict_proba(texts)

    monkeypatch.setattr(compare_transformer, "XLMClassifier", TransformerFixture)
    report = compare_transformer.compare(benchmark_config)
    assert len(report["systems"]) == 2
    assert all(model["metrics"]["by_language"]["en"]["n"] == 24 for model in report["systems"])
    assert (tmp_path / "reports" / "transformer_comparison.md").is_file()
    assert load_json(tmp_path / "reports" / "transformer_comparison.json") == report

    profile = tmp_path / "pilot.yaml"
    write_json(profile, benchmark_config)
    presentation["transformer"] = {"profile": str(profile)}
    monkeypatch.setattr(registry, "_config", presentation)
    monkeypatch.setattr(registry, "_service", None)
    monkeypatch.setattr(registry, "_transformer", None)
    monkeypatch.setattr(xlm_roberta, "XLMClassifier", TransformerFixture)
    with TestClient(create_app()) as client:
        info = client.get("/api/model-info").json()
        assert info["active_classifier"]["version"] == "main-tfidf"
        assert "integrated_gradients" in info["research_classifier"]["methods"]
        assert client.get("/api/evaluation").json()["available"]["transformer_comparison"] == report
        primary = client.post("/api/predict", json={"text": "blue sky"}).json()
        research = client.post("/api/predict", json={"text": "blue sky", "classifier": "transformer"}).json()
        assert primary["model"]["version"] == "main-tfidf"
        assert research["model"]["version"] == "research-xlmr"
        assert client.get("/health").json()["model_version"] == "main-tfidf"


def test_benchmark_checkpoint_policy_and_analytics_are_served(benchmark_config, tmp_path, monkeypatch):
    benchmark_config.update(benchmark_name="HateXplain", model_version="linear-hatexplain-test")
    results = run_public_baseline(benchmark_config)
    config_path = tmp_path / "serve.yaml"
    write_json(config_path, benchmark_config)
    monkeypatch.setenv("HATEXAI_CONFIG", str(config_path))
    monkeypatch.setattr(registry, "_config", None)
    monkeypatch.setattr(registry, "_service", None)
    evaluation_dir = Path(benchmark_config["output_dir"]) / "eval"
    training = load_json(evaluation_dir / "baseline_training.json")
    report = {"n": 24, "checkpoint_sha256": training["checkpoint_sha256"],
              "test_sha256": training["dataset_splits"]["test"]["sha256"]}
    write_json(evaluation_dir / "error_analysis.json", report)
    write_json(evaluation_dir / "faithfulness.json", {**report, "checkpoint_sha256": "stale"})

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
        assert evaluation["available"]["error_analysis"] == report
        assert "faithfulness" in evaluation["missing"]
        assert "ablation" in evaluation["missing"]

    threshold_path = Path(benchmark_config["output_dir"]) / "eval" / "threshold.json"
    write_json(threshold_path, {"low": 0, "high": 1})
    with pytest.raises(ValueError, match="do not match"):
        build_service_from_config(benchmark_config)
