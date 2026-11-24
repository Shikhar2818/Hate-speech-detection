from __future__ import annotations

"""Model registry.

The model is loaded once per process and reused for every request. Loading a
TF-IDF + logistic-regression pipeline is cheap, but the same registry is what
would hold an XLM-R checkpoint, where per-request loading would be catastrophic
for latency and memory.
"""

import os
import logging
from pathlib import Path
from threading import RLock
from typing import Any

from hatexai.inference import build_service_from_config
from hatexai.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
MODEL_VERSION = "linear-union-calibrated-0.2.0"

# Re-entrant: get_service() holds the lock while calling get_config(), which
# also takes it. A plain Lock deadlocks the first request (fixed after it hung
# every /health and /api/predict call at startup).
_lock = RLock()
_service: Any | None = None
_baseline: Any | None = None
_baseline_config: dict | None = None
_config: dict[str, Any] | None = None


def get_config(config_path: str | Path | None = None) -> dict[str, Any]:
    global _config
    with _lock:
        if _config is None:
            _config = load_config(config_path or os.getenv("HATEXAI_CONFIG") or DEFAULT_CONFIG)
        return _config


def get_service():
    """Return the process-wide service instance, loading it on first use."""
    global _service
    with _lock:
        if _service is None:
            config = get_config()
            settings = config.get("transformer", {})
            if settings.get("enabled"):
                try:
                    _service = load_transformer(config, settings)
                except (OSError, ValueError, RuntimeError, KeyError, StopIteration) as exc:
                    logging.getLogger(__name__).warning("XLM-R unavailable; using baseline: %s", exc)
            if _service is None:
                _service = get_baseline_service()
            load_generator(_service, config)
        return _service


def get_baseline_service():
    global _baseline, _baseline_config
    with _lock:
        config = get_config()
        if _baseline is None or _baseline_config is not config:
            _baseline = build_service_from_config(config)
            _baseline_config = config
        return _baseline


def transformer_report(config):
    from hatexai.data.public_sources import file_digest, load_json

    settings = config["transformer"]
    report = load_json(Path(settings["evaluation_dir"]) / "transformer_comparison.json")
    manifest = Path(config["data"]["manifest_path"])
    training = Path(settings["checkpoint"]) / "training.json"
    if (report["dataset_manifest_sha256"] != file_digest(manifest)
            or report["classifier_training_sha256"] != file_digest(training)
            or report["test_sha256"] != load_json(manifest)["splits"]["test"]["sha256"]
            or load_json(training)["settings"]["max_length"] != settings["max_length"]):
        raise ValueError("Transformer evaluation does not match the benchmark, checkpoint or token limit")
    return report


def load_transformer(config, settings):
    from hatexai.inference import TransformerHateService
    from hatexai.models.xlm_roberta import XLMClassifier
    from hatexai.utils import resolve_device

    report = transformer_report(config)
    policy = next(row["threshold"] for row in report["systems"] if row["model"] == "XLM-RoBERTa-base")
    bounds = (float(policy["low"]), float(policy["high"]))
    if not 0 <= bounds[0] <= bounds[1] <= 1:
        raise ValueError("Invalid validation-selected decision bounds")
    model = XLMClassifier(settings["checkpoint"], resolve_device(config.get("device", "auto")),
                          settings["max_length"], settings["batch_size"])
    return TransformerHateService(model, bounds, config.get("xai", {}))


def load_generator(service, config):
    settings = config.get("explanation_policy", {})
    if not settings.get("enabled"):
        return
    from hatexai.models.llm_policy import ExplanationPolicy
    from hatexai.utils import resolve_device

    optimized = Path(settings["optimized_checkpoint"])
    paths = [(str(optimized), "DPO checkpoint")] if (optimized / "training_history.json").is_file() else []
    paths.append((settings.get("base_checkpoint") or settings["base_model"], "pretrained Flan-T5"))
    service.generator_status = "checkpoint unavailable"
    for path, name in paths:
        try:
            service.generator = ExplanationPolicy(path, resolve_device(config.get("device", "auto")),
                                                  local_only=True)
            service.generator_status = name
            return
        except (OSError, ValueError, RuntimeError) as exc:
            logging.getLogger(__name__).warning("Explanation checkpoint unavailable: %s", exc)


def reset_service() -> None:
    """Test hook: drop the cached model so the next call reloads it."""
    global _service, _baseline
    with _lock:
        _service = _baseline = None
