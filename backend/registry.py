from __future__ import annotations

"""Model registry.

The model is loaded once per process and reused for every request. Loading a
TF-IDF + logistic-regression pipeline is cheap, but the same registry is what
would hold an XLM-R checkpoint, where per-request loading would be catastrophic
for latency and memory.
"""

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
_config: dict[str, Any] | None = None


def get_config(config_path: str | Path | None = None) -> dict[str, Any]:
    global _config
    with _lock:
        if _config is None:
            _config = load_config(config_path or DEFAULT_CONFIG)
        return _config


def get_service():
    """Return the process-wide service instance, loading it on first use."""
    global _service
    with _lock:
        if _service is None:
            config = get_config()
            _service = build_service_from_config(config)
        return _service


def reset_service() -> None:
    """Test hook: drop the cached model so the next call reloads it."""
    global _service
    with _lock:
        _service = None
