from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.registry import MODEL_VERSION, PROJECT_ROOT, get_config, get_service
from backend.schemas import (
    ExplainRequest,
    ExplainResponse,
    FeedbackRecord,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PredictRequest,
    PredictionResponse,
    ReviewRequest,
)
from backend.store import FeedbackStore

FRONTEND_DIR = PROJECT_ROOT / "frontend"
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
FEEDBACK_DIR = PROJECT_ROOT / "outputs" / "feedback"

MAX_IMAGE_BYTES = 5 * 1024 * 1024
IMAGE_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}

# Simple fixed-window rate limiter, per client. Enough to keep a local research
# demo from being trivially hammered; a real deployment would use a shared store.
RATE_LIMIT_REQUESTS = int(os.getenv("HATEXAI_RATE_LIMIT", "60"))
RATE_LIMIT_WINDOW_SECONDS = 60.0
_request_log: dict[str, deque[float]] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _request_log[client]
    while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    window.append(now)


def store() -> FeedbackStore:
    # Env override keeps tests and alternate deployments from writing into the
    # repository's outputs/ directory.
    return FeedbackStore(os.getenv("HATEXAI_FEEDBACK_DIR", str(FEEDBACK_DIR)))


def _evaluation_dir() -> Path:
    config = get_config()
    if config["data"].get("manifest_path"):
        return Path(config["output_dir"]) / "eval"
    return EVAL_DIR


def _model_version() -> str:
    return get_config().get("model_version", MODEL_VERSION)


def _model_info() -> dict[str, Any]:
    config = get_config()
    benchmark = config.get("benchmark_name")
    data_description = (
        f"{benchmark} public benchmark with an English held-out test set."
        if benchmark else "data/synthetic (24 train / 6 val / 6 test, 36 records, illustrative only)"
    )
    data_limitations = (
        ["The benchmark classifier was evaluated on English HateXplain posts.",
         "This run provides no evidence about Hindi or Hinglish performance."]
        if benchmark else [
            "The demo model is trained on 24 synthetic wiring examples.",
            "Probabilities on the bundled test split are compressed near 0.5.",
        ]
    )
    return {
        "name": "HateXAI",
        "version": _model_version(),
        "architecture": {
            "served_model": "TF-IDF (word + char n-grams) + logistic regression, Platt-calibrated",
            "optional_models": [
                "XLM-RoBERTa text+context encoder (implemented, no checkpoint in this environment)",
                "gated CLIP vision fusion (implemented, requires image-labelled data)",
                "Flan-T5 explanation policy with DPO preference alignment (implemented, untrained here)",
            ],
        },
        "supported_languages": ["English"] if benchmark else [
            "English",
            "Hindi (Devanagari and romanised)",
            "Hindi-English code-mixed (Hinglish)",
        ],
        "modalities": {
            "text": True,
            "context": True,
            "image": False,
            "image_note": "No vision model is trained or evaluated; image inputs are validated but not analysed.",
        },
        "explanation_methods": ["lime", "occlusion"],
        "evaluation": {
            "artifacts_available": _evaluation_dir().exists(),
            "note": "Metrics are read from the configured model's evaluation directory.",
        },
        "known_limitations": [
            *data_limitations,
            "Labels are culturally contested; the model is a research prototype, not an enforcement tool.",
            "Explanation scores are model-derived attributions, not evidence of intent.",
        ],
        "responsible_use": (
            "For research and assisted moderation only. Do not use as a sole basis for "
            "punitive action against a person or account. Always keep a human in the loop."
        ),
        "dataset": {
            "shipped": data_description,
            "probes": "data/probes/hard_cases.jsonl (36 author-constructed adversarial probes)",
            "recommended_public": ["HateXplain", "HASOC", "OLID/SOLID", "MMHS150K", "Hateful Memes"],
        },
    }


def create_app() -> FastAPI:
    activity: Counter[str] = Counter()
    activity_lock = Lock()

    def record_prediction(result: dict[str, Any]) -> None:
        with activity_lock:
            activity[result.get("decision") or result["label"]] += 1

    app = FastAPI(
        title="HateXAI API",
        version=MODEL_VERSION,
        description="Explainable hate-speech detection research backend.",
    )

    origins = os.getenv(
        "HATEXAI_ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            service_loaded = get_service() is not None
        except Exception:
            service_loaded = False
        return HealthResponse(
            status="ok" if service_loaded else "degraded",
            model_loaded=service_loaded,
            model_version=_model_version(),
            artifacts_present=_evaluation_dir().exists(),
        )

    @app.get("/api/model-info")
    def model_info() -> dict[str, Any]:
        return _model_info()

    @app.post("/api/predict", response_model=PredictionResponse, dependencies=[Depends(rate_limit)])
    def predict(payload: PredictRequest) -> dict[str, Any]:
        # Model loading happens inside the try so a failed/absent checkpoint
        # returns a clean 500 rather than an unhandled server error.
        try:
            service = get_service()
            result = service.predict(payload.text, payload.context)
        except Exception as exc:  # noqa: BLE001 - surface a clean error, log details server-side
            print(f"[predict] inference failed: {type(exc).__name__}: {exc}")
            raise HTTPException(status_code=500, detail="Inference failed. The model may be unavailable.") from exc
        result["model"] = {"name": "HateXAI", "version": _model_version(), "type": "linear-calibrated"}
        record_prediction(result)
        return result

    @app.post("/api/explain", response_model=ExplainResponse, dependencies=[Depends(rate_limit)])
    def explain(payload: ExplainRequest) -> dict[str, Any]:
        try:
            service = get_service()
            result = service.analyze(payload.text, payload.context, method=payload.method)
        except Exception as exc:  # noqa: BLE001
            print(f"[explain] inference failed: {type(exc).__name__}: {exc}")
            raise HTTPException(status_code=500, detail="Explanation failed. The model may be unavailable.") from exc
        result["model"] = {"name": "HateXAI", "version": _model_version(), "type": "linear-calibrated"}
        record_prediction(result)
        return result

    @app.post("/api/vision-check", dependencies=[Depends(rate_limit)])
    def vision_check(image: UploadFile = File(...)) -> dict[str, Any]:
        """Validate an uploaded image, then refuse to pretend to analyse it.

        No vision model is trained, so this endpoint never fabricates a
        prediction. It exists to exercise safe upload handling: type check, size
        cap, a real decode attempt, and guaranteed temp-file cleanup.
        """
        if image.content_type not in IMAGE_FORMATS:
            raise HTTPException(status_code=415, detail=f"Unsupported image type: {image.content_type}")

        try:
            from PIL import Image

            # FastAPI has already spooled the upload. Decode it in the route's worker thread.
            total = image.file.seek(0, os.SEEK_END)
            if total > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail="Image exceeds 5 MB limit")
            image.file.seek(0)
            with Image.open(image.file) as img:
                width, height = img.size
                fmt = img.format
                if fmt != IMAGE_FORMATS[image.content_type]:
                    raise HTTPException(status_code=415, detail="Image contents do not match the file type")
                img.load()
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="File is not a decodable image") from exc
        finally:
            image.file.close()

        return {
            "received_bytes": total,
            "width": width,
            "height": height,
            "format": fmt,
            "analyzed": False,
            "status": "vision_not_available",
            "message": "Image validated. Image classification is unavailable.",
        }

    @app.get("/api/evaluation")
    def evaluation() -> dict[str, Any]:
        eval_dir = _evaluation_dir()
        public_benchmark = bool(get_config()["data"].get("manifest_path"))
        artifacts = {
            "baseline_test": "baseline_test.json",
            "threshold": "threshold.json",
            "test_three_way": "test_three_way.json",
            "probes": "probes.json",
            "faithfulness": "faithfulness.json",
            "comparison": "comparison.json",
            "ablation": "ablation.json",
            "run_manifest": "run_manifest.json",
        }
        if public_benchmark:
            artifacts.update(baseline_training="baseline_training.json", baseline_val="baseline_val.json",
                             error_analysis="error_analysis.json")
        training_path = eval_dir / "baseline_training.json"
        training = json.loads(training_path.read_text(encoding="utf-8")) if (
            public_benchmark and training_path.exists()
        ) else {}
        with activity_lock:
            counts = dict(activity)
        out: dict[str, Any] = {
            "available": {}, "missing": [],
            "activity": {"total": sum(counts.values()), "by_decision": counts},
        }
        for key, filename in artifacts.items():
            path = eval_dir / filename
            if not path.exists():
                out["missing"].append(key)
                continue
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if public_benchmark and key in {"ablation", "faithfulness", "error_analysis"} and (
                not training or artifact.get("checkpoint_sha256") != training.get("checkpoint_sha256")
                or artifact.get("test_sha256") != training.get("dataset_splits", {}).get("test", {}).get("sha256")
            ):
                out["missing"].append(key)
                continue
            out["available"][key] = artifact
        out["run_hint"] = (
            "python scripts/run_pipeline.py --config configs/hatexplain_baseline.yaml --stage baseline"
            if get_config().get("benchmark_name") == "HateXplain"
            else "python experiments/run_experiments.py --config configs/default.yaml"
        )
        return out

    @app.post("/api/feedback", response_model=FeedbackResponse, status_code=201, dependencies=[Depends(rate_limit)])
    def submit_feedback(payload: FeedbackRequest, feedback_store: FeedbackStore = Depends(store)) -> FeedbackResponse:
        record = feedback_store.add(
            {
                "text": payload.text,
                "context": payload.context,
                "predicted_label": payload.predicted_label,
                "probability_hate": payload.probability_hate,
                "model_version": payload.model_version,
                "user_label": payload.user_label,
                "comment": payload.comment,
            }
        )
        return FeedbackResponse(
            id=record["id"],
            status=record["status"],
            created_at=record["created_at"],
            message=(
                "Thank you. Your report is stored as a review candidate and will not be used "
                "as a training label without human review."
            ),
        )

    @app.get("/api/feedback", response_model=list[FeedbackRecord])
    def list_feedback(limit: int = 50, feedback_store: FeedbackStore = Depends(store)) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        return feedback_store.list(limit=limit)

    @app.get("/api/feedback/summary")
    def feedback_summary(feedback_store: FeedbackStore = Depends(store)) -> dict[str, Any]:
        return feedback_store.summary()

    @app.post("/api/feedback/{feedback_id}/review")
    def review_feedback(
        feedback_id: str,
        payload: ReviewRequest,
        feedback_store: FeedbackStore = Depends(store),
    ) -> dict[str, Any]:
        review = feedback_store.review(feedback_id, payload.status, payload.reviewer_note, payload.reviewed_label)
        if review is None:
            raise HTTPException(status_code=404, detail="Unknown feedback id")
        return review

    if FRONTEND_DIR.exists():
        # Served from the same origin as the API, so the dashboard needs no CORS
        # allowances in the common case.
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
