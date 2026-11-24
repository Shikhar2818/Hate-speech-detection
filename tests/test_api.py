import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HATEXAI_FEEDBACK_DIR", str(tmp_path / "feedback"))
    from backend.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 16), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    response = client.get("/api/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["modalities"]["image"] is False
    assert any("synthetic" in note.lower() for note in body["known_limitations"])


def test_predict_happy_path(client):
    response = client.post("/api/predict", json={"text": "These people are vermin and should leave.", "context": "news"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"hate", "non_hate"}
    assert 0.0 <= body["probability_hate"] <= 1.0
    assert abs(body["probability_hate"] + body["probability_non_hate"] - 1.0) < 1e-6
    assert "explanation" not in body  # /api/predict is decision-only
    assert "uncertainty" in body


def test_predict_rejects_blank_text(client):
    assert client.post("/api/predict", json={"text": "   "}).status_code == 422
    assert client.post("/api/predict", json={"text": "x" * 6000}).status_code == 422
    assert client.post("/api/predict", json={}).status_code == 422


def test_explain_returns_attributions_and_disclaimer(client):
    response = client.post("/api/explain", json={"text": "These people are vermin.", "method": "occlusion"})
    assert response.status_code == 200
    explanation = response.json()["explanation"]
    assert explanation["method"] == "occlusion"
    assert len(explanation["tokens"]) == len(explanation["scores"])
    assert explanation["top_evidence"]
    assert "not proof of intent" in explanation["disclaimer"]
    assert "comprehensiveness" in explanation["faithfulness"]


def test_explain_rejects_unknown_method(client):
    assert client.post("/api/explain", json={"text": "hello", "method": "magic"}).status_code == 422


def test_evaluation_endpoint_reports_missing_honestly(client):
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["available"], dict)
    assert isinstance(body["missing"], list)


def test_vision_modality_is_not_faked(client):
    ok = client.post("/api/vision-check", files={"image": ("a.png", _png_bytes(), "image/png")})
    assert ok.status_code == 200
    body = ok.json()
    assert body["analyzed"] is False
    assert body["status"] == "vision_not_available"

    bad_type = client.post("/api/vision-check", files={"image": ("a.txt", b"hello", "text/plain")})
    assert bad_type.status_code == 415

    not_an_image = client.post("/api/vision-check", files={"image": ("a.png", b"not really a png", "image/png")})
    assert not_an_image.status_code == 400


def test_feedback_is_a_review_candidate_and_reviewable(client):
    submit = client.post(
        "/api/feedback",
        json={"text": "counterspeech example", "predicted_label": "hate", "probability_hate": 0.8, "user_label": "non_hate", "comment": "this is counterspeech"},
    )
    assert submit.status_code == 201
    created = submit.json()
    assert created["status"] == "pending_review"
    assert "review candidate" in created["message"]

    listing = client.get("/api/feedback").json()
    assert listing and listing[0]["id"] == created["id"]
    assert "message" not in listing[0]

    review = client.post(f"/api/feedback/{created['id']}/review", json={"status": "accepted", "reviewer_note": "confirmed"})
    assert review.status_code == 200
    assert review.json()["status"] == "accepted"

    summary = client.get("/api/feedback/summary").json()
    assert summary["by_status"]["accepted"] == 1

    missing = client.post("/api/feedback/does-not-exist/review", json={"status": "accepted"})
    assert missing.status_code == 404


def test_inference_failure_returns_500_not_a_crash(client, monkeypatch):
    def boom():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("backend.main.get_service", boom)
    response = client.post("/api/predict", json={"text": "hello world"})
    assert response.status_code == 500
    assert "Inference failed" in response.json()["detail"]


def test_metrics_are_strict_json_in_evaluation_payload(client):
    """The dashboard parses this with JSON.parse, which rejects NaN/Infinity."""
    body = client.get("/api/evaluation").content.decode("utf-8")
    assert "NaN" not in body
    assert "Infinity" not in body
    json.loads(body)
