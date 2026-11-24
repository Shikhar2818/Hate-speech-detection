from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Hard input limits: enforced at the schema boundary so no oversized payload
# reaches the model or the feedback store.
MAX_TEXT_CHARS = 5000
MAX_CONTEXT_CHARS = 2000
MAX_COMMENT_CHARS = 2000

ExplanationMethod = Literal["lime", "occlusion"]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    context: str = Field("", max_length=MAX_CONTEXT_CHARS)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value


class ExplainRequest(PredictRequest):
    method: ExplanationMethod = "lime"


class Uncertainty(BaseModel):
    margin: float
    normalized_entropy: float
    is_uncertain: bool


class Explanation(BaseModel):
    method: str
    predicted_label: int
    predicted_prob: float
    tokens: list[str]
    scores: list[float]
    top_evidence: list[str]
    rationale: str
    faithfulness: dict[str, float | None]
    disclaimer: str


class PredictionResponse(BaseModel):
    label: str
    label_id: int
    decision: Literal["hate", "non_hate", "needs_review"] | None = None
    probability_hate: float
    probability_non_hate: float
    confidence: float
    language: str
    code_mixed: bool
    normalized_text: str
    uncertainty: Uncertainty
    model: dict[str, Any]


class ExplainResponse(PredictionResponse):
    explanation: Explanation


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    context: str = Field("", max_length=MAX_CONTEXT_CHARS)
    predicted_label: str | None = Field(None, max_length=32)
    probability_hate: float | None = Field(None, ge=0.0, le=1.0)
    model_version: str | None = Field(None, max_length=64)
    user_label: Literal["hate", "non_hate", "unsure"] | None = None
    comment: str = Field("", max_length=MAX_COMMENT_CHARS)


class FeedbackResponse(BaseModel):
    id: str
    status: str
    created_at: str
    message: str


class FeedbackRecord(BaseModel):
    """A stored report as returned by the queue.

    Note this does NOT inherit FeedbackResponse: that model's ``message`` field
    is a submit-time acknowledgement and is absent from stored records, so
    inheriting it made every GET /api/feedback fail response validation.
    """

    id: str
    status: str
    created_at: str
    text: str
    context: str = ""
    predicted_label: str | None = None
    probability_hate: float | None = None
    model_version: str | None = None
    user_label: str | None = None
    comment: str = ""
    reviewer_note: str | None = None
    reviewed_label: Literal["hate", "non_hate"] | None = None
    reviewed_at: str | None = None


class ReviewRequest(BaseModel):
    status: Literal["pending_review", "accepted", "rejected", "needs_discussion"]
    reviewer_note: str = Field("", max_length=MAX_COMMENT_CHARS)
    reviewed_label: Literal["hate", "non_hate"] | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    artifacts_present: bool
