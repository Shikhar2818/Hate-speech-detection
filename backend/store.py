from __future__ import annotations

"""Append-only feedback store.

Design notes that matter for research integrity:

* Submitted feedback is a **review candidate**, never a training label. Records
  are written with ``status: pending_review`` and the API never feeds them back
  into any model automatically; a human review step changes the status.
* Corrections and review actions are written to separate append-only files, so
  the raw submission is never overwritten and an audit trail survives.
* Nothing here is used for evaluation, which keeps reported metrics free of
  feedback leakage.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from hatexai.utils import read_jsonl

_LOCK = Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FeedbackStore:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_path = self.base_dir / "feedback.jsonl"
        self.reviews_path = self.base_dir / "feedback_reviews.jsonl"

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "id": uuid.uuid4().hex[:12],
            "created_at": _now(),
            "status": "pending_review",
            **payload,
        }
        with _LOCK:
            with open(self.feedback_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return record

    def _latest_status(self) -> dict[str, dict[str, Any]]:
        if not self.reviews_path.exists():
            return {}
        latest: dict[str, dict[str, Any]] = {}
        for review in read_jsonl(self.reviews_path):
            latest[review["id"]] = review
        return latest

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.feedback_path.exists():
            return []
        records = read_jsonl(self.feedback_path)
        latest = self._latest_status()
        for record in records:
            review = latest.get(record["id"])
            if review:
                record["status"] = review["status"]
                record["reviewer_note"] = review.get("reviewer_note", "")
                record["reviewed_label"] = review.get("reviewed_label")
                record["reviewed_at"] = review.get("reviewed_at")
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return records[:limit]

    def get(self, feedback_id: str) -> dict[str, Any] | None:
        for record in self.list(limit=10_000):
            if record["id"] == feedback_id:
                return record
        return None

    def review(self, feedback_id: str, status: str, reviewer_note: str = "",
               reviewed_label: str | None = None) -> dict[str, Any] | None:
        if self.get(feedback_id) is None:
            return None
        review = {
            "id": feedback_id,
            "status": status,
            "reviewer_note": reviewer_note,
            "reviewed_label": reviewed_label if status == "accepted" else None,
            "reviewed_at": _now(),
        }
        with _LOCK:
            with open(self.reviews_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(review, ensure_ascii=False, allow_nan=False) + "\n")
        return review

    def summary(self) -> dict[str, Any]:
        records = self.list(limit=10_000)
        counts: dict[str, int] = {}
        for record in records:
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        return {"total": len(records), "by_status": counts}
