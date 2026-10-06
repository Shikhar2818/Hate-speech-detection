from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from langdetect import detect
except Exception:  # pragma: no cover
    detect = None


CODEMIX_PATTERN = re.compile(
    r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF"
    r"\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF"
    r"\u0D00-\u0D7F]+"
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.strip()
    text = URL_PATTERN.sub(" <URL> ", text)
    text = MENTION_PATTERN.sub(" <USER> ", text)
    text = HASHTAG_PATTERN.sub(r" \1 ", text)
    text = text.replace("\n", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def detect_language(text: str) -> str:
    if not text.strip():
        return "und"
    if detect is None:
        return "en" if text.isascii() else "unk"
    try:
        return detect(text)
    except Exception:
        return "unk"


def is_code_mixed(text: str) -> bool:
    has_latin = bool(re.search(r"[A-Za-z]", text))
    has_indic = bool(CODEMIX_PATTERN.search(text))
    return has_latin and has_indic


@dataclass
class TextFeatures:
    raw: str
    normalized: str
    language: str
    code_mixed: bool


def extract_text_features(text: str) -> TextFeatures:
    normalized = normalize_text(text)
    return TextFeatures(
        raw=text,
        normalized=normalized,
        language=detect_language(normalized),
        code_mixed=is_code_mixed(normalized),
    )
