from __future__ import annotations

from typing import Callable

import numpy as np


def _masked_text(tokens: list[str]) -> str:
    masked = " ".join(t for t in tokens if t)
    return masked if masked.strip() else " "


def comprehensiveness(
    text: str,
    tokens: list[str],
    scores: list[float],
    predict_proba: Callable[[list[str]], np.ndarray],
    target_class: int = 1,
    top_k: int = 5,
) -> float:
    """Mean probability drop when the top-k attributed tokens are removed.

    Higher is better: the highlighted evidence actually supported the
    prediction. Returns 0.0 for an empty explanation rather than raising.
    """
    if not tokens:
        return 0.0
    base = float(predict_proba([text])[0][target_class])
    ranked = sorted(range(len(tokens)), key=lambda i: abs(scores[i]), reverse=True)[:top_k]
    keep = [i for i in range(len(tokens)) if i not in set(ranked)]
    masked_prob = float(predict_proba([_masked_text([tokens[i] for i in keep])])[0][target_class])
    return base - masked_prob


def sufficiency(
    text: str,
    tokens: list[str],
    scores: list[float],
    predict_proba: Callable[[list[str]], np.ndarray],
    target_class: int = 1,
    top_k: int = 5,
) -> float:
    """Probability retained when *only* the top-k tokens are kept.

    Higher means the highlighted tokens alone carry the decision. Reported
    alongside comprehensiveness because either metric in isolation is easy to
    game.
    """
    if not tokens:
        return 0.0
    ranked = sorted(range(len(tokens)), key=lambda i: abs(scores[i]), reverse=True)[:top_k]
    kept = _masked_text([tokens[i] for i in ranked])
    return float(predict_proba([kept])[0][target_class])


def explanation_stability(
    texts: list[str],
    explain_fn: Callable[[str], tuple[list[str], list[float]]],
    top_k: int = 5,
) -> float:
    """Mean Jaccard overlap of top-k token sets across a list of inputs.

    Used as a coarse stability probe on near-duplicate inputs (e.g. the same
    post with punctuation noise); 1.0 means identical evidence.
    """
    if len(texts) < 2:
        return 1.0
    sets = []
    for text in texts:
        tokens, scores = explain_fn(text)
        ranked = sorted(range(len(tokens)), key=lambda i: abs(scores[i]), reverse=True)[:top_k]
        sets.append({tokens[i] for i in ranked})
    overlaps = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            overlaps.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return float(np.mean(overlaps)) if overlaps else 1.0
