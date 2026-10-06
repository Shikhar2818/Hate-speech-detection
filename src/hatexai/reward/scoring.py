from __future__ import annotations


def heuristic_reward(
    text: str,
    explanation: str,
    gold_label: int,
    predicted_label: int | None = None,
) -> float:
    """Lightweight reward for data construction and offline ranking.

    Combines label correctness, faithfulness (token overlap), and
    refusal of stereotyped or ungrounded explanations.
    """
    explanation_l = explanation.lower()
    predicted = predicted_label if predicted_label is not None else _label_from_text(explanation_l)
    correctness = 1.0 if predicted == gold_label else 0.0

    tokens = set(t for t in text.lower().split() if len(t) > 3)
    exp_tokens = set(explanation_l.split())
    overlap = len(tokens & exp_tokens) / max(1, len(tokens))
    faithfulness = min(1.0, overlap * 3.0)

    length_penalty = 1.0 if 8 <= len(explanation.split()) <= 80 else 0.4
    vague = any(w in explanation_l for w in ["i think maybe", "not sure", "could be anything"])
    overclaim = "always" in explanation_l and "because" not in explanation_l
    style = 0.0 if vague or overclaim else 1.0

    return 0.5 * correctness + 0.3 * faithfulness + 0.1 * length_penalty + 0.1 * style


def _label_from_text(explanation: str) -> int:
    if "non_hate" in explanation or "non-hate" in explanation or "not hate" in explanation:
        return 0
    if "hate" in explanation:
        return 1
    return -1
