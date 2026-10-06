from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np
import torch


@dataclass
class Explanation:
    method: str
    predicted_label: int
    predicted_prob: float
    tokens: list[str]
    scores: list[float]
    rationale: str
    top_evidence: list[str]


def top_k_tokens(tokens: list[str], scores: list[float], k: int = 8) -> list[str]:
    ranked = sorted(zip(tokens, scores), key=lambda x: abs(x[1]), reverse=True)
    return [tok for tok, _ in ranked[:k] if tok.strip()]


def attention_attribution(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
) -> tuple[list[str], list[float]]:
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        attentions = outputs["attentions"]
    # Average last-layer attention from CLS to each token.
    last = attentions[-1][0].mean(dim=0)  # heads x seq x seq -> seq x seq after mean heads? 
    # attentions[-1]: (batch, heads, seq, seq)
    cls_to_tokens = last[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    scores = cls_to_tokens[: len(tokens)].tolist()
    return tokens, scores


def lime_attribution(text: str, predict_proba: Callable[[list[str]], np.ndarray], samples: int = 200) -> tuple[list[str], list[float]]:
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=["non_hate", "hate"])
    explanation = explainer.explain_instance(text, predict_proba, num_features=12, num_samples=samples)
    pairs = explanation.as_list()
    tokens = [p[0] for p in pairs]
    scores = [float(p[1]) for p in pairs]
    return tokens, scores


def integrated_gradients_attribution(
    embeddings: torch.Tensor,
    grads_fn: Callable[[torch.Tensor], torch.Tensor],
    steps: int = 32,
) -> torch.Tensor:
    baseline = torch.zeros_like(embeddings)
    alphas = torch.linspace(0, 1, steps, device=embeddings.device).view(-1, 1, 1)
    path = baseline + alphas * (embeddings - baseline)
    grads = []
    for i in range(steps):
        x = path[i : i + 1].detach().requires_grad_(True)
        grads.append(grads_fn(x))
    avg_grad = torch.stack(grads).mean(dim=0)
    return (embeddings - baseline) * avg_grad


def counterfactual_hints(text: str, toxic_cues: list[str] | None = None) -> list[str]:
    toxic_cues = toxic_cues or ["kill", "hate", "stupid", "go back", "vermin", "inferior"]
    lowered = text.lower()
    hints = []
    for cue in toxic_cues:
        if cue in lowered:
            hints.append(f"Removing or neutralizing '{cue}' would likely reduce hate probability.")
    if not hints:
        hints.append("No high-risk lexical cue found; decision relies more on context and target mention.")
    return hints


def format_rationale(label: int, prob: float, evidence: list[str], method: str) -> str:
    name = "hateful" if label == 1 else "non-hateful"
    evidence_str = ", ".join(evidence[:6]) if evidence else "overall context"
    return (
        f"The post is predicted as {name} (p={prob:.2f}) using {method}. "
        f"The most influential evidence is: {evidence_str}."
    )


def explanation_to_dict(exp: Explanation) -> dict[str, Any]:
    return asdict(exp)
