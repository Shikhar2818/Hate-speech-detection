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


def _tokenize_for_occlusion(text: str) -> list[str]:
    return [tok for tok in text.split(" ") if tok != ""]


def occlusion_attribution(
    text: str,
    predict_proba: Callable[[list[str]], np.ndarray],
    target_class: int = 1,
    token_replacement: str = "",
) -> tuple[list[str], list[float]]:
    """Delete-one occlusion attribution.

    For each whitespace token, remove it and measure the change in the target
    class probability. Model-agnostic (works for the linear baseline and any
    transformer) and directly interpretable: it is the same masking operation
    used by the comprehensiveness/sufficiency faithfulness tests, so the
    attribution and its evaluation cannot drift apart.
    """
    tokens = _tokenize_for_occlusion(text)
    if not tokens:
        return [], []
    base = float(predict_proba([text])[0][target_class])
    scores: list[float] = []
    for i, _ in enumerate(tokens):
        masked_tokens = tokens[:i] + ([token_replacement] if token_replacement else []) + tokens[i + 1 :]
        masked = " ".join(t for t in masked_tokens if t)
        if not masked.strip():
            masked = " "
        prob = float(predict_proba([masked])[0][target_class])
        scores.append(base - prob)
    return tokens, scores


def lime_attribution(
    text: str,
    predict_proba: Callable[[list[str]], np.ndarray],
    samples: int = 200,
) -> tuple[list[str], list[float]]:
    """LIME local surrogate over whitespace tokens (Ribeiro et al., 2016)."""
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=["non_hate", "hate"])
    explanation = explainer.explain_instance(
        text, predict_proba, num_features=12, num_samples=samples
    )
    pairs = explanation.as_list()
    tokens = [p[0] for p in pairs]
    scores = [float(p[1]) for p in pairs]
    return tokens, scores


def attention_attribution(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
) -> tuple[list[str], list[float]]:
    """Average last-layer attention from the CLS token to each position.

    Defensive: returns empty attributions instead of raising when a model does
    not expose attentions, so an explainability failure degrades gracefully
    rather than breaking the request (PROJECT_AUDIT.md section 3.9).
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    attentions = outputs["attentions"] if isinstance(outputs, dict) else getattr(outputs, "attentions", None)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    if not attentions:
        return tokens, [0.0] * len(tokens)

    last = attentions[-1][0]  # (heads, seq, seq)
    if last.dim() == 3:
        head_mean = last.mean(dim=0)  # (seq, seq)
    else:  # already (seq, seq)
        head_mean = last
    cls_row = head_mean[0].detach().cpu().numpy()
    scores = cls_row[: len(tokens)].tolist()
    return tokens, scores


def integrated_gradients_attribution(
    embeddings: torch.Tensor,
    grads_fn: Callable[[torch.Tensor], torch.Tensor],
    steps: int = 32,
) -> torch.Tensor:
    """Integrated gradients in an embedding space travelled from a zero baseline."""
    baseline = torch.zeros_like(embeddings)
    alphas = torch.linspace(0, 1, steps, device=embeddings.device).view(-1, 1, 1)
    path = baseline + alphas * (embeddings - baseline)
    grads = []
    for i in range(steps):
        x = path[i : i + 1].detach().requires_grad_(True)
        grads.append(grads_fn(x))
    avg_grad = torch.stack(grads).mean(dim=0)
    return (embeddings - baseline) * avg_grad


def integrated_gradients_for_module(
    model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    target_class: int = 1,
    steps: int = 16,
) -> torch.Tensor:
    """Integrated gradients over a module that accepts ``inputs_embeds``.

    This is the wired-to-a-real-model version; the module must accept
    ``inputs_embeds`` and ``attention_mask`` and return logits. Attribution is
    token-wise ||IG|| produced by summing the per-embedding-dimension scores.
    """
    baseline = torch.zeros_like(inputs_embeds)

    def score_fn(embeds: torch.Tensor) -> torch.Tensor:
        out = model(inputs_embeds=embeds, attention_mask=attention_mask)
        logits = out["logits"] if isinstance(out, dict) else out
        return logits[0, target_class]

    alphas = torch.linspace(0, 1, steps, device=inputs_embeds.device).view(-1, 1, 1)
    grads = []
    for i in range(steps):
        x = (baseline + alphas[i] * (inputs_embeds - baseline)).detach().requires_grad_(True)
        grad = torch.autograd.grad(score_fn(x), x, retain_graph=False)[0]
        grads.append(grad)
    avg_grad = torch.stack(grads).mean(dim=0)
    attributions = (inputs_embeds - baseline) * avg_grad
    return attributions.sum(dim=-1)  # (seq,) token-wise importance


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
