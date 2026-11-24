from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ExplanationPolicy:
    """Seq2seq LLM that generates a label plus a natural-language rationale."""

    def __init__(self, model_name: str, device: torch.device, *, local_only: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=local_only).to(device).eval()
        self.device = device

    def prompt_for(self, text: str, context: str = "") -> str:
        return (
            "Classify the social media post as HATE or NON_HATE and explain the decision. "
            "Focus on target group, intent, and whether the language attacks or dehumanizes. "
            f"Post: {text} Context: {context or 'none'}"
        )

    def generate(self, text: str, context: str = "", max_new_tokens: int = 96) -> str:
        prompt = self.prompt_for(text, context)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def logprob_of_completion(self, source: str, target: str) -> torch.Tensor:
        """Sum of token log-probabilities of ``target`` given ``source``."""
        totals, _ = completion_logprobs(self, [source], [target])
        return totals[0]

    def trainable_parameters(self):
        return self.model.parameters()

    @staticmethod
    def evidence_prompt(text, context, label, confidence, decision, evidence):
        return (
            f"Explain the classifier's prediction using only the listed evidence. "
            f"Class: {label}. Confidence: {confidence:.1%}. Moderation: {decision}. "
            f"Evidence: {', '.join(evidence)}. Context: {context or '(none)'}. Post: {text}"
        )

    @torch.inference_mode()
    def grounded_rationale(self, text, context, label, confidence, decision, evidence):
        if not evidence:
            return None
        source = self.evidence_prompt(text, context, label, confidence, decision, evidence)
        # Score grounded alternatives with Flan-T5. All quoted evidence comes from
        # the attributed input and no completion can assert author intent.
        verdict = {"hate": "Hate", "non_hate": "Non-hate", "needs_review": "Needs review"}.get(decision, label)
        options = [
            f"The moderation result is {verdict}. The highlighted token “{token}” influenced the classifier score."
            for token in evidence[:3]
        ]
        if decision == "needs_review" or confidence < 0.65:
            options = [option + " A reviewer can assess the meaning in context." for option in options]
        totals, counts = completion_logprobs(self, [source] * len(options), options)
        return options[int((totals / counts.clamp_min(1)).argmax())]


def completion_logprobs(policy, sources, targets, *, max_source_length=384, max_target_length=128):
    """Return token log-probability sums and valid token counts per completion."""
    source = policy.tokenizer(
        sources, padding=True, truncation=True, max_length=max_source_length, return_tensors="pt",
    ).to(policy.device)
    target = policy.tokenizer(
        targets, padding=True, truncation=True, max_length=max_target_length, return_tensors="pt",
    ).to(policy.device)
    labels = target["input_ids"].masked_fill(target["input_ids"] == policy.tokenizer.pad_token_id, -100)
    # T5 shifts the decoder inputs internally when labels are provided.
    logits = policy.model(**source, labels=labels).logits
    valid = labels != -100
    token_logp = logits.log_softmax(dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return token_logp.masked_fill(~valid, 0).sum(dim=1), valid.sum(dim=1)
