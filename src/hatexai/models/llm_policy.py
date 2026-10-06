from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ExplanationPolicy:
    """Seq2seq LLM that generates a label plus a natural-language rationale."""

    def __init__(self, model_name: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
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
        source_enc = self.tokenizer(source, return_tensors="pt", truncation=True, max_length=384).to(self.device)
        target_enc = self.tokenizer(target, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        labels = target_enc["input_ids"]
        outputs = self.model(**source_enc, labels=labels)
        # Token-average log-likelihood of the completion.
        token_count = (labels != self.tokenizer.pad_token_id).sum().clamp(min=1)
        return -outputs.loss * token_count / token_count

    def trainable_parameters(self):
        return self.model.parameters()
