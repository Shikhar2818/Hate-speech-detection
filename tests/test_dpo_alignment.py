"""Regression tests for the DPO log-probability alignment bug (PROJECT_AUDIT.md 3.2).

The original implementation gathered logits at unshifted label positions, which is
not the model's likelihood of the completion. These tests would fail on the old
code and pass on the corrected shifting.
"""

from types import SimpleNamespace

import pytest
import torch

from hatexai.reward.dpo import dpo_loss
from hatexai.train.train_dpo import _sequence_logprob


class _Batch(dict):
    """Mimics a transformers BatchEncoding: dict-like with a .to() method."""

    def to(self, device):
        return self


class _StubTokenizer:
    pad_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        # Same fixed encoding for sources and targets: the test only cares about
        # the shifted-logit arithmetic, not about real tokenisation.
        return _Batch(
            {
                "input_ids": torch.tensor([[1, 2, 3, 0]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }
        )


class _StubModel(torch.nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.fixed_logits = torch.randn(1, 4, 5, generator=generator)

    def forward(self, **kwargs):  # accepts input_ids/attention_mask/labels
        return SimpleNamespace(logits=self.fixed_logits)


def _policy(seed: int = 0):
    return SimpleNamespace(tokenizer=_StubTokenizer(), model=_StubModel(seed), device=torch.device("cpu"))


def test_sequence_logprob_uses_shifted_targets():
    policy = _policy()
    result = _sequence_logprob(policy, ["src"], ["tgt"])

    labels = torch.tensor([1, 2, 3, -100])
    shift_labels = labels[1:]  # [2, 3, -100]
    logp = torch.log_softmax(policy.model.fixed_logits[:, :-1, :], dim=-1)
    mask = shift_labels != -100
    expected = (logp[0].gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask).sum() / mask.sum()

    assert torch.isclose(result[0], expected, atol=1e-5)


def test_sequence_logprob_is_not_the_unshifted_value():
    """Guard against regressing to the pre-fix (unshifted) computation."""
    policy = _policy()
    result = float(_sequence_logprob(policy, ["src"], ["tgt"])[0])

    labels = torch.tensor([1, 2, 3, -100])
    logp = torch.log_softmax(policy.model.fixed_logits, dim=-1)
    mask = labels != -100
    unshifted = (logp[0].gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask).sum() / mask.sum()

    assert not torch.isclose(torch.tensor(result), unshifted, atol=1e-5)


def test_dpo_loss_prefers_higher_policy_margin():
    ref = torch.tensor([0.0])
    better = dpo_loss(torch.tensor([2.0]), torch.tensor([-1.0]), ref, ref, beta=0.5)
    worse = dpo_loss(torch.tensor([-1.0]), torch.tensor([1.0]), ref, ref, beta=0.5)
    assert better < worse
    assert float(better) >= 0.0
