"""Check completion scores against T5's teacher-forced loss and validate input pairs."""

import json
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from hatexai.reward.dpo import dpo_loss
from hatexai.train.train_dpo import _sequence_logprob, load_reviewed_preferences, train_dpo, verify_preference_isolation
from hatexai.utils import write_json, write_jsonl


class _Batch(dict):
    """Mimics a transformers BatchEncoding: dict-like with a .to() method."""

    def to(self, device):
        return self


class _StubTokenizer:
    pad_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
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


def test_sequence_logprob_includes_first_token_and_masks_padding():
    policy = _policy()
    result = _sequence_logprob(policy, ["src"], ["tgt"])

    labels = torch.tensor([1, 2, 3, -100])
    logp = torch.log_softmax(policy.model.fixed_logits, dim=-1)
    expected = logp[0, torch.arange(3), labels[:3]].mean()

    assert torch.isclose(result[0], expected, atol=1e-5)


def test_sequence_logprob_matches_real_teacher_forced_t5_loss():
    from transformers import T5Config, T5ForConditionalGeneration

    policy = _policy()
    policy.model = T5ForConditionalGeneration(T5Config(
        vocab_size=8, d_model=16, d_ff=32, num_layers=1, num_heads=2,
        pad_token_id=0, decoder_start_token_id=0, eos_token_id=1,
    )).eval()
    with torch.no_grad():
        expected = policy.model(
            **policy.tokenizer(["src"]), labels=torch.tensor([[1, 2, 3, -100]]),
        ).loss
        actual = _sequence_logprob(policy, ["src"], ["tgt"])[0]
    assert torch.isclose(actual, -expected, atol=1e-5)


def test_preference_reader_requires_distinct_rationales_and_review(tmp_path):
    path = tmp_path / "reviewed.jsonl"
    pair = {"id": "pair-1", "text": "A post", "chosen_explanation": "Choice A",
            "rejected_explanation": "Choice B", "review_status": "accepted", "reviewer_id": "r1"}
    write_jsonl(path, [pair])
    assert load_reviewed_preferences(path)[0]["chosen_explanation"] == "Choice A"
    write_jsonl(path, [pair, pair])
    with pytest.raises(ValueError, match="repeats"):
        load_reviewed_preferences(path)
    write_jsonl(path, [{**pair, "rejected_explanation": "Choice A  "}])
    with pytest.raises(ValueError, match="identical"):
        load_reviewed_preferences(path)
    write_jsonl(path, [])
    with pytest.raises(ValueError, match="non-empty"):
        load_reviewed_preferences(path)
    write_jsonl(path, [{**pair, "review_status": "pending_review"}])
    config = {"dpo": {"preferences_path": str(path)}, "output_dir": str(tmp_path / "output")}
    with pytest.raises(ValueError, match="accepted human review"):
        train_dpo(config)
    assert not (tmp_path / "output").exists()


def test_dpo_loss_prefers_higher_policy_margin():
    ref = torch.tensor([0.0])
    better = dpo_loss(torch.tensor([2.0]), torch.tensor([-1.0]), ref, ref, beta=0.5)
    worse = dpo_loss(torch.tensor([-1.0]), torch.tensor([1.0]), ref, ref, beta=0.5)
    assert better < worse
    assert float(better) >= 0.0


def test_grounded_preferences_require_prompt_and_disjoint_posts(tmp_path):
    path = tmp_path / "pairs.jsonl"
    row = {"id": "1", "text": "An example social media post for duplicate checks",
           "chosen_explanation": "Chosen", "rejected_explanation": "Rejected",
           "review_status": "accepted", "reviewer_id": "fixture"}
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match="classifier-evidence prompt"):
        load_reviewed_preferences(path, require_prompt=True)
    write_jsonl(path, [{**row, "prompt": "A reviewed prompt"}])
    [train] = load_reviewed_preferences(path, require_prompt=True)
    assert train["prompt"] == "A reviewed prompt"
    with pytest.raises(ValueError, match="near-duplicate"):
        verify_preference_isolation([train], [{**train, "id": "2", "text": train["text"] + "."}])


def test_dpo_smoke_updates_policy_and_saves_training_record(tmp_path, monkeypatch):
    class TestPolicy:
        pad_token_id = 0

        def __init__(self, model_name, device):
            self.device = device
            self.tokenizer = self
            self.model = self
            self.logits = torch.nn.Parameter(torch.zeros(3))

        def __call__(self, *args, **kwargs):
            if args:
                ids = torch.tensor([[1 if text == "Choice A" else 2] for text in args[0]])
                return _Batch(input_ids=ids)
            return SimpleNamespace(logits=self.logits.expand(*kwargs["labels"].shape, 3))

        def prompt_for(self, text, context):
            return text

        def parameters(self):
            return [self.logits]

        trainable_parameters = parameters

        def eval(self):
            pass

        def train(self):
            pass

        def save_pretrained(self, directory):
            write_json(directory / "smoke-test.json", {"weights": self.logits.tolist()})

    path = tmp_path / "reviewed.jsonl"
    write_jsonl(path, [{"id": "fixture", "text": "A post", "chosen_explanation": "Choice A",
                        "rejected_explanation": "Choice B", "review_status": "accepted",
                        "reviewer_id": "test-fixture"}])
    module = import_module("hatexai.train.train_dpo")
    monkeypatch.setattr(module, "ExplanationPolicy", TestPolicy)
    output = tmp_path / "output"
    policy = train_dpo({
        "seed": 42, "device": "cpu", "output_dir": str(output),
        "dpo": {"preferences_path": str(path), "policy_model": "fixture", "epochs": 1,
                "batch_size": 1, "learning_rate": 0.01, "beta": 0.1},
    })
    assert policy.logits[1] > policy.logits[2]
    checkpoint = output / "checkpoints" / "dpo_policy"
    history = json.loads((checkpoint / "training_history.json").read_text(encoding="utf-8"))
    assert history["history"] == [{"epoch": 1, "pairs": 1, "mean_loss": pytest.approx(0.693147)}]
    assert (checkpoint / "smoke-test.json").exists()
