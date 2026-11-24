import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForSequenceClassification

from hatexai.inference import TransformerHateService
from hatexai.models.xlm_roberta import XLMClassifier


@pytest.fixture
def classifier(tmp_path):
    vocab = {token: i for i, token in enumerate(
        ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "aaj", "join", "us", "please", "!"]
    )}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
                                       pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")
    tokenizer.save_pretrained(tmp_path)
    model = XLMRobertaForSequenceClassification(XLMRobertaConfig(
        vocab_size=len(vocab), hidden_size=12, intermediate_size=16, num_attention_heads=2,
        num_hidden_layers=1, max_position_embeddings=32, id2label={0: "non_hate", 1: "hate"},
        label2id={"non_hate": 0, "hate": 1},
    ))
    model.save_pretrained(tmp_path)
    return XLMClassifier(tmp_path, torch.device("cpu"), max_length=8, batch_size=4)


def test_xlmr_loads_local_weights_and_obeys_token_limit(classifier):
    ids = classifier.encode("aaj " * 20)["input_ids"]
    assert ids.shape == (1, 8)
    probs = classifier.predict_proba(["aaj join", "join us please"])
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1)


@pytest.mark.parametrize("method", ["integrated_gradients", "shap"])
def test_transformer_attributions_preserve_prediction_and_omit_special_tokens(classifier, method):
    service = TransformerHateService(classifier, (0.0, 1.0), {"ig_steps": 4, "shap_evals": 13, "top_k_tokens": 2})
    prediction = service.predict("  aaj join us!  ")
    result = service.analyze("  aaj join us!  ", method=method)
    assert result["decision"] == "needs_review"
    assert prediction["probability_hate"] == result["probability_hate"]
    exp = result["explanation"]
    assert exp["method"] == method
    assert exp["tokens"] == ["aaj", "join", "us", "!"]
    assert len(exp["scores"]) == 4 and np.isfinite(exp["scores"]).all()
    assert set(exp["faithfulness"]) == {"comprehensiveness", "sufficiency", "random_mask_drop"}


def test_grounded_flan_selection_uses_supplied_evidence(monkeypatch):
    from hatexai.models import llm_policy

    def score(_policy, prompts, completions):
        assert all("Class: hate" in prompt and "Context: example context" in prompt for prompt in prompts)
        assert all("reviewer" in text for text in completions)
        return torch.tensor([-2.0, -1.0]), torch.tensor([1, 1])

    monkeypatch.setattr(llm_policy, "completion_logprobs", score)
    policy = llm_policy.ExplanationPolicy.__new__(llm_policy.ExplanationPolicy)
    output = policy.grounded_rationale("aaj join us", "example context", "hate", 0.55,
                                       "needs_review", ["aaj", "join"])
    assert "“join”" in output and "“aaj”" not in output
