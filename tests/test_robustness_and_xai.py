import numpy as np

from hatexai.eval.robustness import evaluate_probe_set, load_probes
from hatexai.xai.attributions import lime_attribution, occlusion_attribution, top_k_tokens
from hatexai.xai.faithfulness import comprehensiveness, sufficiency


def _keyword_model(hate_words=("vermin", "burn")):
    def predict(texts):
        out = []
        for text in texts:
            low = text.lower()
            p = 0.9 if any(w in low for w in hate_words) else 0.1
            out.append([1 - p, p])
        return np.array(out)

    return predict


def test_probe_set_has_both_classes_and_categories():
    probes = load_probes("data/probes/hard_cases.jsonl")
    assert len(probes) >= 30
    labels = {p["label"] for p in probes}
    assert labels == {0, 1}
    assert any(p.get("ambiguous") for p in probes)
    assert len({p["category"] for p in probes}) >= 5


def test_probe_evaluation_separates_ambiguous_subset():
    probes = load_probes("data/probes/hard_cases.jsonl")
    report = evaluate_probe_set(lambda text, context: 1.0 if "vermin" in text.lower() else 0.0, probes)
    assert report["overall"]["n"] == len(probes)
    assert report["unambiguous"]["n"] + report["ambiguous"]["n"] == len(probes)
    assert "quoted_hate" in report["by_category"]


def test_occlusion_attribution_flags_the_trigger_word():
    text = "these people are vermin"
    tokens, scores = occlusion_attribution(text, _keyword_model(), target_class=1)
    ranked = top_k_tokens(tokens, scores, k=1)
    assert ranked and ranked[0] == "vermin"


def test_faithfulness_metrics_are_consistent_with_attribution():
    predict = _keyword_model()
    text = "these people are vermin"
    tokens, scores = occlusion_attribution(text, predict, target_class=1)
    comp = comprehensiveness(text, tokens, scores, predict, target_class=1)
    suff = sufficiency(text, tokens, scores, predict, target_class=1)
    # Removing the one supporting token should drop the probability sharply.
    assert comp > 0.5
    # Keeping only that token should retain a high probability.
    assert suff > 0.5


def test_lime_returns_attributions():
    text = "these people are vermin"
    tokens, scores = lime_attribution(text, _keyword_model(), samples=80)
    assert len(tokens) == len(scores)
    assert any("vermin" in t for t in tokens)


def test_lime_keeps_repeated_tokens_and_punctuation_in_input_order():
    text = "one vermin, one 🙂"
    result = lime_attribution(text, _keyword_model(), samples=80)
    tokens, scores = result
    assert tokens == text.split()
    assert len(tokens) == len(scores)
    assert scores[1] > 0
    assert result == lime_attribution(text, _keyword_model(), samples=80)
