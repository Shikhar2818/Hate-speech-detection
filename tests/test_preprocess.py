from hatexai.data.preprocess import is_code_mixed, normalize_text
from hatexai.reward.scoring import heuristic_reward


def test_normalize_and_codemix():
    text = normalize_text("Hello @user see https://x.com and नमस्ते")
    assert "<USER>" in text
    assert "<URL>" in text
    assert is_code_mixed("Yeh movie zabardast thi") is False
    assert is_code_mixed("Yeh movie बहुत good thi") is True


def test_heuristic_reward_prefers_faithful():
    post = "These people are vermin and should leave the city"
    good = "HATE. The post dehumanizes a group as vermin and demands exclusion from the city."
    bad = "NON_HATE. I think maybe this could be anything."
    assert heuristic_reward(post, good, gold_label=1, predicted_label=1) > heuristic_reward(
        post, bad, gold_label=1, predicted_label=0
    )
