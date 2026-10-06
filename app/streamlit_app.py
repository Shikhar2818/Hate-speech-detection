from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatexai.baselines.linear import train_linear_baseline
from hatexai.inference import LinearHateService
from hatexai.utils import load_config


st.set_page_config(page_title="HateXAI", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_service() -> LinearHateService:
    config = load_config(ROOT / "configs" / "default.yaml")
    ckpt = ROOT / config["output_dir"] / "checkpoints" / "linear_baseline.joblib"
    if not ckpt.exists():
        train_linear_baseline(config)
    return LinearHateService(ckpt)


def main() -> None:
    st.title("Hate Speech Detection with Explainable AI")
    st.caption("Reward-guided, multilingual, context-aware moderation prototype")

    service = get_service()
    col1, col2 = st.columns([1.2, 1])
    with col1:
        text = st.text_area("Social media post (T)", height=140, placeholder="Paste a post...")
        context = st.text_input("Context (C)", placeholder="Reply thread, page, or event")
        run = st.button("Analyze", type="primary")
    with col2:
        st.markdown(
            """
            **Input model:** \(x_i = \{T_i, V_i, C_i\}\)

            **Output:** \(\hat{y}_i \in \{0,1\}\) plus a LIME rationale and a reward score
            for explanation quality.
            """
        )

    if run and text.strip():
        result = service.analyze(text, context)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prediction", result["label"].replace("_", " ").title())
        m2.metric("P(hate)", f"{result['probability_hate']:.3f}")
        m3.metric("Language", result["language"])
        m4.metric("Explanation reward", f"{result['reward']:.2f}")
        st.info(result["explanation"]["rationale"])
        scores = result["explanation"]["scores"]
        tokens = result["explanation"]["tokens"]
        if tokens:
            chart = pd.DataFrame({"token": tokens, "attribution": scores})
            st.bar_chart(chart.set_index("token"))
        st.json(result)


if __name__ == "__main__":
    main()
