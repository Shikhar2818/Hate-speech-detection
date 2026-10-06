# Hate Speech Detection through Explainable AI and Reward-Optimized LLMs

Bennett University project under **Dr. Pradeepika Verma** (`pradeepika.verma@bennett.edu.in`, N2-WS-062).

This repository implements an explainable, multilingual, optionally multimodal hate-speech detector. A post is modeled as

\[
x_i = \{T_i, V_i, C_i\}, \quad f_\theta(x_i) \rightarrow y_i \in \{0,1\}
\]

where \(T_i\) is text (including code-mixed language), \(V_i\) is optional image/meme content, and \(C_i\) is conversational or page context. Predictions are paired with token-level explanations (LIME / attention) and a Direct Preference Optimization (DPO) loop that aligns LLM rationales with faithful, less biased explanations.

## Deliverables

| Item | Location |
| --- | --- |
| Source code | `src/hatexai`, `scripts/`, `app/` |
| ACL paper | `paper/acl_paper.tex` |
| Presentation | `slides/presentation.html` |

## Architecture

1. **Preprocessing.** URL/user normalization, language ID, code-mix flags.
2. **Lightweight baseline.** Character n-gram TF-IDF + logistic regression (runs on CPU with the bundled synthetic data).
3. **Multimodal classifier.** XLM-RoBERTa (text + context) with optional frozen CLIP vision features and gated fusion.
4. **XAI.** LIME, attention rollout from the CLS token, counterfactual cue hints, explanation faithfulness.
5. **Reward-optimized LLM.** Flan-T5 policy trained with DPO on chosen vs rejected rationales; a heuristic reward scores correctness, faithfulness, and style.
6. **Fairness eval.** Macro-F1 and false-positive rate by language group.

```
x_i={T,V,C} -> encoder/fusion -> y_hat
                 |                -> LIME/attention rationale
                 +-> LLM policy -> DPO(reward, preferences)
```

## Setup

Python 3.10+ recommended.

```powershell
cd C:\Users\shikh\OneDrive\Desktop\NLP
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
```

## Quick start (CPU, no large downloads)

Train and evaluate the linear baseline on the synthetic multilingual/code-mixed split:

```powershell
python scripts/run_pipeline.py --config configs/default.yaml --stage baseline
```

Launch the interactive explainer:

```powershell
streamlit run app/streamlit_app.py
```

Run unit tests:

```powershell
python -m pytest tests -q
```

## Full research pipeline

This stage downloads Hugging Face models (`xlm-roberta-base`, optional CLIP, `google/flan-t5-small`). Use a GPU if possible.

```powershell
python scripts/run_pipeline.py --config configs/default.yaml --stage classifier
python scripts/run_pipeline.py --config configs/default.yaml --stage dpo
```

CPU-friendly encoder:

```powershell
python scripts/run_pipeline.py --config configs/cpu.yaml --stage classifier
```

Enable vision fusion by setting `model.use_vision: true` in the YAML config and providing `image_path` on each JSONL record.

## Data format

Each line of JSONL:

```json
{
  "id": "tr_001",
  "text": "...",
  "label": 1,
  "language": "hi",
  "code_mixed": true,
  "context": "reply thread",
  "image_path": null,
  "chosen_explanation": "HATE. ...",
  "rejected_explanation": "NON_HATE. ..."
}
```

`data/synthetic/` is a small illustrative set for wiring the pipeline. For the paper, convert public datasets (HateXplain, HASOC, MMHS150K, or a code-mixed corpus you have license to use):

```powershell
python scripts/prepare_public_data.py --input path\to\raw.jsonl --output data\processed\train.jsonl
```

Then point `configs/default.yaml` at the processed files.

**Do not** scrape private accounts. Use licensed research datasets and follow platform terms of service.

## Suggested public datasets

- [HateXplain](https://github.com/punyajoy/HateXplain) — labels plus human rationales (ideal for XAI and DPO preferences)
- HASOC (Hindi/English/code-mix)
- OLID / SOLID
- MMHS150K or Hateful Memes (multimodal)

## Paper and slides

- Compile the ACL draft after installing [ACL style files](https://github.com/acl-org/acl-style-files):

```powershell
cd paper
pdflatex acl_paper.tex
bibtex acl_paper
pdflatex acl_paper.tex
pdflatex acl_paper.tex
```

If `acl.sty` is missing, the same source still compiles as a two-column article via `paper/acl_paper.tex` fallback packages.

- Open `slides/presentation.html` in a browser for the project defense deck.

## Ethics

This code is for research and assisted moderation. It should not be used as a sole automated punishment system. Hate-speech labels are culturally contested; we report group-wise error rates and keep a human in the loop. The synthetic examples are minimal academic illustrations, not a production training set.

## Citation

See `paper/acl_paper.tex` and `paper/references.bib`.
