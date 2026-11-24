# HateXAI — Hate Speech Detection through Explainable AI and Reward-Optimized LLMs

Bennett University project under **Dr. Pradeepika Verma** (`pradeepika.verma@bennett.edu.in`, N2-WS-062).

A multilingual, explainable, optionally multimodal hate-speech detection prototype. A post is modelled as
`x_i = {T_i, V_i, C_i}` → `y_i ∈ {hate, non_hate, needs_review}`, where `T_i` is text (including
code-mixed language), `V_i` is optional image/meme content, and `C_i` is context. Predictions carry
calibrated probabilities, token-level attributions with measured faithfulness, and an explicit
abstention outcome for ambiguous inputs.

> **Read this first.** The repository is an honest research prototype. Start with [PROJECT_AUDIT.md](PROJECT_AUDIT.md),
> which records exactly what is implemented, what was broken, and what is deliberately **not** claimed.
> Nothing in this README is a benchmark result unless a command is given that reproduces it.

---

## Status at a glance

| Component | State |
| --- | --- |
| Preprocessing, calibrated TF-IDF baseline, robustness + faithfulness evaluation | **Working and reproducible** |
| FastAPI backend (`/api/predict`, `/api/explain`, `/api/evaluation`, `/api/feedback`, …) | **Working, tested** |
| HTML/CSS/JS dashboard (detection, explainability, evaluation, feedback, research) | **Working, tested** |
| XLM-R / DistilBERT transformer classifier | **Implemented; no checkpoint trained in this environment** |
| DPO rationale alignment (Flan-T5) | **Implemented and unit-tested; not trained here** |
| Vision / meme analysis (CLIP fusion) | **Implemented but unevaluated; image inputs are rejected, never faked** |

The shipped, runnable model is the **calibrated word+char TF-IDF baseline**. Everything else is reported as
unavailable rather than simulated.

---

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; use Activate.ps1 in PowerShell
pip install -e .
pip install -r requirements.txt
```

### 1. Generate reproducible evaluation artifacts

```bash
python experiments/run_experiments.py --config configs/default.yaml
```

This writes `outputs/eval/*.json` (metrics, calibration, operating point, probes, ablation, faithfulness,
comparison). The service trains the baseline if no checkpoint exists.

### 2. Run the backend + dashboard

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. The frontend is served from the same origin as the API, so no CORS
configuration is needed in the common case. Interactive API docs are at `/docs`.

### 3. Run the tests

```bash
python -m pytest tests -q     # 40 tests
```

---

## Architecture

```
text + context ──► canonical fusion ──► TF-IDF (word+char) ──► logistic regression ──► Platt calibration
                                              │
                                              ├──► LIME / occlusion attribution ──► token evidence + faithfulness
                                              └──► cost-weighted threshold ──► hate | non_hate | needs_review

(optional, implemented, not trained here)
XLM-R text encoder + frozen CLIP vision ──► gated fusion ──► classifier
Flan-T5 explanation policy ──► DPO over chosen/rejected rationales
```

Key design decisions, and why:

- **One canonical input string.** Baseline and transformer both call `hatexai.data.preprocess.fuse_text`, so
  they see identical features. Previously the transformer also received `Language:`/`CodeMixed:` tokens the
  baseline did not, which confounded comparison (audit §3.6).
- **Calibrated probabilities, then a cost-weighted threshold.** Calibration does not change ranking (AUROC is
  unchanged) but makes the abstention band meaningful.
- **Three-way output.** `needs_review` is a first-class outcome; abstentions are counted, not hidden.
- **Attribution and its evaluation share one masking mechanism**, so an explanation cannot drift from the
  faithfulness number reported next to it.

---

## Measured results (reproducible)

Run `python experiments/run_experiments.py` to regenerate. Generated on the bundled diagnostic data.

**Ablation (selection by train-only 3-fold CV; test/probe are held-out reports):**

| Feature set | Calibrated | CV macro-F1 | Test acc. | Brier | ECE | Probe acc. | Probe (unambiguous) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| char (legacy) | no | 0.706 | 0.667 | 0.236 | 0.149 | 0.639 | 0.710 |
| word | no | 0.747 | 1.000 | 0.213 | 0.462 | 0.750 | 0.774 |
| word | yes | 0.790 | 1.000 | 0.079 | 0.265 | 0.806 | 0.839 |
| union | no | 0.746 | 1.000 | 0.202 | 0.448 | 0.750 | 0.806 |
| **union** | **yes** | **0.831** | 1.000 | 0.087 | 0.271 | 0.750 | 0.806 |

**Robustness on the 36-item adversarial probe set:** at the 0.5 decision threshold, overall accuracy **0.750** (unambiguous
subset **0.806**); at the cost-weighted operating point actually served (θ = 0.394, fn:fp = 5:1) it is **0.639** with
FPR **0.500** and FNR **0.083** — recall is bought with false positives. The model still fails systematically on **quoted hate**, **counterspeech**, and
**neutral identity mentions** that contain abusive words — see `outputs/eval/probes.json`.

**Honest reading of these numbers:**

- The test split has only **6 examples**, so its accuracy moves 16.7 points per flipped item. It is not evidence.
- Cross-validation is over **24 training examples**. The ablation is directional, not conclusive.
- The probe set is **author-constructed**, deliberately adversarial, and not a public benchmark. It is useful
  because it exposes failure modes, and it is labelled as contested where the label itself is debatable.
- No transformer or DPO result is reported because neither was trained here.

**Baseline reference (pre-improvement):** the original char-only baseline scored acc 0.667 / AUROC 0.778 on the
test split and 0.639 on the probes; the audit documents why.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness and model-load status |
| GET | `/api/model-info` | Architecture, languages, modalities, limitations, responsible use |
| POST | `/api/predict` | Decision + calibrated probabilities + uncertainty |
| POST | `/api/explain` | `/api/predict` plus token attributions and faithfulness (`method`: `lime` or `occlusion`) |
| POST | `/api/vision-check` | Validates an image upload (type, 5 MB cap, decodability) and refuses to fake analysis |
| GET | `/api/evaluation` | All evaluation artifacts, with an explicit `missing` list |
| POST | `/api/feedback` | Submit a report → stored as a **review candidate** |
| GET | `/api/feedback` | Review queue (latest status merged from the review log) |
| GET | `/api/feedback/summary` | Counts by review status |
| POST | `/api/feedback/{id}/review` | Record a human review decision |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/explain \
  -H "Content-Type: application/json" \
  -d '{"text":"These people are vermin and should leave.","method":"occlusion"}'
```

Response fields include `label`, `probability_hate`, `confidence`, `uncertainty`, and an `explanation` object
with `tokens`, `scores`, `top_evidence`, `rationale`, `faithfulness`, and a `disclaimer`.

Backend guarantees: input length limits enforced at the schema boundary; the model is loaded **once per
worker**; inference errors return a clean 500; CORS is an explicit allow-list; a per-client rate limit applies;
uploads are temp-file cleaned; no secrets are read from or written to the repository.

---

## Frontend

Plain HTML5 + CSS3 + vanilla JavaScript in `frontend/` — no framework, no build step, no Streamlit.

- `index.html` — landing page with an animated neural-network hero and feature cards
- `analyze.html` — detection workspace: input, char counter, examples, results with token highlighting,
  probability bars, uncertainty indicator, and faithfulness tiles
- `dashboard.html` — evaluation dashboard: metrics, confusion matrix, decision policy, system comparison,
  ablation, robustness, faithfulness
- `feedback.html` — report form + human review queue
- `research.html` — model, languages, datasets, methodology, limitations

Accessibility and motion: keyboard-navigable, `focus-visible` outlines, and a full
`prefers-reduced-motion` path. The dashboard shows skeleton loaders and explicit empty/unavailable states
instead of invented numbers.

The legacy `app/streamlit_app.py` is retained but **deprecated**; Streamlit is no longer a dependency.

---

## Training the transformer and DPO (optional)

These require model downloads and, realistically, a GPU.

```bash
python scripts/run_pipeline.py --config configs/cpu.yaml --stage classifier   # DistilBERT-multilingual
python scripts/run_pipeline.py --config configs/default.yaml --stage dpo      # Flan-T5 + DPO
```

Transformer training now uses **class-weighted cross-entropy** (inverse frequency, normalised to mean 1),
best-checkpoint selection, and early stopping. The DPO sequence log-probability uses the correct
encoder–decoder shift; a regression test pins it.

**Hardware guidance:** the shipped baseline and the whole test suite run on CPU in well under a minute.
`configs/cpu.yaml` (DistilBERT-multilingual + Flan-T5-small) needs roughly 8 GB RAM and no GPU but will be
slow; `configs/default.yaml` (XLM-R-base + CLIP) is intended for a ≥12 GB GPU. The CPU configs are for
smoke-testing the code path, not for producing paper numbers.

---

## Data

| Asset | Location | Notes |
| --- | --- | --- |
| Synthetic wiring set | `data/synthetic/` | 24 train / 6 val / 6 test (36 records, counted from disk). Illustrative only; **not a benchmark** |
| Adversarial probes | `data/probes/hard_cases.jsonl` | 36 author-constructed diagnostic cases, with an `ambiguous` flag |

Convert licensed public corpora to the project schema:

```bash
python scripts/prepare_public_data.py --input path/to/raw.jsonl --output data/processed/train.jsonl
```

Recommended corpora: [HateXplain](https://github.com/punyajoy/HateXplain) (labels + rationales),
HASOC (Hindi/English/code-mix), OLID/SOLID, MMHS150K / Hateful Memes (multimodal).
**Do not** scrape private accounts; follow platform terms and dataset licences. Processed data is git-ignored.

JSONL record format:

```json
{"id":"tr_001","text":"...","label":1,"language":"hi","code_mixed":true,
 "context":"reply thread","image_path":null,
 "chosen_explanation":"HATE. ...","rejected_explanation":"NON_HATE. ..."}
```

---

## Repository layout

```
PROJECT_AUDIT.md          audit: what exists, what was broken, what is claimed
src/hatexai/              library: data, baselines, models, train, eval, reward, xai
backend/                  FastAPI app, schemas, model registry, feedback store
frontend/                 HTML/CSS/JS dashboard
experiments/              reproducible experiment runner → outputs/eval/*.json
configs/                  default (GPU) and cpu configs, with baseline + decision-policy settings
tests/                    40 tests: metrics, thresholds, robustness, XAI, DPO alignment, API
paper/                    ACL draft and bibliography
slides/                   defense deck
```

---

## Limitations and responsible use

- The shipped model is trained on a 30-record synthetic set. It is a software demonstration, not a deployable
  classifier, and its probabilities on the bundled test split are compressed near 0.5.
- No vision or meme capability is trained or evaluated; image inputs are validated and rejected.
- Attribution is **not** evidence of intent, and generated rationales are displayed separately from
  model-derived evidence.
- Hate-speech labels are culturally contested; the probe set marks contested cases explicitly.
- Feedback is a review queue, never an automatic training signal, and never enters the evaluation split.
- Do not use this as the sole basis for punitive action against a person or account. Keep a human in the loop.

## Citation

See `paper/acl_paper.tex` and `paper/references.bib`.
