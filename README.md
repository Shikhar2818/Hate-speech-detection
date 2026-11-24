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
| TF-IDF on the public HateXplain benchmark | **Trained and evaluated on held-out English posts** |
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
python -m pytest tests -q
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

### Public HateXplain benchmark

```bash
python scripts/run_pipeline.py --config configs/hatexplain_baseline.yaml --stage baseline
```

Run the `hatexplain` data preparation command below first. Training uses the existing word and
character TF-IDF model with logistic regression and three group-aware calibration folds on CPU.
The test split is evaluated after the checkpoint is saved and the review band is selected on validation.

| Split | N | Macro F1 | Macro precision | Macro recall | Accuracy | FPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 1,920 | 0.795 | 0.822 | 0.779 | 0.836 | 0.072 | 0.369 |
| Test | 1,924 | 0.792 | 0.813 | 0.778 | 0.832 | 0.082 | 0.362 |

At the binary decision threshold, the test confusion matrix was `[[1221, 109], [215, 379]]`,
with rows for gold non-hate and hate. AUROC was 0.877, Brier score was 0.125 and ECE was 0.020.
The validation-selected review policy routed 284 test posts (14.8%) to human review. All test
records have the English release label. Evaluation across languages requires additional licensed data.
The checkpoint and JSON reports are written under `outputs/hatexplain-baseline-v1/`.

### Synthetic diagnostics

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

Use the release-aware profiles in [configs/public_data.yaml](configs/public_data.yaml). Run these commands
from the repository root. Keep authorized source files and generated data under the ignored
`data/raw/` and `data/processed/` directories.

| Profile | Required source files | Label policy |
| --- | --- | --- |
| `hatexplain` | `Data/dataset.json`, `Data/post_id_divisions.json`, and `LICENSE` from the [pinned HateXplain release](https://github.com/hate-alert/HateXplain/tree/01d742279dac941981f53806154481c0e15ee686), saved under `data/raw/hatexplain/` | Take the majority among the original three classes, then map only `hatespeech` to 1. Preserve the original votes, tokens and rationale masks. |
| `multilingual` | The HateXplain files and authorized [HASOC 2021](https://hasocfire.github.io/hasoc/2021/dataset.html) English and Hindi subtask 1 train/test exports with `task_1` and fine labels in `task_2`. Place the exports at the paths in the profile or set the paths in a local config. | Map `HATE` to 1 and `NOT`, `OFFN` and `PRFN` to 0. Exclude coarse-only `HOF` because it combines hate, offensive speech and profanity. |
| `hateful-memes` | Authorized [Hateful Memes](https://www.drivendata.org/competitions/64/hateful-memes/) Phase 2 archive with `train.jsonl`, `dev_unseen.jsonl`, `test_unseen.jsonl` and `img/` | Keep official binary labels and preserve unlabeled test IDs. |

```bash
python scripts/prepare_public_data.py --profile hatexplain --output data/processed/hatexplain-v1
python scripts/prepare_public_data.py --profile multilingual --authorized-sources --output data/processed/multilingual-v1
python scripts/prepare_public_data.py --profile hateful-memes --authorized-sources --output data/processed/hateful-memes-v1
```

The output directory is versioned by the caller. Each build writes split JSONL files, an exclusion log
and a manifest with source revisions, input/output SHA-256 digests, and label, language, code-mix,
image and duplicate group counts. A meme test set without labels is written to `test_unlabeled.jsonl`.
Image paths refer to the local dataset directory from the profile. Run image consumers from the same
working directory or configure an absolute dataset root.

Official test IDs stay in test. Exact text matches, near text matches at character-trigram Jaccard
similarity of at least 0.85 for strings of 20 or more characters, and exact or perceptually similar
images form groups across inputs. The build removes overlapping rows from train or validation in
favor of the higher-priority split. It derives HASOC validation from eligible training groups.
The rule can remove shared meme captions or related images from development data. The manifest
records these exclusions for reporting. The `language` field refers to the release, and the
`code_mixed` flag detects Latin/Indic script mixing. Romanized-only code mixing requires independent
annotation. The broad-label HASOC conversational track is outside the strict hate label profile.

The HateXplain build was verified on the official data with 15,335 train, 1,920 validation and 1,924
test rows. HASOC and Hateful Memes require authorized local archives for end-to-end verification.
The original `--input` converter remains available for legacy files. Use the profile commands for
benchmark preparation.

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

- The default demo model is trained on 24 synthetic examples. Its probabilities on the 6-record
  demo test split are compressed near 0.5.
- No vision or meme capability is trained or evaluated; image inputs are validated and rejected.
- Attribution is **not** evidence of intent, and generated rationales are displayed separately from
  model-derived evidence.
- Hate-speech labels are culturally contested; the probe set marks contested cases explicitly.
- Feedback is a review queue, never an automatic training signal, and never enters the evaluation split.
- Do not use this as the sole basis for punitive action against a person or account. Keep a human in the loop.

## Citation

See `paper/acl_paper.tex` and `paper/references.bib`.
