# PROJECT_AUDIT — Hate Speech Detection through Explainable AI and Reward-Optimized LLMs

## English and Hinglish decision check

Scored 20 matched English and Hinglish diagnostic pairs and replayed all 40 inputs through both
explanation methods. The two methods produced identical scores and moderation decisions.
The active profile flags hate at 24.19% and routes scores above 14.19% and below 24.19% to review.
The diagnostic set exposed missed exclusionary posts and false flags on neutral identity mentions
and counterspeech. These hand-written examples are troubleshooting cases.

On the checksummed held-out splits, the active review policy cleared 40 of 594 labeled English hate
posts and 21 of 806 labeled Hinglish hate posts. It flagged 357 of 1,330 English non-hate posts and
561 of 1,029 Hinglish non-hate posts. Review outcomes are counted separately in the local diagnostic
report under `outputs/app-check/bilingual-diagnostics/`. The Analyze page now displays a single
hate score and the moderation cutoffs alongside the verdict.

## Local multilingual, context, and meme experiments (2026-11-24)

The `english_hinglish.yaml` profile serves the calibrated TF-IDF model trained on HateXplain and
HiFoX Task A. Held-out reports are under `outputs/english-hinglish-v1/eval/`. HiFoX is available for
local research. Its authors require permission for publication, so the paper keeps the earlier
English-only results. These English and Hinglish cohorts come from different corpora.

The CADD experiment fits the same model family twice, with and without the parent post's title
and body. Shared parent text and duplicate comments are kept out of lower-priority splits.
On 4,901 held-out comments, macro F1 was 0.617 for comments alone and 0.621 with parent context.
The app's context field belongs to the text profile. CADD results cover the separate offline models.
[CADD release and terms](https://github.com/nlpcl-lab/CADD_dataset)

The English Multi3Hate pilot fits three classifiers using caption TF-IDF, frozen TinyCLIP image
features, and their combination. On 57 held-out memes with separate templates, macro F1 was
0.553, 0.474, and 0.683 respectively. The upload endpoint requires a manual caption. The nine
region scores measure the result of covering each image region and rescoring the same caption.
The pilot is for local noncommercial research. [Multi3Hate dataset card](https://huggingface.co/datasets/MinhDucBui/Multi3Hate)

A separate validation-only threshold search minimizes the false-positive-rate gap between the
English and Hinglish cohorts subject to at most three percentage points of validation recall loss
per cohort. The saved test report includes both false-positive and false-negative rates.
This threshold remains an offline comparison. Human-reviewed pairs of moderation rationales are
still needed for the Flan-T5 DPO experiment.

## Project checklist review (2026-11-24)

| Requirement | Current coverage |
| --- | --- |
| Data and preprocessing | Pinned, deduplicated English HateXplain splits with source labels and rationale masks. Text normalization preserves punctuation and emoji. Authorized multilingual and meme datasets are pending. Their context coverage and image preprocessing require verification. |
| Detection | The trained TF-IDF baseline serves text and optional supplied context. Measuring the effect of conversation context requires labeled threads. The transformer and vision fusion code still need trained, evaluated checkpoints. |
| Explanations | LIME, occlusion, attribution summaries, and held-out faithfulness with results split by prediction correctness. Image explanations and model-generated rationales depend on their respective trained models. |
| Feedback and reward optimization | Reviewers can save a corrected hate or non-hate label alongside the original submission. DPO code is tested. Training and comparison require an independent human-reviewed preference corpus. |
| Fairness and evaluation | Held-out classification metrics, feature and calibration comparisons, plus English error breakdowns by annotated target. Cross-language and code-mixed fairness, bias reduction, and multimodal comparisons still need experiments. Target groups describe posts and can overlap. |
| Web app | Text detection, image file validation, feedback review, and analytics. Live request counts reset when the server restarts and contain no post text. |

Reproduce the correct-versus-incorrect attribution and annotated-target reports with
`python experiments/evaluate_public_baseline.py --config configs/hatexplain_baseline.yaml --hatexplain-annotations data/raw/hatexplain/dataset.json`.

## Application verification (2026-11-24)

Ran the existing FastAPI application and all five frontend pages against the English HateXplain
checkpoint. Fixed LIME for punctuation and emoji, aligned token attributions with the original
input, and used P(hate) as the common direction for LIME and occlusion. The decision-only API
now scores text directly. Over ten warm local requests, median latency for the same input fell
from 81 ms to 13 ms.

Feedback attaches the input that was actually scored. Clearing an analysis resets its counter and
discards any late response. The analysis page now exposes image file validation, including a full
decode and a check that the contents match the declared type. The dashboard tables scroll within
their cards on mobile. The dashboard reports binary and review-policy metrics separately.

Rescoring the 1,924 held-out posts reproduced 0.792 macro F1 and 83.2% accuracy with the binary
cutoff. The served policy routed 284 posts to review. Among automatically decided posts,
accuracy was 75.9% and the false-positive rate was 31.5% among decided negatives. The app
remains a human-assisted English research prototype. Vision classification remains unavailable.

## Dataset integration update (2026-11-24)

This section records the current dataset state. The audit below is the historical review of `58a9ec8`.
The current tree already has the FastAPI dashboard, TF-IDF demo checkpoint support, and evaluation
implementations. The older audit's duplicate-check item was marked complete before the corresponding
code existed in the checked-in tree.

The release-aware preparation command verified the official HateXplain files against pinned
SHA-256 digests. Of 20,148 source posts, 919 had no majority under the original three-way annotation
scheme and were outside the published splits. The overlap checks removed 24 train and 2 validation
rows from cross-split duplicate groups and 24 exact train duplicates. The final counts are
15,335 train, 1,920 validation and 1,924 test. All official test IDs remain in the test split.
The generated manifest and exclusion log are under `data/processed/hatexplain-v1/`.

The HASOC 2021 and Hateful Memes readers have fixture coverage. End-to-end verification requires
authorized HASOC English/Hindi exports with fine labels and a Hateful Memes Phase 2 archive.
HASOC's mixed-script flag is heuristic and requires row-level annotations to assess romanized-only
code mixing. Meme labels are needed for local test metrics. The host has CPU-only PyTorch.

## Public baseline update (2026-11-24)

The existing calibrated word/character TF-IDF classifier was fitted on the prepared HateXplain train
split. Three group-aware calibration folds kept duplicate groups together. The checkpoint was saved
and the review policy chosen on validation before scoring the official English test split. On
1,924 held-out posts, macro F1 was 0.792, macro precision 0.813, macro recall 0.778 and accuracy 0.832.
The binary false-positive rate was 0.082 and the false-negative rate was 0.362. The validation-selected
policy sent 284 test posts (14.8%) to review.

Reproduce with `python scripts/run_pipeline.py --config configs/hatexplain_baseline.yaml --stage baseline`
after building the HateXplain profile. The checkpoint is
`outputs/hatexplain-baseline-v1/checkpoints/linear_baseline.joblib`. The corresponding `eval/`
directory contains the training record, classification metrics and review-policy results. This run
supports an English baseline estimate. Multilingual performance, vision and DPO remain pending.

The FastAPI app can serve the benchmark baseline with `HATEXAI_CONFIG=configs/hatexplain_baseline.yaml`.
The service checks the checkpoint and threshold against the training record. The API provides the
review outcome and matching evaluation artifacts, and the frontend uses that outcome for display
and feedback submission.

The public evaluation command scored all 1,924 test posts by original HateXplain label and
compared occlusion evidence with same-size random masks on 1,812 posts longer than five tokens.
False positives occurred in 43 of 782 normal posts and 66 of 548 offensive posts. The mean
probability drop after removing the five selected tokens was 0.096 versus 0.031 under random
deletion. Mean retained probability was 0.832 versus 0.757. These are perturbation measurements
for this checkpoint and do not assess generated rationales or intent.

The fixed public feature ablation produced test macro F1 of 0.723 for character features, 0.792
for word features, 0.793 for their uncalibrated union, and 0.792 for the calibrated union. The
group-calibrated ensemble reduced ECE from 0.104 to 0.020. The experiment compares the full
calibration procedure, which also changes the estimator from a single fit to a fold ensemble.

## DPO validation update (2026-11-24)

The historical recommendation in section 3.2 misidentified the T5 alignment. Passing `labels` to T5
already shifts the decoder inputs, so shifting output logits and labels again mis-scores the completion.
The trainer now uses the same score implementation as the explanation policy and a tiny T5 regression
test compares it directly with the framework loss. The trainer accepts a separate reviewed-pair file,
rejects empty or unreviewed input, and saves loss history alongside a completed checkpoint.
An independent human-reviewed corpus is still needed for a Flan-T5 research run.

---

**Audit date:** 2026-10-13
**Auditor role:** Senior ML / NLP / XAI / full-stack review
**Repository commit audited:** `58a9ec8` ("initial ready"), branch `main`
**Environment:** Windows, `.venv` with Python 3.13.9, torch 2.14.1, transformers 5.18.0, scikit-learn 1.9.1, no GPU assumed. `pytest`, `fastapi` were **not installed** at audit time.

This audit records what is *actually* in the repository versus what the README, paper draft, and config claim. Everything below was verified by reading the source and by running the code that can run in this environment.

---

## 1. Repository structure (as found)

```
app/streamlit_app.py              # only UI
configs/{default,cpu}.yaml
data/synthetic/{train,val,test}.jsonl   # 24 / 6 / 6 records
outputs/checkpoints/linear_baseline.joblib
outputs/eval/pipeline_summary.json
paper/{acl_paper.tex,references.bib}
scripts/{run_pipeline.py,prepare_public_data.py}
src/hatexai/
  data/{preprocess.py,dataset.py}
  baselines/linear.py
  models/{multimodal.py,llm_policy.py}
  train/{train_classifier.py,train_dpo.py}
  eval/{metrics.py,evaluate.py}
  reward/{scoring.py,dpo.py}
  xai/{attributions.py,explain.py}
  inference.py, utils.py
tests/{test_metrics.py,test_preprocess.py}   # 4 tests total
slides/presentation.html
```

Total first-party Python: **~1,242 lines**. There is **no** `frontend/`, **no** `backend/` (FastAPI), **no** `experiments/`, and **no** project-unique evaluation artifacts beyond the single baseline summary.

---

## 2. What is genuinely implemented

| Capability | Status | Evidence |
| --- | --- | --- |
| Text preprocessing (URL/user/hashtag normalization, script-mix detection) | **Real, working** | `data/preprocess.py`; covered by a passing test |
| Character n-gram TF-IDF + logistic-regression baseline | **Real, working, reproducible** | `baselines/linear.py`; I reran it (see §5) |
| Multimodal gated-fusion detector (XLM-R + frozen CLIP) | **Code complete, never successfully trained here** | `models/multimodal.py`; no checkpoint in `outputs/` |
| Attention-rollout + LIME attribution | **LIME real; attention implementation unverified** | `xai/attributions.py`, `xai/explain.py` |
| Integrated gradients | **Dead code** — defined but never called from any model or endpoint | `attributions.py::integrated_gradients_attribution` has zero call sites |
| SHAP | **Not implemented** (in `requirements.txt` only) | no import of `shap` anywhere |
| Heuristic reward | **Real but self-referential in serving** (see §3.1) | `reward/scoring.py`, used in `inference.py` |
| DPO on (chosen, rejected) rationales | **Code complete, never run; log-prob math is wrong** (see §3.2) | `train/train_dpo.py`, `reward/dpo.py` |
| Fairness report (macro-F1 + FPR by language) | **Real but minimal** | `eval/metrics.py::group_fairness` |
| Streamlit UI | **Real, working** | `app/streamlit_app.py` |
| Feedback collection loop | **Does not exist** | no endpoint, no storage |
| FastAPI backend / REST API | **Does not exist** | `fastapi` not even in `requirements.txt` |

---

## 3. Bugs and correctness defects (verified by reading the code)

### 3.1 The served "explanation reward" is meaningless (high severity)
`inference.py::LinearHateService.analyze` computes:

```python
reward = heuristic_reward(feats.normalized, rationale, gold_label=label, predicted_label=label)
```

`heuristic_reward`'s largest term is `correctness = 1.0 if predicted == gold_label`. Because both arguments are the *same variable* (`label` = the model's own prediction), the correctness term is mathematically pinned to `1.0` for every input. The Streamlit UI surfaces this as "Explanation reward", so users see a quality score that cannot ever vary with quality. This is a correctness bug, not a design choice.

### 3.2 DPO log-probabilities are misaligned (high severity)
`train/train_dpo.py::_sequence_logprob` gathers `outputs.logits` at the *unshifted* `labels` positions. For encoder–decoder models, `logits[t]` predicts `labels[t+1]`; the correct computation shifts logits and labels. As written, every sequence log-probability is computed against the wrong target tokens, so the DPO objective optimizes a quantity that is not the model's likelihood of the explanation. HuggingFace computes this shift internally for `loss`, which is why `ExplanationsPolicy.logprob_of_completion` and `_sequence_logprob` disagree.

### 3.3 Dead/broken log-probability helper (medium)
`models/llm_policy.py::logprob_of_completion` ends with `-outputs.loss * token_count / token_count`, which algebraically cancels to `-outputs.loss`. It is also never called. Either fix it to share one correct implementation with `_sequence_logprob`, or delete it.

### 3.4 Paper claims class-weighted cross-entropy; the code does not implement it (medium)
`paper/acl_paper.tex` states the classifier is trained with "class-weighted cross-entropy". `train/train_classifier.py` calls `nn.functional.cross_entropy(logits, labels)` with no `weight`. The linear baseline *does* pass `class_weight="balanced"`, so the two systems are also optimized for different class priors. Claim and code must be reconciled.

### 3.5 Non-standard JSON is written to disk (medium)
`compute_classification_metrics` may return `auroc = float("nan")`. `utils.write_jsonl` uses `json.dumps`, which emits the bare token `NaN`. That is invalid JSON per RFC 8259 and is rejected by JavaScript's `JSON.parse` — i.e. the evaluation dashboard the project now needs could not read the metrics files it is supposed to display.

### 3.6 Feature-space mismatch between baseline and transformer (medium)
`baselines/linear.py` trains on `normalized_text + context`.
`data/dataset.py::_fuse_text` trains the transformer on a string that additionally contains literal `Language: <x>` and `CodeMixed: <0|1>` fields. The two models therefore do not see the same inputs, and the transformer is handed an explicit metadata shortcut. Any baseline-vs-model comparison is currently confounded.

### 3.7 No duplicate / near-duplicate leakage checks (medium)
There is no deduplication or group-aware splitting. HateXplain (arguably the primary intended dataset) contains repeated and near-duplicated posts; random splits leak near-duplicates across train/test and inflate scores. The README recommends "stratified splits" (in the task brief) but nothing enforces it.

### 3.8 Multimodal explanation path is broken (medium)
`xai/explain.py::HateExplainer.predict_proba` calls the model with `input_ids`/`attention_mask` only. For a vision-enabled detector the explanation would be produced from a different forward pass than the prediction, so highlighted tokens would not explain the reported probability.

### 3.9 `attention_attribution` is fragile/unverifiable (low–medium)
It indexes `attentions[-1]` and averages heads by hand, assuming a specific tensor layout, and assumes the model always returns `attentions`. There is no test. In transformers 5.x this returns a tuple of per-layer tensors with a documented shape, but nothing validates it and failures would be silent.

### 3.10 Evaluation gaps (medium, matches Phase 4 requirements)
Missing entirely: confusion matrix, per-class F1, separate FP/FN *rates*, calibration / ECE, abstention or thresholding, out-of-distribution signalling, and language/code-mix-stratified reporting beyond macro-F1+FPR on groups with 1–2 examples.

---

## 4. Research-honesty problems

1. **Abstract overstates maturity.** The paper abstract says explanations "are further aligned using DPO", present tense, but DPO has never been run and its loss is misaligned (§3.2). There is no DPO checkpoint in the repo.
2. **Contribution (3)** claims "a DPO training recipe that treats (chosen, rejected) rationales as preferences" — the code exists but has no results, no logs, and no evaluation.
3. **Multimodal capability is claimed as a contribution** while `use_vision: false` in both configs and no image data exists. The task brief explicitly forbids claiming untrained multimodal ability; the paper should demote this to "implemented, unevaluated extension".
4. **"Reward-optimized"** in the title currently rests on a heuristic score that is, in serving, provably constant (§3.1).
5. `references.bib` contains at least one unused entry (`conneau2019xnli`) — minor, but a sign the bibliography predates the argument.
6. The bundled "dataset" is 36 hand-written illustrative records. That is fine for wiring, but it is not a benchmark, is not licensed data, and must never be presented as a demographic sample (the paper does say this — keep it).

---

## 5. Baseline results (measured, not fabricated)

I ran the pipeline in this environment:

```
$ .venv/Scripts/python.exe scripts/run_pipeline.py --config configs/default.yaml --stage baseline
linear baseline val:  acc 0.667  macro-F1 0.667  auroc 0.778   (n=6)
baseline test:        acc 0.667  macro-F1 0.667  auroc 0.778   (n=6)
```

These numbers are **reproducible** and are stored in `outputs/eval/pipeline_summary.json`. They are also **statistically meaningless**: n=6 test examples, so one flipped item moves accuracy by 16.7 points. No confidence interval is reported. The result validates engineering, not science — the paper already concedes this, and the audit confirms it.

**Not runnable here / not attempted at audit time:**
- `--stage classifier` (XLM-R/distilbert training): weights are download-reachable but require multi-hundred-MB downloads and CPU training; no checkpoint exists.
- `--stage dpo`: Flan-T5-small download + training; not run.
- No GPU, no licensed corpus, no image data. These are documented blockers, not planned work.

Existing tests: **4 tests passing** (`test_metrics.py`, `test_preprocess.py`) once `pytest` is installed. No tests exist for the dataset, model forward pass, DPO math, XAI, or any HTTP surface.

---

## 6. Prioritized improvement plan

Ordered by (impact × feasibility in the current environment). Items marked ✅ were implemented as part of this pass.

### P0 — correctness and honesty
1. ✅ Fix the self-referential reward in serving; make reward an *offline* diagnostic that requires a real label, and separate it from the served prediction payload.
2. ✅ Fix DPO sequence log-probability shifting; add a unit test that pins it against a hand-computed value.
3. ✅ Remove/replace the dead `logprob_of_completion`; delete the unreachable `multimodal` fusion claim or wire it correctly.
4. ✅ Make all persisted metrics strict JSON (no `NaN`); add a serialization test.
5. ✅ Reconcile the class-weighting claim: implement weighted CE for the transformer controlled by config.
6. ✅ Align the baseline and transformer feature space (single shared `fuse_text`).
7. ✅ Add duplicate / near-duplicate detection utilities and a leakage-aware split helper.
8. ✅ Correct the paper: present tense → "implemented, not yet evaluated"; move multimodal out of contributions into limitations/future work.

### P1 — evaluation quality (Phase 4)
9. ✅ Add confusion matrix, per-class P1/R1/F1, FP-rate, FN-rate, AUROC, calibrated ECE, and Brier score.
10. ✅ Add abstention / threshold selection driven by validation data, with a three-way outcome (hate / non-hate / human review).
11. ✅ Add a language- and code-mix-stratified report that refuses to report groups below a minimum support.
12. ✅ Add an author-constructed **adversarial diagnostic probe set** (quoted hate, counterspeech, reclaimed slurs, neutral identity mention, non-hateful profanity, sarcasm) and measure the baseline on it — a genuine robustness result rather than a manufactured one.

### P2 — real, tested explainability (Phase 3)
13. ✅ Wire integrated gradients to a real torch classifier path (not embedding-space dead code) and expose it through the API.
14. ✅ Add faithfulness metrics with tests: comprehensiveness and sufficiency (probability drop under masking / keeping only top-k).
15. ✅ Keep LIME as the default; make attention attribution defensive and tested.

### P3 — backend + frontend (Phases 5–6) — **mandatory**
16. ✅ Build a FastAPI backend with `/health`, `/api/model-info`, `/api/predict`, `/api/explain`, `/api/evaluation`, `/api/feedback`; model loaded once at startup; strict Pydantic validation; upload limits; CORS allow-list; temp-file cleanup.
17. ✅ Replace Streamlit with `frontend/index.html` + `css/styles.css` + `js/app.js` (landing, detection workspace, explainability, evaluation dashboard, feedback, research/model info). Keep `app/streamlit_app.py` in the tree but document it as deprecated.
18. ✅ Persist feedback as *review candidates*, never as trusted labels; expose a review status.

### P4 — experiments and packaging (Phase 7–8)
19. ✅ `experiments/` runner producing machine-readable results and a comparison table, with an explicit `unavailable` state when a system has not been trained.
20. ✅ Tests for the API surface (valid input, invalid input, missing modality, model failure).
21. ✅ Update README with setup, API docs, frontend instructions, hardware notes, and an explicit list of what is and is not demonstrated.

### Known blockers (documented, not worked around)
- No GPU; full XLM-R/CLIP/Flan-T5 training is out of scope for this machine.
- No licensed corpus (HateXplain/HASOC/MMHS150K) is present; only the 36-record synthetic wiring set. Any headline metric must wait for `scripts/prepare_public_data.py` to be run on licensed data.
- No image data → multimodal remains a documented, unevaluated extension.
