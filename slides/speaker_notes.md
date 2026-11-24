# HateXAI — Milestone 1 speaker notes

**Deck:** `slides/milestone1.html` (primary) · `slides/HateXAI_Milestone1.pptx` (same content, editable)
**Timing:** 9 slides × ~45–55 s ≈ **7–8 minutes**, leaving 2–3 minutes for questions.
**Navigation (HTML):** `→` / `Space` next · `←` back · `Home` / `End` jump · `F` full screen.

Three rules for the whole talk:

1. Say the implementation status out loud whenever a component appears (“this one is trained and measured”, “this one is
   implemented but not trained”).
2. Never quote a number without its sample size in the same breath.
3. If you do not know something, say so — do not fill the gap.

---

## Slide 1 — Title and research overview · ~40 s

**Say:** “This is HateXAI, my project under Dr. Pradeepika Verma. The title is *Hate Speech Detection through Explainable AI
and Reward-Optimized LLMs*. The one-line objective is on the slide: model a post as `x = {text, image, context}`, decide
whether it is hate, justify the decision with evidence a moderator can audit, and send genuinely ambiguous cases to a human
instead of forcing a binary answer.”

**Be upfront immediately:** “The repository distinguishes three states, and I will use them all through this talk: one
system is trained and measured, three are implemented but never trained, and one is an unevaluated extension. I will label
each one as I go.”

**If asked why the student name is missing:** it is not recorded anywhere in the repository; add it before submitting.

---

## Slide 2 — Problem domain and motivation · ~55 s

**Say:** “Hate speech detection is not a profanity filter — it is identity-directed abuse or dehumanization. And the two
error types are not symmetric: a false negative leaves a targeted group exposed, while a false positive silences the dialect
or reclaimed language of the very communities a moderator is trying to protect.”

**Walk the two examples — this is the strongest 30 seconds of the talk.** Both are real records from the project's probe
set. Both contain the same high-risk word as genuine hate, both are gold non-hate, and the current model gets both wrong:
P(hate) = 0.58 on the quoted one and 0.64 on the counterspeech one.

**Say:** “A bag-of-words model has no representation of *who is speaking*, *about whom*, or *with what intent*, so the
trigger fires either way. This is the use-versus-mention distinction: systems that cannot tell them apart censor
counterspeech. That is exactly why the application exposes a `needs_review` outcome.”

---

## Slide 3 — Existing systems and the research gap · ~55 s

**Say:** “One row per family of approaches. I want to be explicit: every technique here already exists and works well for
its own task. Lexicons and classical models are fast and reproducible but see only surface cues. Transformers transfer
cross-lingually but need licensed data, GPU time and per-language tuning, and they are opaque. HateXplain gives human
rationales. LIME and SHAP give readable local attributions. RLHF and DPO align generation with preference. Multimodal work
catches hate carried by the image.”

**Then point at the two boxes:** “So the gap is not ‘no existing system can do this’. The gap is the combination and the
audit: explanations that are *measured* rather than illustrated, reward optimization applied to *justifications* instead of
labels, honest reporting of hard negatives, and a deferral outcome instead of a forced binary.”

---

## Slide 4 — Problem statement and objectives · ~50 s

**Say:** “Formally, a post is `x = {T, V, C}`: text possibly code-mixed, optional visual content, and context taken only
from available metadata — nothing invented. The research formulation is binary, `y ∈ {0,1}`, so metrics stay comparable
with published work.”

**Then the second box:** “The deployed decision adds an abstention region and emits `hate`, `non-hate` or `needs_review`
under an explicit error-cost model. I am showing both deliberately: prior work reports two classes, but a real moderation
queue needs a route to human review.”

**Objectives:** say the six in two breaths — reproducible baseline, multilingual and code-mixed classification,
interpretable evidence-based predictions, reward-guided rationale optimization, evaluation beyond accuracy (error types,
calibration, subgroups), and multimodal only where data and a trained model exist.

---

## Slide 5 — Methodology and architecture · ~55 s

Point at one lane at a time; do not read the boxes.

**Served lane (green):** “This is what actually runs. One shared canonical input string, word and character n-grams,
logistic regression, Platt calibration, then a cost-weighted threshold chosen on validation data only — never on test — and
a three-way output.”

**Middle lane (cyan):** “Attribution and its evaluation share the same masking mechanism, so the explanation cannot drift
from the faithfulness number shown next to it. Then the FastAPI service and the dashboard, with feedback stored as a review
queue rather than an automatic label.”

**Right lane (amber, dashed):** “This is a separate training-time pathway — the transformer, the CLIP fusion and the DPO
rationale alignment. It is implemented but never trained, and it is deliberately not wired into serving. Every one of them
is reported as `available: false`, never simulated.”

**Close on the grey box:** “The data is 36 synthetic records — 24 train, 6 validation, 6 test — plus 36 author-built
adversarial probes, 5 of which I flagged as genuinely ambiguous. That is enough to prove the software works. It is not
enough to claim scientific performance.”

---

## Slide 6 — Algorithms and why they are relevant · ~60 s

For each row give role → why → limitation, in one sentence each.

**Say also, explicitly:** “Two things I will not claim. TF-IDF does not model context the way a transformer does — it has no
word order and no notion of who is speaking. And highlighted tokens do not prove an author's intent; attribution shows what
moved the model, not what the author meant.”

**On calibration:** “Calibration does not change ranking — AUROC is identical — but it drops the Brier score from 0.202 to
0.087, and that is what makes an abstention band meaningful rather than arbitrary.”

**On the DPO row:** “This is the reward-optimization component, and it is not trained. It is also the riskiest part: if the
rejected explanations are weak, the policy learns to reward verbosity or fluent-but-unfaithful text. The objective is
unit-tested after I found and fixed a misaligned encoder–decoder log-probability shift.”

---

## Slide 7 — Feasibility, resources and current implementation · ~55 s

**Say:** “Feasibility is the part I can actually demonstrate. The pipeline runs end to end, the FastAPI service has eight
endpoints and loads the model once per worker, the dashboard has six screens and was verified in a browser, and 40 tests
pass on CPU in about 21 seconds. One command regenerates every number in this deck, with a run manifest recording seed,
device and data paths.”

**Implemented, not trained:** “The transformer, vision fusion and DPO trainer exist in code with no checkpoint. They are
reported as unavailable in the API and dashboard rather than filled with an estimate.”

**Blockers:** “No GPU, no licensed corpus, no image data, and no LaTeX toolchain, so the paper is not compiled here.”

**Data table:** read the counts out loud and note that they were counted from disk; an earlier internal figure of 30
training records was stale — the verified count is 24.

**Why still feasible:** “The engineering risk is retired. What remains is data and training, not architecture. Compute
needs are bounded and known.”

---

## Slide 8 — Evaluation plan and preliminary diagnostics · ~60 s

**Lead with the plan, not the numbers.** Six aspects: classification metrics with separate FP and FN rates; calibration and
abstention; robustness on counterexamples; subgroup reporting with a minimum-support rule, so a group with two examples is
reported as insufficient rather than scored; explanation faithfulness and stability; ablations plus three-seed statistics.

**Then the amber box:** “These are preliminary engineering diagnostics, not results. The test split has 6 items, so one
flipped example moves accuracy by 16.7 points. Cross-validation covers 24 training records. The 36-item probe set is
author-constructed, not a public benchmark.”

**Two findings worth stating:**
- “At the operating point implied by the cost model — a five-to-one penalty on false negatives — probe accuracy is 0.639
  with a 0.50 false-positive rate. Recall is bought with false positives, and that is a policy choice with real
  consequences.”
- “Train-only cross-validation selects the union of features, but held-out probes slightly prefer the word-only model.
  With this little data the selection signal and the held-out signal disagree — that disagreement is itself a finding.”

**Close the slide:** “The plan is to run exactly this protocol on a licensed corpus so the comparison table has measured
values for all four systems, or an explicit reason why a system is absent.”

---

## Slide 9 — Conclusion, limitations and next steps · ~50 s

**Say:** “The measured contribution is deliberately narrow: a calibrated CPU baseline with a cost-weighted three-way
decision, an eight-configuration ablation selected by train-only cross-validation, faithfulness that is measured rather
than illustrated, and an adversarial probe set that exposes systematic failure on quoted hate and counterspeech.”

**Limitations, without hedging:** the numbers come from 36 hand-written records; no transformer, vision or DPO system is
trained or evaluated; the false-negative-averse operating point produces a 0.50 false-positive rate; calibration is fitted
on too few examples to be reliable; attribution is not evidence of intent and the labels themselves are contested.

**Next steps:** more licensed data with documented provenance, a leakage-aware frozen baseline protocol, transformer
training and comparison, DPO on real reviewed preference pairs with a reward audit, explainability and fairness validation
per subgroup, multimodal only if data and compute arrive, then the full experimental analysis and paper.

**Finish:** “Everything in this deck is reproducible with two commands: the experiment runner and the test suite.”
Then invite questions — the two you should expect are “when does the transformer get trained?” and “who sets the costs in
the threshold model?”.

---

## Anticipated questions

| Question | Short answer |
| --- | --- |
| “Is the accuracy of 1.000 real?” | No — it is six test items, one flip per 16.7 points. Report it only with that caveat. |
| “Why not start with XLM-R?” | No GPU, no licensed corpus, and the audit comes first. The pipeline and evaluation harness had to exist before any model could be compared fairly. |
| “Is the reward optimization real?” | The DPO objective is implemented and unit-tested, but it is not trained, so no reward result is claimed. The previously served reward score was self-referential and was removed. |
| “What is the cost model based on?” | A policy assumption — five-to-one against false negatives — not a measured utility. Slide 8 reports what that assumption does to the false-positive rate. |
| “Why an abstention class?” | Because contested posts exist and a forced binary label hides them; abstentions are counted and reported. |
| “Does any of this say the highlighted words prove intent?” | No. Attribution shows what moved the model; it is not evidence of intent. |
