"""Build slides/HateXAI_Milestone1.pptx from the Milestone-1 content.

The deck is generated rather than hand-drawn so the numbers stay in sync with
``outputs/eval/*.json`` and the HTML deck (``slides/milestone1.html``).

Usage:  python scripts/build_milestone1_pptx.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides" / "HateXAI_Milestone1.pptx"

BG = RGBColor(0x0B, 0x12, 0x20)
PANEL = RGBColor(0x15, 0x22, 0x3B)
PANEL2 = RGBColor(0x1B, 0x2B, 0x48)
LINE = RGBColor(0x2A, 0x40, 0x66)
TEXT = RGBColor(0xE8, 0xEE, 0xFB)
MUTED = RGBColor(0x9F, 0xB0, 0xCC)
ACCENT = RGBColor(0x5E, 0xA9, 0xFF)
ACCENT2 = RGBColor(0xA7, 0x8B, 0xFA)
CYAN = RGBColor(0x4F, 0xD1, 0xE0)
GOOD = RGBColor(0x4A, 0xDE, 0x80)
WARN = RGBColor(0xFB, 0xBF, 0x24)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W, H = Inches(13.333), Inches(7.5)


def number(path: str, key: str, default=None):
    """Read one number from an evaluation artifact so slides cannot drift from it."""
    try:
        data = json.loads((ROOT / path).read_text(encoding="utf-8"))
    except Exception:
        return default
    for part in key.split("."):
        if isinstance(data, list):
            data = data[int(part)]
        elif isinstance(data, dict) and part in data:
            data = data[part]
        else:
            return default
    return data


def fmt(value, places: int = 3) -> str:
    return "--" if value is None else f"{value:.{places}f}"


def slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    return s


def box(s, x, y, w, h, fill=PANEL, line=LINE, radius=False):
    from pptx.enum.shapes import MSO_SHAPE

    shape = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, x, y, w, h
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    shape.text_frame.text = ""
    return shape


def text(s, x, y, w, h, runs, size=14, color=TEXT, bold=False, space=6, align=PP_ALIGN.LEFT):
    """runs: str, or list of str (paragraphs), or list of (text, level, bold, color, size)."""
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    items = runs if isinstance(runs, list) else [runs]
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            label, level = item[0], item[1]
            b = item[2] if len(item) > 2 else bold
            c = item[3] if len(item) > 3 else color
            sz = item[4] if len(item) > 4 else size
        else:
            label, level, b, c, sz = item, 0, bold, color, size
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = label
        p.level = level
        p.alignment = align
        p.space_after = Pt(space if level == 0 else max(2, space - 3))
        for r in p.runs:
            r.font.size = Pt(sz if level == 0 else sz - 1.5)
            r.font.bold = bool(b) and level == 0
            r.font.color.rgb = c
            r.font.name = "Segoe UI"
    return tb


def header(s, kicker: str, title: str, req: str):
    text(s, Inches(0.55), Inches(0.28), Inches(9.5), Inches(0.3), kicker.upper(),
         size=11, color=ACCENT, bold=True)
    text(s, Inches(0.55), Inches(0.58), Inches(10.3), Inches(0.75), title, size=27, color=WHITE, bold=True)
    bar = box(s, Inches(0.57), Inches(1.34), Inches(0.55), Inches(0.045), fill=ACCENT, line=None)
    bar.fill.gradient()
    text(s, Inches(10.6), Inches(0.5), Inches(2.25), Inches(0.6), req,
         size=9.5, color=MUTED, align=PP_ALIGN.RIGHT)


def table(s, x, y, w, rows, col_widths, font=10.5, header_row=True, row_h=Inches(0.3)):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = s.shapes.add_table(n_rows, n_cols, x, y, w, row_h * n_rows)
    tbl = shape.table
    total = sum(col_widths)
    for i, cw in enumerate(col_widths):
        tbl.columns[i].width = Emu(int(w * (cw / total)))
    for r, row in enumerate(rows):
        tbl.rows[r].height = row_h
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(val)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.025)
            cell.margin_bottom = Inches(0.025)
            cell.fill.solid()
            if header_row and r == 0:
                cell.fill.fore_color.rgb = PANEL2
            else:
                cell.fill.fore_color.rgb = PANEL if r % 2 else RGBColor(0x11, 0x1B, 0x2E)
            for p in cell.text_frame.paragraphs:
                p.space_after = Pt(0)
                for run in p.runs:
                    run.font.size = Pt(font if not (header_row and r == 0) else font - 0.5)
                    run.font.bold = header_row and r == 0
                    run.font.name = "Segoe UI"
                    run.font.color.rgb = CYAN if (header_row and r == 0) else TEXT
    return tbl


def notes(s, body: str):
    s.notes_slide.notes_text_frame.text = body


def footer(s, n: int):
    text(s, Inches(0.55), Inches(7.02), Inches(6.0), Inches(0.3),
         "HateXAI · Bennett University · Dr. Pradeepika Verma · 14 Oct 2026",
         size=9, color=MUTED)
    text(s, Inches(11.9), Inches(7.02), Inches(0.9), Inches(0.3), f"{n} / 9",
         size=9, color=MUTED, align=PP_ALIGN.RIGHT)


# ---------------------------------------------------------------- read artifacts
PROBE_ACC = number("outputs/eval/ablation.json", "rows.4.probe_overall_accuracy")
PROBE_UNAMB = number("outputs/eval/ablation.json", "rows.4.probe_unambiguous_accuracy")
CV = number("outputs/eval/ablation.json", "rows.4.cv_f1_macro_train")
BRIER = number("outputs/eval/ablation.json", "rows.4.test_brier")
LEGACY_PROBE = number("outputs/eval/ablation.json", "rows.0.probe_overall_accuracy")
LEGACY_BRIER = number("outputs/eval/ablation.json", "rows.0.test_brier")
PROBE_OP = number("outputs/eval/probes.json", "overall.accuracy")
PROBE_FPR = number("outputs/eval/probes.json", "overall.false_positive_rate")
PROBE_FNR = number("outputs/eval/probes.json", "overall.false_negative_rate")
THRESHOLD = number("outputs/eval/threshold.json", "threshold")
COMP = number("outputs/eval/faithfulness.json", "mean_comprehensiveness")
SUFF = number("outputs/eval/faithfulness.json", "mean_sufficiency")
ABSTAIN = number("outputs/eval/test_three_way.json", "abstained")

NOTES = {
    1: ("Open by naming the project and the promise. HateXAI is a hate-speech detector that must also justify itself and "
        "know when to abstain. Say plainly: the repository is an honest prototype — one system is trained and measured, "
        "three are implemented but untrained. Student name is not printed here because it is not recorded anywhere in the "
        "repository; add it before submitting."),
    2: ("Define the domain as identity-directed abuse, not profanity. Walk the two real probe examples: both contain the "
        "same high-risk word as genuine hate, both are gold non-hate, and the current model gets both wrong. That is the "
        "use-versus-mention problem, and it is the reason the app exposes needs_review."),
    3: ("Review existing systems in thirty seconds, one row per approach, and be explicit that every technique already "
        "exists and works for its own task. The gap is the combination and the audit: measured faithfulness, reward "
        "optimization applied to justifications rather than labels, hard-negative reporting, and a deferral outcome."),
    4: ("State the formal objective x = {T, V, C} and the binary research formulation, then separate it from the "
        "three-way application output. This distinction matters to faculty: metrics stay comparable with published work "
        "while the deployed system can route contested posts to a human."),
    5: ("Point at one lane at a time. The left lane is the pipeline that actually runs; the middle lane is the "
        "explainability and interface; the right lane is implemented but never trained and is explicitly not wired into "
        "serving. Close on the data reality box: 36 synthetic records and 36 author-built probes test the software, not "
        "the science."),
    6: ("For each technique, give the role, the reason it is the right choice, and the limitation. Two honesty statements "
        "to say out loud: TF-IDF does not understand context the way a transformer does, and highlighted tokens do not "
        "prove an author's intent. Then show the DPO objective and note the sequence log-probability defect that was "
        "found and fixed with a regression test."),
    7: ("Feasibility is the strongest part of the pitch: the pipeline runs, the API is tested, 40 tests pass on CPU in "
        "about 21 seconds, and every number regenerates from one command. Be equally direct about the blockers — no GPU, "
        "no licensed corpus, no image data, no LaTeX toolchain — and about the honest count of 24 training records."),
    8: ("Present it as a plan first, then disclose the preliminary diagnostics with their sample sizes in the same breath. "
        "Emphasise the two findings that matter: the false-negative-averse operating point buys recall with a 0.50 "
        "false-positive rate, and train-only CV and held-out probes disagree, which is expected with this little data."),
    9: ("Summarise the narrow, defensible contribution, restate the limitations without hedging, and close on the seven "
        "next steps. End with the two commands that reproduce everything, then invite questions about the transformer "
        "training plan and the cost model for the review band."),
}


def build() -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ---------------------------------------------------------- 1 title
    s = slide(prs)
    box(s, 0, 0, W, Inches(0.06), fill=ACCENT, line=None)
    text(s, Inches(0.7), Inches(0.75), Inches(9.0), Inches(0.4),
         "BENNETT UNIVERSITY  ·  MILESTONE 1 REVIEW", size=12, color=CYAN, bold=True)
    text(s, Inches(0.7), Inches(1.25), Inches(11.9), Inches(1.6),
         "Hate Speech Detection through Explainable AI and Reward-Optimized LLMs",
         size=38, color=WHITE, bold=True)
    text(s, Inches(0.7), Inches(2.95), Inches(11.4), Inches(1.1),
         "HateXAI — a multilingual, explainable detection prototype that classifies a post x = {T, V, C} as "
         "hate / non-hate, justifies each decision with measurable token evidence, and escalates genuinely "
         "ambiguous cases to human review instead of forcing a binary label.",
         size=15, color=TEXT)
    for i, (k, v) in enumerate([("Project", "HateXAI"),
                                ("Project instructor", "Dr. Pradeepika Verma · N2-WS-062"),
                                ("Presentation date", "14 October 2026"),
                                ("Format", "Milestone 1 · 9 slides")]):
        text(s, Inches(0.7 + i * 3.0), Inches(4.35), Inches(2.85), Inches(0.25), k, size=10, color=MUTED)
        text(s, Inches(0.7 + i * 3.0), Inches(4.6), Inches(2.85), Inches(0.4), v, size=13, color=WHITE, bold=True)
    box(s, Inches(0.7), Inches(5.5), Inches(11.9), Inches(0.02), fill=LINE, line=None)
    text(s, Inches(0.7), Inches(5.7), Inches(11.9), Inches(1.0),
         ["Requirement 1 — problem domain identification and analysis",
          "Requirement 2 — review of existing systems and project feasibility",
          "Requirement 3 — objectives and methodology of the proposed work",
          "Requirement 4 — relevance of the algorithms and techniques used"],
         size=11.5, color=MUTED, space=1)
    notes(s, NOTES[1])

    # ---------------------------------------------------------- 2 problem domain
    s = slide(prs)
    header(s, "Requirement 1 · Problem domain", "Why hate-speech detection is still an open NLP problem",
           "1 Problem domain identification & analysis")
    text(s, Inches(0.55), Inches(1.5), Inches(12.2), Inches(0.75),
         "Detection is identity-directed abuse, not profanity — and its errors are asymmetric: a false negative leaves a "
         "targeted group exposed, a false positive silences the dialect or reclaimed language of the very communities a "
         "moderator should protect [Sap 2019; Vidgen 2020].", size=12.5, color=TEXT)
    cols = [
        ("Language is context-dependent", ["Sarcasm, irony and dog-whistles invert surface meaning.",
                                           "Quoting abuse to condemn it looks lexical-identical to abuse.",
                                           "Reclaimed slurs inside a community differ from out-group use [ElSherief 2021]."]),
        ("Multilingual & code-mixed text", ["Hinglish and romanized Indic text break English word lexicons.",
                                            "Shared tasks show large drops when English models transfer [Mandl 2019].",
                                            "Script mixing also defeats naive tokenizers."]),
        ("Black-box decisions", ["Moderators cannot audit or appeal a score with no evidence.",
                                 "Attribution can look plausible yet be unfaithful [Jacovi 2020].",
                                 "Metrics alone hide systematic failure on hard cases."]),
    ]
    for i, (head, items) in enumerate(cols):
        x = Inches(0.55 + i * 4.12)
        box(s, x, Inches(2.35), Inches(3.9), Inches(2.15), radius=False)
        text(s, x + Inches(0.15), Inches(2.45), Inches(3.6), Inches(0.3), head, size=12.5, color=ACCENT, bold=True)
        text(s, x + Inches(0.15), Inches(2.8), Inches(3.6), Inches(1.6),
             [(f"• {t}", 0) for t in items], size=10.5, color=TEXT, space=4)
    box(s, Inches(0.55), Inches(4.7), Inches(7.9), Inches(2.05), PANEL, RGBColor(0xF8, 0x71, 0x71))
    text(s, Inches(0.72), Inches(4.8), Inches(7.6), Inches(0.3),
         "Two real examples from the project's probe set", size=12.5, color=WHITE, bold=True)
    text(s, Inches(0.72), Inches(5.12), Inches(7.6), Inches(1.5),
         ["gold: non-hate — “The poster said “immigrants are vermin” and the audience applauded. I am quoting it here "
          "because it should be condemned.” → measured P(hate) = 0.58 · wrong",
          "gold: non-hate — “Anyone claiming that community are vermin is repeating a lie. Please stop sharing that meme.” "
          "→ measured P(hate) = 0.64 · wrong"],
         size=10.5, color=TEXT, space=5)
    box(s, Inches(8.65), Inches(4.7), Inches(4.1), Inches(2.05), PANEL, LINE)
    text(s, Inches(8.82), Inches(4.8), Inches(3.8), Inches(1.8),
         [("What this shows", 0, True, ACCENT, 12.5),
          ("Both posts use the same high-risk word as genuine hate but are not hate. A bag-of-words model has no "
           "representation of who is speaking or with what intent, so the trigger fires either way.",
           0, False, TEXT, 10.5),
          ("This is the use-vs-mention distinction: systems that cannot tell them apart censor counterspeech "
           "[Gligorić 2024] — hence an explicit needs-review outcome.", 0, False, WARN, 10.5)],
         size=10.5, space=5)
    footer(s, 2)
    notes(s, NOTES[2])

    # ---------------------------------------------------------- 3 existing systems
    s = slide(prs)
    header(s, "Requirement 2 · Review of existing systems", "What already works, and where the gap actually is",
           "2 Review of existing systems & feasibility")
    rows = [["Approach", "Established strength", "Documented limitation for this project"],
            ["Lexicons & classical ML\nn-grams, TF-IDF + linear", "Fast, CPU-cheap, reproducible; strong in-domain English [Davidson 2017]",
             "Surface cues only — no context, sarcasm or code-mix; identity mentions inflate false positives"],
            ["Transformer classifiers\nHateBERT, XLM-R", "Contextual representations; strong cross-lingual transfer [Caselli 2021; Conneau 2020; Lees 2022]",
             "Needs licensed corpora, GPU time and per-language tuning; opaque to moderators"],
            ["Rationale-supervised data\nHateXplain", "Human token rationales enable explanation-aware training [Mathew 2021]",
             "English-centric; free-text rationales can be plausible without being faithful"],
            ["Explainable AI\nLIME, SHAP, attention", "Local, model-agnostic attribution moderators can read [Ribeiro 2016; Lundberg 2017]",
             "Often shown as illustrations; faithfulness is rarely measured [Jacovi 2020]"],
            ["Preference-optimized LLMs\nRLHF / DPO", "Aligns generation with preference without an explicit RL loop [Ouyang 2022; Rafailov 2023]",
             "Mostly applied to labels or dialogue, not justification quality; costly to train and evaluate"],
            ["Multimodal detection\nHateful Memes, MMHS150K", "Captures hate carried by the image, not the caption [Kiela 2020; Gomez 2020]",
             "Requires image-labelled corpora; none is present here, so nothing can be claimed yet"]]
    table(s, Inches(0.55), Inches(1.55), Inches(12.2), rows, [2.2, 3.6, 4.4], font=9.5, row_h=Inches(0.42))
    box(s, Inches(0.55), Inches(4.95), Inches(7.6), Inches(1.85), PANEL, ACCENT)
    text(s, Inches(0.72), Inches(5.03), Inches(7.3), Inches(1.7),
         [("The specific gap this project addresses", 0, True, WHITE, 12.5),
          ("• Explanations are asserted, not audited — we require a measured faithfulness number beside every attribution.", 0, False, TEXT, 10.5),
          ("• Reward optimization targets labels, rarely justifications — we treat (chosen, rejected) rationales as the preference signal.", 0, False, TEXT, 10.5),
          ("• Hard negatives are under-reported — quoted hate, counterspeech and neutral identity mentions are what we measure.", 0, False, TEXT, 10.5),
          ("• Binary outputs over-claim — an uncertain post should reach a human, not a coin flip.", 0, False, TEXT, 10.5)],
         size=10.5, space=4)
    box(s, Inches(8.35), Inches(4.95), Inches(4.4), Inches(1.85), PANEL, LINE)
    text(s, Inches(8.52), Inches(5.03), Inches(4.1), Inches(1.7),
         [("How the gap is framed honestly", 0, True, ACCENT2, 12.5),
          ("Every technique above already exists and performs its own task well. We do not claim prior systems cannot "
           "detect hate speech. The contribution is the combination and the audit: one shared, reproducible pipeline "
           "coupling calibrated probabilities, cost-weighted abstention, measured faithfulness and a preference-alignment "
           "experiment — reporting its failures as prominently as its successes.", 0, False, TEXT, 10.5)],
         size=10.5, space=4)
    footer(s, 3)
    notes(s, NOTES[3])

    # ---------------------------------------------------------- 4 problem statement
    s = slide(prs)
    header(s, "Requirement 3 · Objectives", "Problem statement and project objectives",
           "3 Objectives & methodology")
    box(s, Inches(0.55), Inches(1.5), Inches(6.0), Inches(2.35), PANEL, LINE)
    text(s, Inches(0.72), Inches(1.6), Inches(5.7), Inches(2.15),
         [("Research formulation", 0, True, WHITE, 12.5),
          ("x = { T , V , C }   →   fθ(x) = y ∈ {0, 1}", 0, True, RGBColor(0xFF, 0xE6, 0xA8), 13),
          ("• T — text, possibly multilingual or code-mixed (e.g. Hinglish)", 0, False, TEXT, 10.5),
          ("• V — optional image / meme content (not yet trained or evaluated)", 0, False, TEXT, 10.5),
          ("• C — context from available metadata or thread information only", 0, False, TEXT, 10.5),
          ("• y = 1 denotes identity-directed abuse or dehumanization, not mere profanity", 0, False, TEXT, 10.5)],
         size=10.5, space=4)
    box(s, Inches(6.75), Inches(1.5), Inches(6.0), Inches(2.35), PANEL, WARN)
    text(s, Inches(6.92), Inches(1.6), Inches(5.7), Inches(2.15),
         [("Two formulations, stated separately", 0, True, WARN, 12.5),
          ("Binary research formulation: y ∈ {0,1}, so metrics stay comparable with published work.", 0, False, TEXT, 10.5),
          ("Three-way application output: {hate, non-hate, needs_review} under an explicit cost model, with abstentions "
           "counted rather than hidden.", 0, False, TEXT, 10.5),
          ("Keeping both is deliberate: prior work reports two classes, while a real moderation queue needs a route to "
           "human review.", 0, False, MUTED, 10.5)],
         size=10.5, space=4)
    heads = ["Objectives — detection & reproducibility",
             "Objectives — language, explanation, alignment",
             "Objectives — evaluation & extension"]
    items = [["Build a reproducible hate-speech baseline that runs on CPU and regenerates every reported number from one command.",
              "Compare against a meaningful pre-improvement baseline, selected by train-only cross-validation."],
             ["Investigate multilingual and code-mixed classification (English, Hindi, Hinglish).",
              "Provide interpretable, evidence-based predictions with measured faithfulness.",
              "Investigate reward-guided rationale optimization via preference data."],
             ["Evaluate accuracy, error types, calibration and subgroup behaviour — not accuracy alone.",
              "Explore multimodal capability only where data and a trained model exist."]]
    for i, (head, its) in enumerate(zip(heads, items)):
        x = Inches(0.55 + i * 4.12)
        box(s, x, Inches(4.05), Inches(3.9), Inches(2.7), PANEL, LINE, radius=False)
        text(s, x + Inches(0.15), Inches(4.15), Inches(3.6), Inches(2.5),
             [(head, 0, True, ACCENT, 12)] + [(f"• {t}", 0) for t in its], size=10.5, color=TEXT, space=5)
    footer(s, 4)
    notes(s, NOTES[4])

    # ---------------------------------------------------------- 5 architecture
    s = slide(prs)
    header(s, "Requirement 3 · Methodology", "System architecture: what runs today, and what is a separate experiment",
           "3 Methodology & architecture")
    lanes = [
        ("SERVED PIPELINE — TRAINED, TESTED, REPRODUCIBLE", GOOD,
         "Input x = {T, V, C}  →  preprocess + canonical fusion (one shared string, fuse_text)  →  TF-IDF word + char "
         "n-grams → logistic regression (class_weight = balanced)  →  Platt calibration  →  cost-weighted threshold "
         f"fn:fp = 5:1  (θ = {fmt(THRESHOLD)}, band ±0.10, chosen on validation only)  →  hate | non-hate | needs_review"),
        ("EXPLAINABILITY & INTERFACE — WORKING", CYAN,
         "LIME / occlusion / attention attribution  →  measured faithfulness: comprehensiveness "
         f"{fmt(COMP)} · sufficiency {fmt(SUFF)} (n = 6)  →  FastAPI service + web dashboard (prediction, evidence, "
         "uncertainty, limitations, explicit “not run” states)  →  feedback stored as a review queue, never an automatic label"),
        ("SEPARATE EXPERIMENTAL PATHWAY — IMPLEMENTED, NOT TRAINED (training-time only, not wired into serving)", WARN,
         "XLM-R / DistilBERT + context (class-weighted loss, early stopping)  ·  + gated CLIP vision fusion (no image "
         "data present)  ·  Flan-T5 policy + DPO on chosen/rejected rationales (objective unit-tested, never run). "
         "Each is reported as available: false — never simulated."),
        ("DATA REALITY (VERIFIED)", MUTED,
         "36 synthetic records — 24 train / 6 val / 6 test;  36 author-built adversarial probes (11 categories, "
         "5 flagged ambiguous). Enough to test the pipeline, metrics and API — not enough to claim performance."),
    ]
    y = 1.55
    for head, color, body in lanes:
        box(s, Inches(0.55), Inches(y), Inches(12.2), Inches(1.06), PANEL, color)
        text(s, Inches(0.72), Inches(y + 0.06), Inches(11.9), Inches(0.28), head, size=11, color=color, bold=True)
        text(s, Inches(0.72), Inches(y + 0.35), Inches(11.85), Inches(0.65), body, size=10.5, color=TEXT)
        y += 1.2
    text(s, Inches(0.55), Inches(6.35), Inches(12.2), Inches(0.5),
         "Legend: trained & verified (green) · working (cyan) · implemented, not trained (amber) · data reality (grey). "
         "Reward/DPO is a training-time pathway, not a component of every prediction; multimodal processing is marked with "
         "its true evaluation status.", size=10, color=MUTED)
    footer(s, 5)
    notes(s, NOTES[5])

    # ---------------------------------------------------------- 6 algorithms
    s = slide(prs)
    header(s, "Requirement 4 · Algorithm relevance", "Why these techniques, and where each one breaks",
           "4 Relevance of algorithms & techniques")
    rows = [["Technique", "Role in the system", "Why it is the right choice here", "Limitation we accept"],
            ["Word + character n-gram TF-IDF with logistic regression",
             "The served classifier; the reproducible baseline every other system is compared against.",
             "Deterministic, CPU-only, seconds to train; character n-grams survive code-mixing and typos that English word lexicons miss.",
             "No word order and no notion of who is speaking: it cannot resolve irony, quotation or implicit hate."],
            ["Calibrated probabilities (Platt scaling) [Guo 2017]",
             "Turns scores into probabilities a threshold can be trusted against.",
             f"Leaves ranking (AUROC) unchanged but cuts Brier score from {fmt(LEGACY_BRIER)} to {fmt(BRIER)} — what makes an abstention band meaningful.",
             "Fitted on very few examples; ECE stays at 0.271, so probabilities are indicative, not numerically reliable."],
            ["Cost-weighted three-way decision with abstention [Madras 2018]",
             "Maps a probability to hate / non-hate / needs_review under an explicit error-cost model.",
             "Moderation errors are asymmetric: visible hate costs more than over-flagging. A deferral region is the honest output for contested posts.",
             f"The selected point buys recall with false positives: probe accuracy {fmt(PROBE_OP)} with FPR {fmt(PROBE_FPR)}. The cost model is a policy choice."],
            ["Transformer classification: XLM-R / DistilBERT-multilingual [Conneau 2020; Caselli 2021]",
             "The planned contextual multilingual upgrade over n-grams.",
             "Cross-lingual pretraining transfers to Hindi and romanized code-mixed text, where the baseline is weakest; class-weighted loss handles imbalance.",
             "Implemented but never trained here: no checkpoint, no GPU, no licensed corpus. Reported as available: false."],
            ["Explainability: LIME, occlusion, attention [Ribeiro 2016; Lundberg 2017]",
             "Produces per-token evidence for the moderate-confidence decisions a human will read.",
             "Model-agnostic, so one explainer serves the baseline and any future transformer; occlusion needs no gradients, so it cannot silently mis-attribute.",
             "Attribution is not evidence of intent, and plausibility is not faithfulness [Jacovi 2020]. Attention is diagnostic only."],
            ["Direct Preference Optimization on rationales [Rafailov 2023; Ouyang 2022]",
             "The reward-optimization experiment: align generated justifications with moderator-preferred explanations.",
             "DPO avoids an explicit reward model and RL loop — both need far more data and compute than this project has.",
             "Not trained. Weak negatives let the policy reward verbosity or fluent-but-unfaithful text; the shipped reward score was self-referential until audited and fixed."]]
    table(s, Inches(0.55), Inches(1.5), Inches(12.2), rows, [2.4, 3.0, 3.6, 3.2], font=8.5, row_h=Inches(0.5))
    box(s, Inches(0.55), Inches(5.15), Inches(7.0), Inches(1.45), PANEL, ACCENT2)
    text(s, Inches(0.72), Inches(5.25), Inches(6.7), Inches(1.25),
         "We do not claim TF-IDF understands context the way a transformer does, and we do not claim highlighted tokens "
         "prove an author's intent. Both statements would be false; the table marks the difference explicitly.",
         size=10.5, color=TEXT)
    box(s, Inches(7.75), Inches(5.15), Inches(5.0), Inches(1.45), PANEL, WARN)
    text(s, Inches(7.92), Inches(5.25), Inches(4.7), Inches(1.25),
         "DPO objective actually implemented (the encoder–decoder log-probability shift is unit-tested after a defect fix):  "
         "L = −log σ( β [ log πθ(e+|x)/πref(e+|x) − log πθ(e−|x)/πref(e−|x) ] )",
         size=10, color=RGBColor(0xFF, 0xE6, 0xA8))
    footer(s, 6)
    notes(s, NOTES[6])

    # ---------------------------------------------------------- 7 feasibility
    s = slide(prs)
    header(s, "Requirement 2 · Feasibility & evidence", "Feasibility, resources, and exactly what exists today",
           "2 Feasibility & current implementation")
    cards = [
        ("complete — working and verified", GOOD,
         ["Preprocessing plus calibrated baseline trained end to end.",
          "FastAPI service: /health, /api/model-info, /api/predict, /api/explain, /api/evaluation, /api/feedback, /api/vision-check — loaded once per worker.",
          "HTML/CSS/JS dashboard: six screens, verified in a browser.",
          "Test suite: 40 tests pass (exit 0, ~21 s, CPU).",
          "Experiment runner writes 8 machine-readable artifacts plus a run manifest (seed 42, cpu, Python 3.13.9)."]),
        ("implemented, not trained — no checkpoint", WARN,
         ["XLM-R / DistilBERT classifier with class-weighted loss and early stopping.",
          "Gated CLIP vision-fusion module (no image data present).",
          "Flan-T5 policy + DPO trainer (objective unit-tested; never run).",
          "Each is surfaced as available: false in the API and dashboard rather than filled with an estimate."]),
        ("blockers, stated plainly", RGBColor(0xF8, 0x71, 0x71),
         ["No GPU on the development machine.",
          "No licensed corpus (HateXplain / HASOC / MMHS150K) is present.",
          "No image data, so multimodal cannot be evaluated.",
          "No LaTeX toolchain (pdflatex), so the ACL draft is not compiled here."]),
    ]
    for i, (head, color, its) in enumerate(cards):
        x = Inches(0.55 + i * 4.12)
        box(s, x, Inches(1.5), Inches(3.9), Inches(2.6), PANEL, color, radius=False)
        text(s, x + Inches(0.15), Inches(1.58), Inches(3.6), Inches(0.3), head, size=11.5, color=color, bold=True)
        text(s, x + Inches(0.15), Inches(1.92), Inches(3.6), Inches(2.1),
             [(f"• {t}", 0) for t in its], size=10, color=TEXT, space=4)
    rows = [["Split", "Records", "Hate", "Non-hate", "Languages"],
            ["Train", "24", "12", "12", "17 en · 7 hi (4 code-mixed)"],
            ["Validation", "6", "3", "3", "5 en · 1 hi"],
            ["Test", "6", "3", "3", "5 en · 1 hi"],
            ["Total synthetic", "36", "18", "18", "illustrative wiring data"],
            ["Adversarial probes", "36", "12", "24", "11 categories, 5 flagged ambiguous"]]
    table(s, Inches(0.55), Inches(4.3), Inches(7.0), rows, [2.0, 1.0, 0.8, 1.1, 2.6], font=9, row_h=Inches(0.3))
    text(s, Inches(0.55), Inches(6.25), Inches(7.0), Inches(0.7),
         "Counted by reading the JSONL files directly. 36 hand-written records prove the pipeline works; they are not a "
         "demographic sample and cannot establish scientific performance.", size=9.5, color=MUTED)
    box(s, Inches(7.75), Inches(4.3), Inches(5.0), Inches(2.0), PANEL, LINE)
    text(s, Inches(7.92), Inches(4.38), Inches(4.7), Inches(1.85),
         [("Why the project is still feasible", 0, True, WHITE, 12),
          ("• The full pipeline already runs, tests and reproduces on CPU with no GPU, so engineering risk is retired.", 0, False, TEXT, 10),
          ("• Remaining work is data and training, not architecture: a conversion script maps licensed JSON/JSONL corpora into the project schema.", 0, False, TEXT, 10),
          ("• Compute is bounded: CPU config ≈ 8 GB RAM, no GPU, slow; full config targets a ≥12 GB GPU.", 0, False, TEXT, 10),
          ("• Every claim reproduces with the experiment runner and the test suite.", 0, False, TEXT, 10)],
         size=10, space=3)
    footer(s, 7)
    notes(s, NOTES[7])

    # ---------------------------------------------------------- 8 evaluation
    s = slide(prs)
    header(s, "Requirement 3 · Evaluation", "Evaluation plan — and the preliminary diagnostics we already have",
           "3 Methodology & evaluation")
    rows = [["Aspect", "Planned protocol (target state)"],
            ["Classification", "macro-F1, weighted-F1, precision, recall, per-class F1, confusion matrix, separate FP and FN rates."],
            ["Calibration & abstention", "Brier score, ECE, validation-selected operating point, abstention rate reported alongside decided-case metrics."],
            ["Robustness", "Quoted hate, counterspeech, reclaimed slurs, neutral identity mentions, non-hateful profanity, sarcasm, code-mixed; contested cases flagged."],
            ["Subgroups", "Per-language and per-code-mix reporting with a minimum-support rule (categories under 3 items are reported as insufficient, not scored) [Borkan 2019]."],
            ["Explanations", "Comprehensiveness and sufficiency under one shared masking mechanism; rationale text judged separately from model evidence."],
            ["Ablations", "Feature set (char / word / union) × calibration, selected by train-only CV; plus text-only vs +context vs +vision vs +DPO."],
            ["Statistics", "Three seeds with mean ± std; confidence intervals once the corpus supports them."]]
    table(s, Inches(0.55), Inches(1.5), Inches(7.3), rows, [1.9, 5.4], font=8.5, row_h=Inches(0.36))
    box(s, Inches(8.05), Inches(1.5), Inches(4.7), Inches(4.05), PANEL, WARN)
    text(s, Inches(8.22), Inches(1.58), Inches(4.4), Inches(3.9),
         [("preliminary — engineering diagnostic, not a result", 0, True, WARN, 11.5),
          ("char n-grams (legacy):  CV F1 0.706 · test acc 0.667 · Brier " + fmt(LEGACY_BRIER) + " · probe acc " + fmt(LEGACY_PROBE), 0, False, TEXT, 10),
          ("word + char + Platt (selected):  CV F1 " + fmt(CV) + " · test acc 1.000 · Brier " + fmt(BRIER) + " · probe acc " + fmt(PROBE_ACC), 0, False, TEXT, 10),
          ("Read this with its sample sizes: the test split has 6 items, so one flipped example moves accuracy by 16.7 points; "
           "CV covers 24 training records; the 36-item probe set is author-constructed, not a public benchmark. "
           "Probe accuracy 0.639 → 0.750 overall, and 0.710 → 0.806 on the unambiguous subset.", 0, False, MUTED, 10),
          ("At the deployed operating point (θ = " + fmt(THRESHOLD) + ", fn:fp = 5:1) probe accuracy is " + fmt(PROBE_OP) +
           " with FPR " + fmt(PROBE_FPR) + " / FNR " + fmt(PROBE_FNR) + ": recall is bought with false positives, and "
           + str(int(ABSTAIN)) + " of 6 test items falls into needs_review.", 0, False, MUTED, 10),
          ("Faithfulness measured so far: comprehensiveness " + fmt(COMP) + ", sufficiency " + fmt(SUFF) +
           " (n = 6) — a low comprehensiveness means the top tokens do not explain much.", 0, False, MUTED, 10)],
         size=10, space=5)
    box(s, Inches(0.55), Inches(5.75), Inches(12.2), Inches(0.95), PANEL, ACCENT2)
    text(s, Inches(0.72), Inches(5.85), Inches(11.9), Inches(0.8),
         "Expected outcome of Milestone 2: the same protocol run on a licensed corpus with the transformer and, if the "
         "compute allows, the DPO variant — so the comparison table shows measured numbers for all four systems or an "
         "explicit reason why a system is absent.", size=11, color=TEXT)
    footer(s, 8)
    notes(s, NOTES[8])

    # ---------------------------------------------------------- 9 conclusion
    s = slide(prs)
    header(s, "Closing · limitations & next steps", "Conclusion, limitations and what happens next",
           "All four requirements · 9 slides")
    box(s, Inches(0.55), Inches(1.5), Inches(6.0), Inches(2.3), PANEL, ACCENT)
    text(s, Inches(0.72), Inches(1.58), Inches(5.7), Inches(2.15),
         [("Contribution so far", 0, True, WHITE, 12.5),
          ("HateXAI is a working, tested, reproducible research prototype that returns an auditable decision. The measured "
           "contribution is deliberately narrow: a calibrated CPU baseline with a cost-weighted three-way decision, an "
           "eight-configuration ablation selected by train-only CV, faithfulness that is measured rather than illustrated, "
           "and an adversarial probe set exposing systematic failure on quoted hate and counterspeech. The transformer, "
           "vision and DPO components are reported as untrained — not presented as results.", 0, False, TEXT, 10.5),
          ("Intended outcome: an ACL-format study of whether reward-guided rationales plus human review produce trustworthy "
           "moderation, not just a higher accuracy number.", 0, False, ACCENT2, 10.5)],
         size=10.5, space=5)
    box(s, Inches(6.75), Inches(1.5), Inches(6.0), Inches(2.3), PANEL, RGBColor(0xF8, 0x71, 0x71))
    text(s, Inches(6.92), Inches(1.58), Inches(5.7), Inches(2.15),
         [("Limitations we will not paper over", 0, True, WHITE, 12.5),
          ("• Headline numbers come from 36 hand-written records and an author-constructed probe set: engineering evidence, not scientific evidence.", 0, False, TEXT, 10.5),
          ("• No transformer, vision or DPO system has been trained or evaluated, so no claim is made for them.", 0, False, TEXT, 10.5),
          ("• The false-negative-averse operating point produces a 0.50 false-positive rate on probes.", 0, False, TEXT, 10.5),
          ("• Calibration is fitted on too few examples to be numerically reliable (ECE 0.271).", 0, False, TEXT, 10.5),
          ("• Attribution is not evidence of intent; hate-speech labels are culturally contested.", 0, False, TEXT, 10.5)],
         size=10.5, space=4)
    box(s, Inches(0.55), Inches(4.0), Inches(12.2), Inches(2.3), PANEL, LINE)
    text(s, Inches(0.72), Inches(4.08), Inches(3.9), Inches(2.15),
         [("Next steps (1–3)", 0, True, WHITE, 12),
          ("1. Prepare a properly licensed research corpus (HateXplain, HASOC, MMHS150K) and document provenance, licensing and preprocessing.", 0, False, TEXT, 10),
          ("2. Establish a leakage-aware, stratified baseline protocol on that corpus and freeze it.", 0, False, TEXT, 10),
          ("3. Train and evaluate the transformer with class weighting and compare against the baseline.", 0, False, TEXT, 10)],
         size=10, space=4)
    text(s, Inches(4.82), Inches(4.08), Inches(3.9), Inches(2.15),
         [("Next steps (4–6)", 0, True, WHITE, 12),
          ("4. Train the DPO variant on real, reviewed preference pairs and audit the reward it actually optimizes.", 0, False, TEXT, 10),
          ("5. Validate explainability (faithfulness, stability) and fairness per language and code-mix subgroup.", 0, False, TEXT, 10),
          ("6. Evaluate multimodal capability only if image-labelled data and compute become available.", 0, False, TEXT, 10)],
         size=10, space=4)
    text(s, Inches(8.92), Inches(4.08), Inches(3.7), Inches(2.15),
         [("Next steps (7) — and reproducibility", 0, True, WHITE, 12),
          ("7. Complete the experimental analysis, error analysis and the ACL-format paper with logged versions and hyperparameters.", 0, False, TEXT, 10),
          ("python experiments/run_experiments.py", 0, False, RGBColor(0xFF, 0xE6, 0xA8), 10),
          ("python -m pytest tests -q   → 40 passed", 0, False, RGBColor(0xFF, 0xE6, 0xA8), 10)],
         size=10, space=4)
    box(s, Inches(8.05), Inches(6.3), Inches(4.7), Inches(0.58), PANEL2, ACCENT2)
    text(s, Inches(8.2), Inches(6.38), Inches(4.4), Inches(0.45), "Thank you — Questions & Discussion",
         size=13.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    footer(s, 9)
    notes(s, NOTES[9])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
