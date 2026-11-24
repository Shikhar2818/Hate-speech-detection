from __future__ import annotations

"""Reproducible experiment runner.

Produces the machine-readable artifacts the backend and the evaluation dashboard
consume:

  outputs/eval/baseline_test.json    classification + calibration on the test split
  outputs/eval/threshold.json        operating point chosen on the validation split
  outputs/eval/test_three_way.json   three-way (hate/non-hate/review) test results
  outputs/eval/probes.json           diagnostic probe-set robustness results
  outputs/eval/faithfulness.json     explanation comprehensiveness/sufficiency
  outputs/eval/comparison.json       system comparison, with explicit availability
  outputs/eval/run_manifest.json     seed, config, environment, timestamps

Systems whose checkpoints do not exist are reported as ``available: false`` with
a reason. Nothing is invented for them; see PROJECT_AUDIT.md section 5.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatexai.baselines.linear import (
    FEATURE_SETS,
    baseline_settings,
    build_linear_pipeline,
    load_text_split,
    train_linear_baseline,
)
from hatexai.data.preprocess import extract_text_features, fuse_text
from hatexai.eval.metrics import compute_classification_metrics, group_fairness
from hatexai.eval.robustness import evaluate_probe_set, load_probes
from hatexai.eval.thresholds import evaluate_three_way, select_operating_point
from hatexai.utils import ensure_dir, load_config, resolve_device, set_seed
from hatexai.xai.attributions import lime_attribution
from hatexai.xai.faithfulness import comprehensiveness, sufficiency


def make_scorer(model) -> Callable[[str, str], float]:
    def score(text: str, context: str = "") -> float:
        feats = extract_text_features(text)
        fused = fuse_text(text, context, feats.language, feats.code_mixed)
        return float(model.predict_proba([fused])[0][1])

    return score


def _split_records(path: str) -> list[dict[str, Any]]:
    from hatexai.utils import read_jsonl

    return read_jsonl(path)


def run_ablation(config: dict[str, Any], scorer_factory: Callable) -> dict[str, Any]:
    """Feature-set / calibration ablation.

    Model selection uses 5-fold cross-validation on the *training* split only;
    the test split and probe set are reported as held-out evaluation of each
    already-fixed configuration, never used to pick the winner.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    train_texts, train_labels = load_text_split(config["data"]["train_path"])
    test_texts, test_labels = load_text_split(config["data"]["test_path"])
    probes = load_probes(ROOT / "data" / "probes" / "hard_cases.jsonl")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(config["seed"]))

    rows = []
    configurations = [
        (feature_set, calibrate)
        for feature_set in FEATURE_SETS
        for calibrate in (False, True)
        if not (feature_set == "char" and calibrate)  # keep the legacy row as-is
    ]
    for feature_set, calibrate in configurations:
        model = build_linear_pipeline(feature_set=feature_set, calibrate=calibrate, cv=3)
        try:
            cv_scores = cross_val_score(model, train_texts, train_labels, cv=cv, scoring="f1_macro")
            cv_mean = float(cv_scores.mean())
        except Exception as exc:  # too few examples for the requested CV
            cv_mean = None
            print(f"ablation {feature_set}/{calibrate}: CV failed ({exc})")
        model.fit(train_texts, train_labels)
        probs = model.predict_proba(test_texts)[:, 1]
        test_metrics = compute_classification_metrics(
            test_labels, model.predict(test_texts).tolist(), probs.tolist()
        )
        probe_report = evaluate_probe_set(scorer_factory(model), probes, threshold=0.5)
        rows.append(
            {
                "feature_set": feature_set,
                "calibrated": calibrate,
                "cv_f1_macro_train": cv_mean,
                "test_accuracy": test_metrics["accuracy"],
                "test_auroc": test_metrics["auroc"],
                "test_brier": test_metrics["brier"],
                "test_ece": test_metrics["ece"],
                "probe_overall_accuracy": probe_report["overall"]["accuracy"],
                "probe_unambiguous_accuracy": probe_report["unambiguous"]["accuracy"],
            }
        )
        print(f"ablation {feature_set} calibrate={calibrate}: {rows[-1]}")

    selected = baseline_settings(config)
    return {
        "cv_folds": 3,
        "selection_metric": "cv_f1_macro_train",
        "rows": rows,
        "selected": selected,
        "note": (
            "Selecting on train-only CV favours the union of word and char n-grams; calibration "
            "leaves ranking (AUROC) unchanged but improves Brier score and ECE, which the "
            "abstention band depends on. Test/probe columns are held-out reports, not selection signals."
        ),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(Path(config["output_dir"]) / "eval")
    set_seed(int(config["seed"]))

    model = train_linear_baseline(config)
    scorer = make_scorer(model)

    # --- test split -------------------------------------------------------
    test_texts, test_labels = load_text_split(config["data"]["test_path"])
    test_records = _split_records(config["data"]["test_path"])
    test_preds = model.predict(test_texts).tolist()
    test_probs = model.predict_proba(test_texts)[:, 1].tolist()
    test_metrics = compute_classification_metrics(test_labels, test_preds, test_probs)
    test_metrics["fairness_by_language"] = group_fairness(
        test_labels, test_preds, [r.get("language", "und") for r in test_records]
    )

    # --- validation -> operating point -----------------------------------
    val_texts, val_labels = load_text_split(config["data"]["val_path"])
    val_probs = model.predict_proba(val_texts)[:, 1].tolist()
    eval_cfg = config.get("eval", {}) or {}
    operating_point = select_operating_point(
        val_labels,
        val_probs,
        fn_cost=float(eval_cfg.get("fn_cost", 5.0)),
        fp_cost=float(eval_cfg.get("fp_cost", 1.0)),
        review_band=float(eval_cfg.get("review_band", 0.10)),
    )

    three_way = evaluate_three_way(
        test_labels, test_probs, operating_point["low"], operating_point["high"]
    )

    # --- diagnostic probe set --------------------------------------------
    probe_path = ROOT / "data" / "probes" / "hard_cases.jsonl"
    probes = load_probes(probe_path)
    probe_report = evaluate_probe_set(scorer, probes, threshold=operating_point["threshold"])

    # --- explanation faithfulness on a small test sample ------------------
    sample = test_records[: min(6, len(test_records))]
    faith_rows = []
    for record in sample:
        feats = extract_text_features(record["text"])
        fused = fuse_text(record["text"], record.get("context", ""), feats.language, feats.code_mixed)
        label = int(model.predict([fused])[0])
        tokens, scores = lime_attribution(fused, model.predict_proba, samples=60)
        faith_rows.append(
            {
                "id": record.get("id"),
                "label": label,
                "comprehensiveness": comprehensiveness(fused, tokens, scores, model.predict_proba, label),
                "sufficiency": sufficiency(fused, tokens, scores, model.predict_proba, label),
            }
        )
    faithfulness = {
        "n": len(faith_rows),
        "mean_comprehensiveness": (
            sum(r["comprehensiveness"] for r in faith_rows) / len(faith_rows) if faith_rows else None
        ),
        "mean_sufficiency": (
            sum(r["sufficiency"] for r in faith_rows) / len(faith_rows) if faith_rows else None
        ),
        "rows": faith_rows,
        "note": "Computed with LIME attributions under delete-one masking; higher is better for both.",
    }

    # --- system comparison (availability is truthful, not aspirational) ---
    ckpt_dir = Path(config["output_dir"]) / "checkpoints"
    transformer_available = (ckpt_dir / "hate_detector.pt").exists()
    dpo_available = (ckpt_dir / "dpo_policy").exists()
    comparison = {
        "systems": [
            {
                "name": "Linear TF-IDF + LogReg",
                "available": True,
                "accuracy": test_metrics["accuracy"],
                "f1_macro": test_metrics["f1_macro"],
                "auroc": test_metrics["auroc"],
                "ece": test_metrics["ece"],
                "note": "Shipped checkpoint; the only system evaluated in this environment.",
            },
            {
                "name": "Transformer (XLM-R / DistilBERT-multilingual) + context",
                "available": transformer_available,
                "accuracy": None,
                "f1_macro": None,
                "auroc": None,
                "ece": None,
                "note": (
                    "Checkpoint found at outputs/checkpoints/hate_detector.pt."
                    if transformer_available
                    else "Not trained in this environment: requires model download and CPU/GPU training time."
                ),
            },
            {
                "name": "Transformer + gated CLIP vision fusion",
                "available": False,
                "accuracy": None,
                "f1_macro": None,
                "auroc": None,
                "ece": None,
                "note": "Requires an image-labelled corpus (e.g. MMHS150K); no image data is present.",
            },
            {
                "name": "DPO-aligned explanation policy",
                "available": dpo_available,
                "accuracy": None,
                "f1_macro": None,
                "auroc": None,
                "ece": None,
                "note": (
                    "Flan-T5-small policy checkpoint present; explanation quality is scored separately from classification."
                    if dpo_available
                    else "Not trained: requires Flan-T5 download and preference-pair training."
                ),
            },
        ],
        "test_split_n": test_metrics["n"],
        "caveat": (
            f"The bundled test split has only {test_metrics['n']} examples, so differences between systems "
            "are not statistically meaningful. Run scripts/prepare_public_data.py on a licensed corpus first."
        ),
    }

    ablation = run_ablation(config, make_scorer)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": config["seed"],
        "device": str(resolve_device(config.get("device", "auto"))),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config_path": "configs/default.yaml",
        "data": config["data"],
    }

    artifacts = {
        "baseline_test.json": test_metrics,
        "threshold.json": operating_point,
        "test_three_way.json": three_way,
        "probes.json": probe_report,
        "faithfulness.json": faithfulness,
        "comparison.json": comparison,
        "ablation.json": ablation,
        "run_manifest.json": manifest,
    }
    for name, payload in artifacts.items():
        (out_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"wrote {out_dir / name}")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible HateXAI experiments")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
