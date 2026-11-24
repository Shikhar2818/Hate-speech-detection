"""Evaluate a fixed public baseline with native-label errors and token masking."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hatexai.baselines.linear import FEATURE_SETS, baseline_settings, build_linear_pipeline, load_text_split
from hatexai.data.preprocess import fuse_text
from hatexai.data.public_sources import file_digest, load_json
from hatexai.eval.metrics import compute_classification_metrics, group_fairness
from hatexai.inference import build_service_from_config
from hatexai.utils import load_config, read_jsonl, write_json
from hatexai.xai.attributions import occlusion_attribution
from hatexai.xai.faithfulness import comprehensiveness, sufficiency
from scripts.run_pipeline import verify_benchmark_split


def analyze_errors(records, predictions):
    buckets = defaultdict(Counter)
    groups = []
    for row, prediction in zip(records, predictions, strict=True):
        provenance = row["provenance"]
        source_label = (provenance["dataset"], provenance["source_label"], row["label"])
        buckets[source_label]["support"] += 1
        buckets[source_label]["errors"] += int(prediction != row["label"])
        groups.append("mixed_script" if row["code_mixed"] else "no_mixed_script_detected")
    mix_report = group_fairness([row["label"] for row in records], predictions, groups)
    mix_report.setdefault("mixed_script", {"support": 0, "insufficient_support": True})
    return {
        "n": len(records),
        "native_labels": [
            {"dataset": dataset, "source_label": source_label, "label": label, **counts,
             "error_rate": counts["errors"] / counts["support"]}
            for (dataset, source_label, label), counts in sorted(buckets.items())
        ],
        "by_script_mix": mix_report,
        "note": "Errors use the fixed binary classifier. Script mixing is a heuristic.",
    }


def measure_faithfulness(texts, predictions, predict_proba, *, seed, top_k=5):
    """Compare evidence with an equal-size random mask on the same examples."""
    rng = random.Random(seed)
    rows = []
    for text, target in zip(texts, predictions, strict=True):
        if len(text.split()) <= top_k:
            continue
        tokens, scores = occlusion_attribution(text, predict_proba, target_class=target)
        random_positions = set(rng.sample(range(len(tokens)), top_k))
        random_scores = [float(i in random_positions) for i in range(len(tokens))]
        rows.append({
            "comprehensiveness": comprehensiveness(text, tokens, scores, predict_proba, target, top_k),
            "sufficiency": sufficiency(text, tokens, scores, predict_proba, target, top_k),
            "random_comprehensiveness": comprehensiveness(
                text, tokens, random_scores, predict_proba, target, top_k,
            ),
            "random_sufficiency": sufficiency(text, tokens, random_scores, predict_proba, target, top_k),
        })
    return {
        "n": len(rows),
        "excluded_at_most_top_k_tokens": len(texts) - len(rows),
        "top_k": top_k,
        "seed": seed,
        "method": "occlusion",
        **{
            f"mean_{key}": fmean(row[key] for row in rows) if rows else None
            for key in ("comprehensiveness", "sufficiency", "random_comprehensiveness", "random_sufficiency")
        },
        "note": (
            f"Held-out posts with more than {top_k} tokens. Comprehensiveness measures the probability "
            "drop after deleting evidence. Sufficiency is the probability retained using only evidence. "
            "Both use the model's binary predicted class. Random masks use the same token counts."
        ),
    }


def fit_feature_ablations(train_path):
    texts, labels = load_text_split(train_path)
    models = []
    for feature_set in FEATURE_SETS:
        model = build_linear_pipeline(feature_set=feature_set, calibrate=False)
        model.fit(texts, labels)
        models.append((feature_set, False, model))
    return models


def evaluate_ablations(models, test_path):
    texts, labels = load_text_split(test_path)
    rows = []
    for feature_set, calibrated, model in models:
        probabilities = model.predict_proba(texts)
        metrics = compute_classification_metrics(labels, probabilities.argmax(axis=1), probabilities[:, 1])
        rows.append({
            "feature_set": feature_set,
            "calibrated": calibrated,
            "cv_f1_macro_train": None,
            "test_f1_macro": metrics["f1_macro"],
            "test_accuracy": metrics["accuracy"],
            "test_brier": metrics["brier"],
            "test_ece": metrics["ece"],
            "probe_overall_accuracy": None,
            "probe_unambiguous_accuracy": None,
        })
    return {
        "n": len(labels),
        "rows": rows,
        "note": (
            "Fixed word, character, and combined variants fitted on train. "
            "The final row is the saved reference. All configurations were fixed before test scoring."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis", choices=("errors-faithfulness", "ablation"),
                        default="errors-faithfulness")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = config["data"]["manifest_path"]
    manifest = load_json(Path(manifest_path))
    eval_dir = Path(config["output_dir"]) / "eval"
    training = load_json(eval_dir / "baseline_training.json")
    if training["dataset_manifest_sha256"] != file_digest(Path(manifest_path)):
        raise ValueError("Benchmark manifest does not match the fixed checkpoint's training run")
    service = build_service_from_config(config)
    provenance = {
        "test_sha256": manifest["splits"]["test"]["sha256"],
        "checkpoint_sha256": training["checkpoint_sha256"],
    }
    if args.analysis == "ablation":
        verify_benchmark_split(config, manifest, "train")
        models = fit_feature_ablations(config["data"]["train_path"])
        settings = baseline_settings(training["config"])
        models.append((settings["feature_set"], settings["calibrate"], service.model))
        verify_benchmark_split(config, manifest, "test")
        report = evaluate_ablations(models, config["data"]["test_path"])
        write_json(eval_dir / "ablation.json", {**report, **provenance})
        print(f"Evaluated {len(models)} fixed TF-IDF variants on {report['n']} test records")
        return
    verify_benchmark_split(config, manifest, "test")
    records = read_jsonl(config["data"]["test_path"])
    texts = [fuse_text(row["text"], row.get("context", "")) for row in records]
    predict_proba = service.predict_proba_texts
    decisions = predict_proba(texts).argmax(axis=1).tolist()
    errors = analyze_errors(records, decisions)
    write_json(eval_dir / "error_analysis.json", {**errors, **provenance})
    faithfulness = measure_faithfulness(texts, decisions, predict_proba, seed=config["seed"])
    write_json(eval_dir / "faithfulness.json", {**faithfulness, **provenance})
    print(f"Reported errors on {errors['n']} test records and faithfulness on {faithfulness['n']}")


if __name__ == "__main__":
    main()
