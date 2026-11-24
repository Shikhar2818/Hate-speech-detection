from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatexai.baselines.linear import load_text_split, train_linear_baseline
from hatexai.data.public_sources import file_digest, load_json
from hatexai.eval.metrics import compute_classification_metrics, group_fairness
from hatexai.eval.thresholds import evaluate_three_way, select_operating_point
from hatexai.utils import load_config, maybe_set_hf_offline, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hate speech detection pipeline")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--stage", choices=["baseline", "classifier", "dpo", "all"], default="baseline")
    return parser.parse_args()


def score_split(model, path: str):
    texts, labels = load_text_split(path)
    preds = model.predict(texts)
    probs = model.predict_proba(texts)[:, 1].tolist()
    return labels, preds, probs


def eval_split(path: str, scores, *, with_language=False) -> dict:
    labels, preds, probs = scores
    metrics = compute_classification_metrics(labels, preds, probs)
    if with_language:
        languages = [row["language"] for row in read_jsonl(path)]
        metrics["by_language"] = group_fairness(labels, preds, languages)
        for language, entry in metrics["by_language"].items():
            if not entry["insufficient_support"]:
                indices = [i for i, value in enumerate(languages) if value == language]
                entry.update(compute_classification_metrics(
                    [labels[i] for i in indices], [preds[i] for i in indices],
                    [probs[i] for i in indices],
                ))
    return metrics


def verify_benchmark_split(config, manifest, split):
    path = Path(config["data"][f"{split}_path"])
    expected = manifest["splits"][split]
    if expected["file"] != path.name or "unlabeled" in expected["labels"]:
        raise ValueError(f"{split} does not name a labeled benchmark split")
    if file_digest(path) != expected["sha256"]:
        raise ValueError(f"{split} does not match the benchmark manifest checksum")


def run_public_baseline(config):
    from sklearn import __version__ as sklearn_version

    manifest_path = Path(config["data"]["manifest_path"])
    manifest = load_json(manifest_path)
    for split in ("train", "val"):
        verify_benchmark_split(config, manifest, split)
    started = perf_counter()
    model = train_linear_baseline(config)
    duration = perf_counter() - started
    val_path = config["data"]["val_path"]
    val_scores = score_split(model, val_path)
    policy = config.get("eval", {})
    operating = select_operating_point(
        val_scores[0], val_scores[2], fn_cost=policy.get("fn_cost", 5.0),
        fp_cost=policy.get("fp_cost", 1.0), review_band=policy.get("review_band", 0.10),
    )
    # Open test only after the final checkpoint and operating point are fixed.
    verify_benchmark_split(config, manifest, "test")
    test_path = config["data"]["test_path"]
    test_scores = score_split(model, test_path)
    results = {
        "baseline_val": eval_split(val_path, val_scores, with_language=True),
        "baseline_test": eval_split(test_path, test_scores, with_language=True),
        "threshold": operating,
        "test_three_way": evaluate_three_way(
            test_scores[0], test_scores[2], operating["low"], operating["high"],
        ),
    }
    checkpoint = Path(config["output_dir"]) / "checkpoints" / "linear_baseline.joblib"
    eval_dir = Path(config["output_dir"]) / "eval"
    for name, result in results.items():
        write_json(eval_dir / f"{name}.json", result)
    write_json(eval_dir / "baseline_training.json", {
        "config": config,
        "dataset_manifest_sha256": file_digest(manifest_path),
        "dataset_splits": manifest["splits"],
        "python": platform.python_version(),
        "scikit_learn": sklearn_version,
        "fit_and_validation_seconds": duration,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": file_digest(checkpoint),
        "threshold_sha256": file_digest(eval_dir / "threshold.json"),
    })
    return results


def main() -> None:
    maybe_set_hf_offline()
    args = parse_args()
    config = load_config(args.config)
    results = {}
    if args.stage in {"baseline", "all"}:
        if config["data"].get("manifest_path"):
            results.update(run_public_baseline(config))
        else:
            linear = train_linear_baseline(config)
            path = config["data"]["test_path"]
            results["baseline_test"] = eval_split(path, score_split(linear, path))
        print("baseline test:", results["baseline_test"])
    if args.stage in {"classifier", "all"}:
        from hatexai.train.train_classifier import train_classifier

        train_classifier(config)
    if args.stage in {"dpo", "all"}:
        from hatexai.train.train_dpo import train_dpo

        train_dpo(config)
    out = Path(config["output_dir"]) / "eval" / "pipeline_summary.json"
    write_json(out, results)


if __name__ == "__main__":
    main()
