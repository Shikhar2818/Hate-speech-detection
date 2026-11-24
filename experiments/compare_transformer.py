"""Compare saved classifiers on identical, checksummed labeled splits."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from hatexai.baselines.linear import load_text_split
from hatexai.data.public_sources import file_digest, load_json
from hatexai.data.preprocess import fuse_text
from hatexai.data.public_splits import canonical_text, text_matches
from hatexai.eval.metrics import compute_classification_metrics
from hatexai.eval.thresholds import evaluate_three_way, select_operating_point
from hatexai.inference import build_service_from_config
from hatexai.models.xlm_roberta import XLMClassifier
from hatexai.utils import load_config, read_jsonl, resolve_device, write_json
from scripts.run_pipeline import eval_split, verify_benchmark_split


def compare(config):
    data = config["data"]
    manifest_path = Path(data["manifest_path"])
    manifest = load_json(manifest_path)
    settings = config["transformer"]
    training = Path(settings["checkpoint"]) / "training.json"
    baseline = load_json(Path(config["output_dir"]) / "eval" / "baseline_training.json")
    digest = file_digest(manifest_path)
    if baseline["dataset_manifest_sha256"] != digest or load_json(training)["manifest_sha256"] != digest:
        raise ValueError("Both classifiers must have been trained with the same published split manifest")
    linear = build_service_from_config(config)
    transformer = XLMClassifier(settings["checkpoint"], resolve_device(config["device"]),
                                settings["max_length"], settings["batch_size"])
    models = {"TF-IDF + logistic regression": linear.predict_proba_texts,
              "XLM-RoBERTa-base": transformer.predict_proba}
    report = {"dataset_manifest_sha256": digest,
              "classifier_training_sha256": file_digest(training),
              "test_sha256": manifest["splits"]["test"]["sha256"],
              "systems": []}
    policy = config.get("eval", {})
    verify_benchmark_split(config, manifest, "val")
    val, y_val = load_text_split(data["val_path"])
    thresholds = {}
    for name, predict in models.items():
        thresholds[name] = select_operating_point(
            y_val, predict(val)[:, 1], fn_cost=policy.get("fn_cost", 5),
            fp_cost=policy.get("fp_cost", 1), review_band=policy.get("review_band", 0.10),
        )
    verify_benchmark_split(config, manifest, "test")
    texts, labels = load_text_split(data["test_path"])
    for name, predict in models.items():
        scores = np.asarray(predict(texts))
        threshold = thresholds[name]
        metrics = eval_split(data["test_path"], (labels, scores.argmax(1), scores[:, 1]),
                             with_language=True)
        report["systems"].append({
            "model": name, "metrics": metrics, "threshold": threshold,
            "moderation": evaluate_three_way(labels, scores[:, 1], threshold["low"], threshold["high"]),
        })
    report["caveat"] = config.get("comparison_scope", "Held-out HateXplain and HiFoX posts. Cohorts are corpus-specific.")
    out = Path(settings["evaluation_dir"])
    write_json(out / "transformer_comparison.json", report)
    table = ["| Model | Accuracy | Macro F1 | English F1 | Hinglish F1 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for model in report["systems"]:
        m = model["metrics"]
        en = m["by_language"].get("en", {}).get("f1_macro")
        hi = m["by_language"].get("hi-en", {}).get("f1_macro")
        values = [m["accuracy"], m["f1_macro"], en, hi]
        row = " | ".join(f"{v:.3f}" if v is not None else "—" for v in values)
        table.append(f"| {model['model']} | {row} |")
    (out / "transformer_comparison.md").write_text("\n".join([*table, "", report["caveat"], ""]),
                                                    encoding="utf-8")
    print("\n".join(table))
    return report


def compare_context(config, manifest_path, limit):
    """Evaluate both classifiers on the same independent context-bearing comments."""
    from backend.registry import load_transformer

    train_manifest = load_json(Path(config["data"]["manifest_path"]))
    verify_benchmark_split(config, train_manifest, "train")
    manifest = load_json(manifest_path)
    path = manifest_path.parent / manifest["splits"]["test"]["file"]
    if file_digest(path) != manifest["splits"]["test"]["sha256"]:
        raise ValueError("Context test data does not match its manifest")
    train = {canonical_text(row["text"]) for row in read_jsonl(config["data"]["train_path"])}
    candidates = [row for row in read_jsonl(path) if row["context"] and canonical_text(row["text"]) not in train]
    rng = np.random.default_rng(config["seed"])
    selected = [candidates[i] for i in rng.choice(len(candidates), min(limit, len(candidates)), replace=False)]
    cutoff = len(train)
    indexed = dict(enumerate([*sorted(train), *(canonical_text(row["text"]) for row in selected)]))
    overlap = {max(i, j) - cutoff for i, j in text_matches(indexed) if (i < cutoff) != (j < cutoff)}
    rows = [row for i, row in enumerate(selected) if i not in overlap]
    if not rows:
        raise ValueError("No eligible context-bearing evaluation records")
    labels = [row["label"] for row in rows]
    systems = {"TF-IDF + logistic regression": build_service_from_config(config),
               "XLM-RoBERTa-base": load_transformer(config, config["transformer"])}
    report = {"test_sha256": file_digest(path), "n": len(rows), "systems": {},
              "scope": "CADD Reddit comments with parent context; checked for training-post text overlap."}
    for name, model in systems.items():
        metrics = {}
        for include in (False, True):
            texts = [fuse_text(row["text"], row["context"] if include else "") for row in rows]
            probs = model.predict_proba_texts(texts)
            metrics["with_context" if include else "text_only"] = compute_classification_metrics(
                labels, probs.argmax(axis=1), probs[:, 1],
            )
        report["systems"][name] = metrics
    write_json(Path(config["transformer"]["evaluation_dir"]) / "context_comparison.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--context-manifest", type=Path)
    parser.add_argument("--context-limit", type=int, default=256)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.context_manifest:
        if args.context_limit < 1:
            parser.error("Context limit must be positive")
        report = compare_context(config, args.context_manifest, args.context_limit)
        print(f"Compared both classifiers on {report['n']} context-bearing comments")
    else:
        compare(config)
