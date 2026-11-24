"""Measure probability drops and attribution stability on held-out text."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from backend.registry import load_transformer, transformer_report
from hatexai.baselines.linear import load_text_split
from hatexai.data.public_sources import load_json
from hatexai.utils import load_config, read_jsonl, write_json
from hatexai.xai.transformer import explain_tokens
from scripts.run_pipeline import verify_benchmark_split


def evaluate(config, per_language):
    comparison = transformer_report(config)
    service = load_transformer(config, config["transformer"])
    data = config["data"]
    manifest = load_json(Path(data["manifest_path"]))
    verify_benchmark_split(config, manifest, "test")
    texts, labels = load_text_split(data["test_path"])
    groups = defaultdict(list)
    for i, row in enumerate(read_jsonl(data["test_path"])):
        groups[row["language"]].append(i)
    rng = np.random.default_rng(config["seed"])
    samples = {group: rng.choice(indices, min(per_language, len(indices)), replace=False).tolist()
               for group, indices in groups.items()}
    report = {key: comparison[key] for key in ("test_sha256", "classifier_training_sha256")}
    report["methods"] = {}
    for method in ("integrated_gradients", "shap"):
        by_language = {}
        for language, indices in samples.items():
            rows = []
            for i in indices:
                label = int(service.predict_proba_texts([texts[i]])[0].argmax())
                settings = config.get("xai", {})
                tokens, scores, faith = explain_tokens(service.classifier, texts[i], label, method, settings)
                if not tokens:
                    continue
                repeat = {**settings, "ig_steps": 2 * settings.get("ig_steps", 16), "shap_seed": 83}
                _, other, _ = explain_tokens(service.classifier, texts[i], label, method, repeat)
                k = min(5, len(tokens))
                a, b = (set(np.argsort(-np.abs(s), kind="stable")[:k]) for s in (scores, other))
                rows.append({**faith, "stability_jaccard": len(a & b) / len(a | b),
                             "correct": label == labels[i]})
            keys = ("comprehensiveness", "sufficiency", "random_mask_drop", "stability_jaccard")
            by_language[language] = {"n": len(rows), **{k: mean(r[k] for r in rows) for k in keys}} if rows else {"n": 0}
        report["methods"][method] = by_language
    report["notes"] = ("Small held-out sample. Stability is top-five overlap across IG integration resolutions "
                       "or SHAP permutation seeds. Masking measures sensitivity to the reference token.")
    write_json(Path(config["transformer"]["evaluation_dir"]) / "transformer_xai.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--per-language", type=int, default=4)
    args = parser.parse_args()
    if args.per_language < 1:
        parser.error("Choose at least one sample per language")
    print(evaluate(load_config(args.config), args.per_language))
