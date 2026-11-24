"""Measure a validation-selected binary threshold on the English and Hinglish cohorts."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from hatexai.baselines.linear import load_text_split
from hatexai.data.public_sources import file_digest, load_json
from hatexai.eval.metrics import error_rates
from hatexai.inference import build_service_from_config
from hatexai.utils import load_config, read_jsonl, write_json
from scripts.run_pipeline import verify_benchmark_split


def rates_at(labels, scores, languages, threshold):
    groups = {
        str(language): error_rates(labels[languages == language], scores[languages == language] >= threshold)
        for language in sorted(set(languages))
    }
    fprs = [group["false_positive_rate"] for group in groups.values()]
    return {"by_cohort": groups, "fpr_gap": max(fprs) - min(fprs)}


def choose_threshold(labels, scores, languages):
    reference = rates_at(labels, scores, languages, 0.5)
    results = [(0.5, reference)]
    for threshold in np.linspace(0.4, 0.65, 51):
        current = rates_at(labels, scores, languages, threshold)
        if all(
            rates["false_negative_rate"] <= reference["by_cohort"][name]["false_negative_rate"] + 0.03
            for name, rates in current["by_cohort"].items()
        ):
            results.append((float(threshold), current))
    return min(results, key=lambda candidate: (candidate[1]["fpr_gap"], abs(candidate[0] - 0.5)))[0]


if __name__ == "__main__":
    config = load_config("configs/english_hinglish.yaml")
    service = build_service_from_config(config)
    manifest_path = Path(config["data"]["manifest_path"])
    manifest = load_json(manifest_path)
    report = {"manifest_sha256": file_digest(manifest_path), "reference_threshold": 0.5}
    for split in ("val", "test"):
        verify_benchmark_split(config, manifest, split)
        path = config["data"][f"{split}_path"]
        texts, labels = load_text_split(path)
        labels = np.array(labels)
        scores = service.predict_proba_texts(texts)[:, 1]
        languages = np.array([row["language"] for row in read_jsonl(path)])
        if split == "val":
            report["chosen_threshold"] = choose_threshold(labels, scores, languages)
        report[split] = {
            "reference": rates_at(labels, scores, languages, 0.5),
            "alternative": rates_at(labels, scores, languages, report["chosen_threshold"]),
        }
    write_json(Path(config["output_dir"]) / "eval" / "language_policy.json", report)
    print(report["chosen_threshold"], report["test"])
