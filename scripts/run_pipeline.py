from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatexai.baselines.linear import load_text_split, train_linear_baseline
from hatexai.eval.metrics import compute_classification_metrics
from hatexai.utils import load_config, maybe_set_hf_offline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hate speech detection pipeline")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--stage", choices=["baseline", "classifier", "dpo", "all"], default="baseline")
    return parser.parse_args()


def eval_split(model, path: str) -> dict:
    texts, labels = load_text_split(path)
    preds = model.predict(texts)
    probs = model.predict_proba(texts)[:, 1].tolist()
    return compute_classification_metrics(labels, preds, probs)


def main() -> None:
    maybe_set_hf_offline()
    args = parse_args()
    config = load_config(args.config)
    results = {}
    if args.stage in {"baseline", "all"}:
        linear = train_linear_baseline(config)
        results["baseline_test"] = eval_split(linear, config["data"]["test_path"])
        print("baseline test:", results["baseline_test"])
    if args.stage in {"classifier", "all"}:
        from hatexai.train.train_classifier import train_classifier

        train_classifier(config)
    if args.stage in {"dpo", "all"}:
        from hatexai.train.train_dpo import train_dpo

        train_dpo(config)
    out = Path(config["output_dir"]) / "eval" / "pipeline_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
