"""Compare the same baseline with and without the CADD parent post."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hatexai.baselines.linear import build_linear_pipeline
from hatexai.data.preprocess import fuse_text
from hatexai.data.public_sources import file_digest, load_json
from hatexai.eval.metrics import compute_classification_metrics
from hatexai.utils import read_jsonl, write_json


def compare(data: Path, output: Path) -> dict:
    manifest = load_json(data / "manifest.json")
    models = {}
    report = {"manifest_sha256": file_digest(data / "manifest.json")}
    for split in ("train", "val", "test"):
        path = data / f"{split}.jsonl"
        if file_digest(path) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split} does not match the CADD manifest")
        rows = read_jsonl(path)
        labels = [row["label"] for row in rows]
        for name, with_context in (("comment_only", False), ("comment_and_parent", True)):
            texts = [fuse_text(row["text"], row["context"] if with_context else "") for row in rows]
            if split == "train":
                models[name] = build_linear_pipeline()
                models[name].fit(texts, labels)
            else:
                model = models[name]
                probs = model.predict_proba(texts)
                report.setdefault(split, {})[name] = compute_classification_metrics(
                    labels, probs.argmax(axis=1), probs[:, 1],
                )
    write_json(output, report)
    return report


if __name__ == "__main__":
    results = compare(Path("data/processed/cadd-v1"), Path("outputs/cadd-context-v1/comparison.json"))
    print({name: round(row["f1_macro"], 4) for name, row in results["test"].items()})
