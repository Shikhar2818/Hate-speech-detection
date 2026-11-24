"""Add a prepared corpus to training while freezing the reference holdouts."""

import argparse
import csv
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hatexai.data.public_pipeline import statistics
from hatexai.data.public_sources import file_digest, load_json, record
from hatexai.data.public_splits import isolate_splits
from hatexai.utils import read_jsonl, write_json, write_jsonl


def verified_path(root, manifest, split):
    entry = manifest["splits"][split]
    path = root / entry["file"]
    if file_digest(path) != entry["sha256"]:
        raise ValueError(f"Checksum mismatch for {path}")
    return path


HINGLISH_REVISION = "c7e831de82ae5894dced27ad66aa840b7735ca3e"
HINGLISH_SHA256 = "4aa51408f8c19f5a4a402447b1d23a54b8bac6b9338bdef2b1382f3bfb703d46"
HINGLISH_SOURCE = "https://github.com/bhargavyagnik/Hateful_speech_Hinglish_social_media_Paper"


def read_hinglish(path):
    """Read author-labeled Hinglish rows from the Srivastava et al. release."""
    rows, excluded = [], Counter()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for index, item in enumerate(csv.DictReader(handle), start=1):
            label = item["label"]
            text = (item["comment"] or "").strip()
            if label not in {"2", "3"} or not text:
                excluded["other_language_or_missing_label_or_text"] += 1
                continue
            row = record("srivastava-hinglish", str(index), "train", text, int(label == "3"),
                         "hi-en", source_label=label, code_mix_evidence="source label")
            rows.append({**row, "code_mixed": True})
    return rows, excluded


def extend_training(reference, extra, output, hinglish=None):
    if output.exists():
        raise FileExistsError(f"Choose a new dataset directory: {output}")
    manifest = load_json(reference / "manifest.json")
    extra_manifest = load_json(extra / "manifest.json")
    paths = {split: verified_path(reference, manifest, split) for split in ("train", "val", "test")}
    extra_rows = read_jsonl(verified_path(extra, extra_manifest, "train"))
    if not extra_rows or any(row["provenance"]["source_split"] != "train" for row in extra_rows):
        raise ValueError("Supplemental rows must come from the original training split")
    manifest["sources"] += extra_manifest["sources"]
    if hinglish is not None:
        if file_digest(hinglish) != HINGLISH_SHA256:
            raise ValueError("Hinglish CSV does not match the pinned authors' release")
        hinglish_rows, excluded = read_hinglish(hinglish)
        extra_rows += hinglish_rows
        manifest["sources"].append({
            "kind": "srivastava-hinglish", "source": HINGLISH_SOURCE,
            "revision": HINGLISH_REVISION, "paper": "https://doi.org/10.1007/978-981-16-3067-5_8",
            "terms": f"{HINGLISH_SOURCE}/blob/{HINGLISH_REVISION}/README.md",
            "license": f"{HINGLISH_SOURCE}/blob/{HINGLISH_REVISION}/LICENSE",
            "label_policy": "Hinglish labels 2 and 3 map to 0 and 1 respectively",
            "files": {"train": {"name": hinglish.name, "sha256": HINGLISH_SHA256}},
            "source_rows": len(hinglish_rows) + sum(excluded.values()), "label_exclusions": dict(excluded),
        })

    rows = [row for path in paths.values() for row in read_jsonl(path)] + extra_rows
    splits, exclusions, matches = isolate_splits(rows, manifest["seed"])
    # Keep parent-text grouping during duplicate checks and calibration, but fit on comments alone.
    extra_ids = {row["id"] for row in extra_rows}
    train = [{**row, "context": ""} if row["id"] in extra_ids else row for row in splits["train"]]
    if {row["label"] for row in train} != {0, 1}:
        raise ValueError("Expanded training split must contain both labels")

    manifest["training_extension"] = {
        "reference_manifest_sha256": file_digest(reference / "manifest.json"),
        "extra_manifest_sha256": file_digest(extra / "manifest.json"),
        "extra_training_rows": len(extra_rows),
        "extra_retained_rows": sum(row["id"] in extra_ids for row in train),
        "supplemental_context": "used for overlap grouping; omitted from training inputs",
    }
    train_exclusions = [row for row in exclusions if row["source_split"] == "train"]
    manifest["overlap_policy"].update({
        "fixed_reference_holdouts": True,
        "matches": matches,
        "excluded_by_split": {"train": dict(Counter(row["reason"] for row in train_exclusions))},
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        build = Path(staging) / "dataset"
        train_path = build / "train.jsonl"
        write_jsonl(train_path, train)
        for split in ("val", "test"):
            shutil.copyfile(paths[split], build / paths[split].name)
        manifest["splits"]["train"] = {
            **statistics(train), "file": train_path.name, "sha256": file_digest(train_path),
        }
        write_jsonl(build / "excluded_overlap.jsonl", train_exclusions)
        write_json(build / "manifest.json", manifest)
        build.rename(output.resolve())
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "extra", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--hinglish", type=Path)
    args = parser.parse_args()
    result = extend_training(args.reference, args.extra, args.output, args.hinglish)
    print(result["splits"]["train"])
