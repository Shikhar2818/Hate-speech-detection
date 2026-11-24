"""Build versioned, leakage-checked dataset artifacts from local source files."""

from __future__ import annotations

import hashlib
import tempfile
from collections import Counter
from pathlib import Path

from hatexai.data.public_sources import (
    HATEXPLAIN_FILES,
    HATEXPLAIN_REVISION,
    file_digest,
    read_hasoc,
    read_hateful_memes,
    read_hatexplain,
)
from hatexai.data.public_splits import (
    NEAR_IMAGE_DISTANCE,
    NEAR_TEXT_DENOMINATOR,
    NEAR_TEXT_MIN_LENGTH,
    NEAR_TEXT_NUMERATOR,
    isolate_splits,
)
from hatexai.utils import write_json, write_jsonl


SOURCE_DETAILS = {
    "hatexplain": {
        "source": "https://github.com/hate-alert/HateXplain",
        "terms": f"https://github.com/hate-alert/HateXplain/blob/{HATEXPLAIN_REVISION}/LICENSE",
        "revision": HATEXPLAIN_REVISION,
        "label_policy": "majority of three-way votes; only hatespeech maps to 1",
    },
    "hasoc2021": {
        "source": "https://hasocfire.github.io/hasoc/2021/dataset.html",
        "terms": "https://hasocfire.github.io/hasoc/2021/dataset.html",
        "revision": "FIRE 2021 subtask 1",
        "label_policy": "HATE maps to 1; NOT/OFFN/PRFN map to 0; coarse-only HOF excluded",
    },
    "hateful_memes": {
        "source": "https://www.drivendata.org/competitions/64/hateful-memes/page/206/",
        "terms": "https://www.drivendata.org/competitions/64/hateful-memes/",
        "revision": "Phase 2 unseen divisions",
        "label_policy": "organizers' binary label; unlabeled test stays unlabeled",
    },
}


def statistics(rows):
    return {
        "total": len(rows),
        "labels": dict(Counter(str(row["label"]) if row["label"] is not None else "unlabeled"
                               for row in rows)),
        "languages": dict(Counter(row["language"] for row in rows)),
        "code_mixed": sum(row["code_mixed"] for row in rows),
        "images": sum(row["image_path"] is not None for row in rows),
        "datasets": dict(Counter(row["provenance"]["dataset"] for row in rows)),
        "duplicate_groups": len({row["duplicate_group"] for row in rows}),
    }


def build_public_dataset(profile, output: Path, *, authorized_sources=False):
    if output.exists():
        raise FileExistsError(f"Choose a new dataset output directory: {output}")
    rows, sources = [], []
    for spec in profile["sources"]:
        kind = spec["kind"]
        if kind != "hatexplain" and not authorized_sources:
            raise ValueError(f"Use locally authorized {kind} files and pass --authorized-sources")
        root = Path(spec["root"])
        files = {role: root / name for role, name in spec["files"].items()}
        digests = {role: file_digest(path) for role, path in files.items()}
        if kind == "hatexplain":
            if digests != HATEXPLAIN_FILES:
                raise ValueError("HateXplain inputs do not match the pinned official release")
            loaded, exclusions = read_hatexplain(files)
        elif kind == "hasoc2021":
            if set(files) != {"train", "test"}:
                raise ValueError("HASOC requires the official train and labeled test exports")
            loaded, exclusions = read_hasoc(files, spec["language"])
        elif kind == "hateful_memes":
            if {role: path.name for role, path in files.items()} != {
                "train": "train.jsonl", "val": "dev_unseen.jsonl", "test": "test_unseen.jsonl",
            }:
                raise ValueError("Hateful Memes requires the Phase 2 unseen divisions")
            loaded, exclusions = read_hateful_memes(files, root)
        else:
            raise ValueError(f"Unknown public dataset source {kind!r}")
        if not loaded:
            raise ValueError(f"No usable rows in {kind}")
        for split in ("train", "test"):
            labels = {row["label"] for row in loaded if row["provenance"]["source_split"] == split}
            valid = labels == {0, 1} or (kind == "hateful_memes" and split == "test" and labels == {None})
            if not valid:
                raise ValueError(f"{kind} {split} must contain both labels or a wholly unlabeled meme test")
        source = {
            **SOURCE_DETAILS[kind],
            "kind": kind,
            "language": spec.get("language", "en"),
            "files": {role: {"name": path.name, "sha256": digests[role]} for role, path in files.items()},
            "source_rows": len(loaded) + sum(exclusions.values()),
            "label_exclusions": dict(exclusions),
        }
        if kind == "hateful_memes":
            image_index = "\n".join(sorted(
                f"{row['id']} {row['provenance']['image_sha256']}" for row in loaded
            ))
            source["image_index_sha256"] = hashlib.sha256(image_index.encode()).hexdigest()
        sources.append(source)
        rows.extend(loaded)
    seed = int(profile["seed"])
    splits, exclusions, matches = isolate_splits(rows, seed)
    for split in ("train", "val"):
        if {row["label"] for row in splits[split]} != {0, 1}:
            raise ValueError(f"Usable {split} split must contain both classes after overlap removal")
    test_labels = {row["label"] for row in splits["test"]}
    if test_labels not in ({0, 1}, {None}):
        raise ValueError("Use separate profiles for labeled and unlabeled held-out benchmarks")

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "sources": sources,
        "overlap_policy": {
            "priority": ["test", "val", "train"],
            "hasoc_validation": "10% of eligible training groups per majority language/label stratum",
            "exact_text": "NFKC, case folding and project text normalization",
            "near_text": {
                "min_characters": NEAR_TEXT_MIN_LENGTH,
                "character_trigram_jaccard": NEAR_TEXT_NUMERATOR / NEAR_TEXT_DENOMINATOR,
            },
            "near_image_dhash_distance": NEAR_IMAGE_DISTANCE,
            "retain_official_test_rows": True,
            "matches": matches,
            "excluded_by_split": {
                split: dict(Counter(row["reason"] for row in exclusions if row["source_split"] == split))
                for split in splits
            },
        },
        "splits": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        build = Path(staging) / "dataset"
        for split, records in splits.items():
            name = "test_unlabeled.jsonl" if split == "test" and test_labels == {None} else f"{split}.jsonl"
            path = build / name
            write_jsonl(path, records)
            manifest["splits"][split] = {
                **statistics(records), "file": name, "sha256": file_digest(path),
            }
        write_jsonl(build / "excluded_overlap.jsonl", exclusions)
        write_json(build / "manifest.json", manifest)
        build.rename(output.resolve())
    return manifest
