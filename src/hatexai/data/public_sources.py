"""Readers for the official public benchmark release formats."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from hatexai.data.preprocess import is_code_mixed, normalize_text
from hatexai.utils import read_jsonl


HATEXPLAIN_REVISION = "01d742279dac941981f53806154481c0e15ee686"
HATEXPLAIN_FILES = {
    "dataset.json": "63bb3340fee0ec469b09690d04cb68f7c187787dd8b83807f071892c084967fb",
    "post_id_divisions.json": "c2fb0d89862e7897b11ea3e9380753f15a793482b4b70ad0532dfb1212212835",
    "LICENSE": "56fc351d0e8ebf939208f88cdb27e00833148b957b347a069f38ee88e5fc8db6",
}
HATEXPLAIN_LABELS = {"hatespeech": 1, "offensive": 0, "normal": 0}
HASOC_LABELS = {"HATE": 1, "OFFN": 0, "PRFN": 0}
CADD_REVISION = "090f41c26976b2f763ff445cd6d5840d42a4ddeb"
HIFOX_REVISION = "420bcd5a129be5c9ecf362a0579aafdef30fdc01"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def record(dataset, source_id, split, text, label, language, **source_fields):
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{dataset}:{source_id} has no usable text")
    # Match script on the post, ignoring user names and URLs introduced by the platform.
    mix_text = normalize_text(text).replace("<URL>", "").replace("<USER>", "")
    mix_text = re.sub(r"<(?:url|user|number|percent)>", "", mix_text, flags=re.I)
    script_mix = is_code_mixed(mix_text)
    return {
        "id": f"{dataset}:{source_id}",
        "text": text,
        "label": label,
        "language": language,
        "code_mixed": script_mix,
        "context": "",
        "image_path": None,
        "provenance": {
            "dataset": dataset,
            "source_id": source_id,
            "source_split": split,
            "code_mix_evidence": "script" if script_mix else None,
            **source_fields,
        },
    }


def read_hatexplain(files):
    posts = load_json(files["dataset.json"])
    divisions = load_json(files["post_id_divisions.json"])
    if set(divisions) != {"train", "val", "test"}:
        raise ValueError("HateXplain requires the three published divisions")
    split_by_id = {}
    for split, post_ids in divisions.items():
        for post_id in post_ids:
            if post_id in split_by_id or post_id not in posts:
                raise ValueError(f"Invalid or repeated HateXplain division ID {post_id}")
            split_by_id[post_id] = split
    rows, excluded = [], Counter()
    for post_id, post in posts.items():
        votes = [annotator["label"] for annotator in post["annotators"]]
        if not votes or set(votes) - HATEXPLAIN_LABELS.keys():
            raise ValueError(f"Invalid HateXplain annotations for {post_id}")
        majority, count = Counter(votes).most_common(1)[0]
        split = split_by_id.get(post_id)
        if count * 2 <= len(votes):
            if split is not None:
                raise ValueError(f"HateXplain division contains tied post {post_id}")
            excluded["no_majority"] += 1
            continue
        if split is None:
            raise ValueError(f"Missing official HateXplain division for {post_id}")
        tokens = post["post_tokens"]
        # Keep the authors' rationale masks verbatim. Their list can be shorter than the vote list.
        rows.append(record(
            "hatexplain", post_id, split, " ".join(tokens), HATEXPLAIN_LABELS[majority], "en",
            source_label=majority, annotator_labels=votes, post_tokens=tokens,
            rationale_masks=post["rationales"],
        ))
    return rows, excluded


def read_hasoc(files, language):
    if language not in {"en", "hi"}:
        raise ValueError("HASOC 2021 subtask 1B fine labels are supported for en and hi")
    rows, excluded = [], Counter()
    for split, path in files.items():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
            for item in reader:
                coarse = item["task_1"].strip().upper()
                fine = (item.get("task_2") or "").strip().upper()
                if coarse == "NOT" and fine in {"", "NONE"}:
                    label = 0
                elif coarse == "HOF" and fine in HASOC_LABELS:
                    label = HASOC_LABELS[fine]
                elif coarse == "HOF" and fine in {"", "NONE"}:
                    excluded["coarse_only"] += 1
                    continue
                else:
                    raise ValueError(f"Unexpected HASOC labels {coarse!r}/{fine!r} in {path.name}")
                source_id = next((item[key] for key in ("_id", "hasoc_id", "text_id", "id")
                                  if item.get(key)), None)
                if source_id is None:
                    raise ValueError(f"Missing HASOC ID in {path.name} line {reader.line_num}")
                rows.append(record(
                    f"hasoc2021-{language}", source_id, split, item["text"], label, language,
                    task_1=coarse, source_label=fine or coarse, tweet_id=item.get("tweet_id"),
                ))
    return rows, excluded


def read_hifox(files):
    rows = []
    for split, path in files.items():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                label = item["Hate"]
                if label not in {"0", "1"}:
                    raise ValueError(f"Unexpected HiFoX hate label {label!r}")
                row = record("hifox", item["Id"], split, item["Tweet"], int(label), "hi-en",
                             source_label=label)
                row["code_mixed"] = True
                row["provenance"]["code_mix_evidence"] = "dataset scope"
                rows.append(row)
    return rows, Counter()


def read_cadd(files):
    rows = []
    for split, path in files.items():
        for source_id, item in load_json(path).items():
            label = int(item["L1"])
            if label not in {0, 1, 2, 3}:
                raise ValueError(f"Unexpected CADD L1 label {label!r}")
            row = record("cadd", source_id, split, item["comment"], int(label == 1), "en",
                         source_label=label, target=item.get("L4"))
            row["context"] = "\n".join(value.strip() for value in (item["title"], item["body"])
                                        if value and value.strip())
            rows.append(row)
    return rows, Counter()


def read_hateful_memes(files, image_root):
    from PIL import Image

    rows = []
    for split, path in files.items():
        for item in read_jsonl(path):
            label = item.get("label")
            if (label is None and split != "test") or (
                label is not None and (type(label) is not int or label not in (0, 1))
            ):
                raise ValueError(f"Invalid Hateful Memes label for {item['id']}")
            relative_path = Path(item["img"])
            image_path = image_root / relative_path
            if not image_path.resolve().is_relative_to(image_root.resolve()):
                raise ValueError(f"Image path escapes the authorized dataset directory: {relative_path}")
            with Image.open(image_path) as image:
                # Horizontal difference hash, after validating that the image decodes.
                small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                pixels = small.tobytes()
            image_hash = sum(
                (1 << (8 * y + x))
                for y in range(8) for x in range(8)
                if pixels[9 * y + x] > pixels[9 * y + x + 1]
            )
            row = record(
                "hateful_memes", str(item["id"]), split, item["text"], label, "en",
                source_label=label, source_file=path.name, source_image=relative_path.as_posix(),
                image_sha256=file_digest(image_path), image_dhash=f"{image_hash:016x}",
            )
            row["image_path"] = image_path.as_posix()
            rows.append(row)
    return rows, Counter()
