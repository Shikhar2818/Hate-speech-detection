import csv
import json
from itertools import combinations
from pathlib import Path

import pytest
from PIL import Image

from hatexai.data.public_pipeline import build_public_dataset
from hatexai.data.public_sources import file_digest, read_cadd, read_hasoc, read_hatexplain, read_hifox, record
from hatexai.data.public_splits import image_matches, isolate_splits, text_matches
from hatexai.utils import read_jsonl, write_json, write_jsonl


def test_hatexplain_majority_before_mapping_and_raw_rationales(tmp_path):
    def post(labels):
        return {"post_tokens": ["Example", "text"], "annotators": [{"label": x} for x in labels],
                "rationales": [[1, 0], [0, 1]]}

    dataset = tmp_path / "dataset.json"
    divisions = tmp_path / "post_id_divisions.json"
    write_json(dataset, {
        "a": post(["normal", "offensive", "hatespeech"]),
        "b": post(["hatespeech", "hatespeech", "offensive"]),
        "c": post(["offensive", "offensive", "normal"]),
    })
    write_json(divisions, {"train": ["b"], "val": [], "test": ["c"]})
    rows, counts = read_hatexplain({dataset.name: dataset, divisions.name: divisions})
    assert counts == {"no_majority": 1}
    assert [(row["id"], row["label"]) for row in rows] == [("hatexplain:b", 1), ("hatexplain:c", 0)]
    assert rows[0]["provenance"]["rationale_masks"] == [[1, 0], [0, 1]]
    assert "chosen_explanation" not in rows[0]
    write_json(divisions, {"train": ["b"], "val": ["b"], "test": ["c"]})
    with pytest.raises(ValueError, match="repeated"):
        read_hatexplain({dataset.name: dataset, divisions.name: divisions})


def test_hifox_uses_hate_label_and_marks_romanized_code_mixing(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("Id,Tweet,Hate,Fake\n1,aaj nice weather,0,1\n", encoding="utf-8")
    [row], _ = read_hifox({"train": path})
    assert (row["label"], row["language"], row["code_mixed"]) == (0, "hi-en", True)


def test_cadd_keeps_parent_context_in_one_split(tmp_path):
    train, test = tmp_path / "train.json", tmp_path / "test.json"
    parent = {"title": "Weekend plans", "body": "Discuss local events"}
    write_json(train, {"a": {**parent, "comment": "Have a good time", "L1": "0"}})
    write_json(test, {"b": {**parent, "comment": "Different comment", "L1": "1"}})
    rows, _ = read_cadd({"train": train, "test": test})
    assert rows[0]["context"] == "Weekend plans\nDiscuss local events"
    assert [row["label"] for row in rows] == [0, 1]
    splits, _, _ = isolate_splits(rows, 42)
    assert not splits["train"]
    assert splits["test"][0]["id"] == "cadd:b"

def test_hasoc_fine_labels_language_and_coarse_only_exclusion(tmp_path):
    path = tmp_path / "test.tsv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["_id", "text", "task_1", "task_2"])
        writer.writerows([
            ["1", "नमस्ते everyone", "HOF", "HATE"],
            ["2", "नमस्ते https://example.org @example", "HOF", "OFFN"],
            ["3", "other", "HOF", "PRFN"],
            ["4", "neutral", "NOT", "NONE"],
            ["5", "coarse", "HOF", ""],
        ])
    rows, counts = read_hasoc({"test": path}, "hi")
    assert counts == {"coarse_only": 1}
    assert [row["label"] for row in rows] == [1, 0, 0, 0]
    assert [row["code_mixed"] for row in rows] == [True, False, False, False]
    assert {row["language"] for row in rows} == {"hi"}
    assert rows[0]["provenance"]["source_label"] == "HATE"


def test_indexed_similarity_matches_brute_force():
    base = "forty descriptive characters in a sentence for testing"
    texts = {index: base[:index] + "X" + base[index + 1:] for index in range(len(base))}
    texts.update({100: base[:-3], 101: base + "abcdefghijklmnop", 102: "unrelated text"})
    grams = {index: {text[i:i + 3] for i in range(len(text) - 2)}
             for index, text in texts.items() if len(text) >= 20}
    expected = {
        (left, right) for left, right in combinations(sorted(grams), 2)
        if len(grams[left] & grams[right]) / len(grams[left] | grams[right]) >= .85
    }
    assert expected
    assert {tuple(sorted(pair)) for pair in text_matches(texts)} == expected
    hashes = {0: 0, 1: (1 << 13) | (1 << 26) | (1 << 39) | (1 << 52), 2: (1 << 64) - 1}
    assert list(image_matches(hashes)) == [(1, 0)]


def test_overlap_components_preserve_official_holdout_across_sources():
    base = "forty descriptive characters in a sentence for testing"
    rows = [
        record("hatexplain", "a", "test", base, 0, "en"),
        record("hatexplain", "b", "test", base.upper(), 0, "en"),
        record("hasoc2021-hi", "a", "train", base + " today", 1, "hi"),
        record("hatexplain", "c", "val", base + "!", 0, "en"),
        record("hatexplain", "d", "train", "a unique post", 1, "en"),
        record("hatexplain", "e", "train", "a unique post", 1, "en"),
    ]
    splits, exclusions, matches = isolate_splits(rows, 42)
    assert [row["id"] for row in splits["test"]] == ["hatexplain:a", "hatexplain:b"]
    assert [row["id"] for row in splits["train"]] == ["hatexplain:d"]
    assert not splits["val"]
    assert {row["id"] for row in exclusions} == {"hasoc2021-hi:a", "hatexplain:c", "hatexplain:e"}
    assert matches["exact_text"] == 2
    assert isolate_splits(list(reversed(rows)), 42) == (splits, exclusions, matches)


def test_hasoc_validation_draws_whole_training_groups():
    rows = [
        record("hasoc2021-hi", str(i), "train", f"post {i}", i % 2, "hi")
        for i in range(40)
    ]
    rows.append(record("hasoc2021-hi", "copy", "train", "post 5", 0, "hi"))
    splits, _, _ = isolate_splits(rows, 42)
    train_groups = {row["duplicate_group"] for row in splits["train"]}
    val_groups = {row["duplicate_group"] for row in splits["val"]}
    assert len(val_groups) == 4
    assert train_groups.isdisjoint(val_groups)
    assert {row["label"] for row in splits["val"]} == {0, 1}


def test_memes_archive_keeps_unlabeled_test_separate(tmp_path):
    raw = tmp_path / "raw"
    (raw / "img").mkdir(parents=True)
    files = {"train": "train.jsonl", "val": "dev_unseen.jsonl", "test": "test_unseen.jsonl"}
    for i, (split, name) in enumerate(files.items()):
        rows = []
        for label in (0, 1):
            index = i * 2 + label
            relative = Path("img") / f"{index}.png"
            image = Image.new("L", (9, 8))
            image.putdata([255 * ((index >> y) & 1) if x % 2 == 0 else 120
                           for y in range(8) for x in range(9)])
            image.save(raw / relative)
            row = {"id": index, "img": relative.as_posix(), "text": f"meme {index}"}
            if split != "test":
                row["label"] = label
            rows.append(row)
        write_jsonl(raw / name, rows)
    profile = {"seed": 42, "sources": [{"kind": "hateful_memes", "root": raw, "files": files}]}
    output = tmp_path / "final"
    with pytest.raises(ValueError, match="authorized"):
        build_public_dataset(profile, output)
    manifest = build_public_dataset(profile, output, authorized_sources=True)
    holdout = output / "test_unlabeled.jsonl"
    assert not (output / "test.jsonl").exists()
    assert [row["label"] for row in read_jsonl(holdout)] == [None, None]
    assert manifest["splits"]["test"]["sha256"] == file_digest(holdout)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError):
        build_public_dataset(profile, output, authorized_sources=True)
    write_jsonl(raw / files["test"], [{"id": 9, "img": "../outside.png", "text": "invalid"}])
    with pytest.raises(ValueError, match="escapes"):
        build_public_dataset(profile, tmp_path / "invalid", authorized_sources=True)
    assert not (tmp_path / "invalid").exists()
