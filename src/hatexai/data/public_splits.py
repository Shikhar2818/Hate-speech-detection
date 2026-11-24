"""Group possible duplicates before partitioning public dataset releases."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import Counter, defaultdict

from hatexai.data.preprocess import normalize_text


NEAR_TEXT_MIN_LENGTH = 20
NEAR_TEXT_NUMERATOR = 17
NEAR_TEXT_DENOMINATOR = 20
NEAR_IMAGE_DISTANCE = 4
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def canonical_text(text):
    return normalize_text(unicodedata.normalize("NFKC", text).casefold())


def text_matches(texts):
    """Find all qualifying trigram Jaccard pairs using a lossless prefix filter."""
    shingles = {index: {text[i:i + 3] for i in range(len(text) - 2)}
                for index, text in texts.items() if len(text) >= NEAR_TEXT_MIN_LENGTH}
    counts = Counter(gram for grams in shingles.values() for gram in grams)
    inverted = defaultdict(list)
    for index, grams in sorted(shingles.items(), key=lambda item: (len(item[1]), item[0])):
        min_overlap = math.ceil(len(grams) * NEAR_TEXT_NUMERATOR / NEAR_TEXT_DENOMINATOR)
        prefix = sorted(grams, key=lambda gram: (counts[gram], gram))[:len(grams) - min_overlap + 1]
        candidates = {other for gram in prefix for other in inverted[gram]
                      if len(shingles[other]) >= min_overlap}
        for other in candidates:
            overlap = len(grams & shingles[other])
            union = len(grams) + len(shingles[other]) - overlap
            if overlap * NEAR_TEXT_DENOMINATOR >= union * NEAR_TEXT_NUMERATOR:
                yield index, other
        for gram in prefix:
            inverted[gram].append(index)


def image_matches(hashes):
    """Index five disjoint bands so a distance of at most four shares a band."""
    inverted = defaultdict(list)
    for index, fingerprint in hashes.items():
        bands = [(offset, (fingerprint >> offset) & 0x1FFF) for offset in range(0, 64, 13)]
        candidates = {other for band in bands for other in inverted[band]}
        for other in candidates:
            if (fingerprint ^ hashes[other]).bit_count() <= NEAR_IMAGE_DISTANCE:
                yield index, other
        for band in bands:
            inverted[band].append(index)


def isolate_splits(rows, seed):
    rows = sorted(rows, key=lambda row: row["id"])
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source record ID")
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def link(left, right):
        left, right = sorted((root(left), root(right)))
        parents[right] = left

    text_index, image_index, tweet_index, hashes = {}, {}, {}, {}
    texts = [canonical_text(row["text"]) for row in rows]
    matches = Counter()
    for index, (row, canonical) in enumerate(zip(rows, texts)):
        if canonical in text_index:
            link(index, text_index[canonical])
            matches["exact_text"] += 1
        else:
            text_index[canonical] = index
        provenance = row["provenance"]
        tweet_id = provenance.get("tweet_id")
        if tweet_id:
            link(index, tweet_index.setdefault(tweet_id, index))
        sha = provenance.get("image_sha256")
        if sha:
            if sha in image_index:
                link(index, image_index[sha])
                matches["exact_image"] += 1
            else:
                image_index[sha] = index
                hashes[index] = int(provenance["image_dhash"], 16)
    for left, right in text_matches({index: value for value, index in text_index.items()}):
        link(left, right)
        matches["near_text_pairs"] += 1
    for left, right in image_matches(hashes):
        link(left, right)
        matches["near_image_pairs"] += 1

    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[root(index)].append(index)

    # HASOC has no official validation set. Sample whole train-only groups
    # within language/label strata, leaving official divisions untouched.
    strata = defaultdict(list)
    for leader, indices in groups.items():
        members = [rows[index] for index in indices]
        if all(row["provenance"]["dataset"].startswith("hasoc2021-")
               and row["provenance"]["source_split"] == "train" for row in members):
            stratum = Counter((row["language"], row["label"]) for row in members).most_common(1)[0][0]
            rank = hashlib.sha256(f"{seed}:{rows[leader]['id']}".encode()).digest()
            strata[stratum].append((rank, leader))
    generated_val = {leader for stratum in strata.values()
                     for _, leader in sorted(stratum)[:(len(stratum) + 5) // 10]}

    splits = {split: [] for split in SPLIT_PRIORITY}
    exclusions = []
    for leader, indices in groups.items():
        group_id = hashlib.sha256(rows[leader]["id"].encode()).hexdigest()
        target = max((rows[index]["provenance"]["source_split"] for index in indices),
                     key=SPLIT_PRIORITY.__getitem__)
        seen = set()
        for index in indices:
            row = rows[index]
            provenance = row["provenance"]
            exact = (texts[index], provenance.get("image_sha256"), row["label"])
            reason = None
            if provenance["source_split"] != target:
                reason = "cross_split_duplicate_group"
            elif exact in seen and target != "test":
                reason = "same_split_exact_duplicate"
            if reason:
                exclusions.append({"id": row["id"], "source_split": provenance["source_split"],
                                   "group_id": group_id, "reason": reason})
                continue
            seen.add(exact)
            split = "val" if leader in generated_val else target
            splits[split].append({**row, "duplicate_group": group_id})
    return splits, exclusions, dict(matches)
