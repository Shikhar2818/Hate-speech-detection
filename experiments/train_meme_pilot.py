"""Fit caption-only, image-only and combined baselines on grouped Multi3Hate memes."""

from io import BytesIO
from pathlib import Path
import sys

import joblib
import numpy as np
from PIL import Image
from pyarrow.parquet import read_table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hatexai.data.preprocess import normalize_text
from hatexai.data.public_sources import file_digest
from hatexai.eval.metrics import compute_classification_metrics
from hatexai.models.meme_pilot import VisionEncoder, combine
from hatexai.utils import write_json


def grouped_split(labels, templates):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    development, test = next(splitter.split(labels, labels, templates))
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train, val = next(splitter.split(labels[development], labels[development], templates[development]))
    return {"train": development[train], "val": development[val], "test": test}


def train(data: Path, encoder_dir: Path, output: Path):
    columns = ["image", "Caption", "US", "Language", "Meme ID", "Template Name"]
    rows = read_table(data, columns=columns, filters=[("Language", "=", "en")]).to_pylist()
    labels = np.array([int(row["US"]) for row in rows])
    templates = np.array([row["Template Name"] for row in rows])
    splits = grouped_split(labels, templates)
    captions = [normalize_text(row["Caption"]) for row in rows]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)
    vectorizer.fit([captions[i] for i in splits["train"]])
    text_features = vectorizer.transform(captions)
    encoder = VisionEncoder(encoder_dir)
    batches = []
    for start in range(0, len(rows), 12):
        images = []
        for row in rows[start:start + 12]:
            with Image.open(BytesIO(row["image"]["bytes"])) as image:
                images.append(image.convert("RGB"))
        batches.append(encoder.encode(images))
    image_features = np.vstack(batches)
    models, results = {}, {}
    for name, features in (
        ("caption", text_features), ("image", image_features),
        ("combined", combine(text_features, image_features)),
    ):
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(features[splits["train"]], labels[splits["train"]])
        models[name] = model
        results[name] = {}
        for split in ("val", "test"):
            index = splits[split]
            probs = model.predict_proba(features[index])
            results[name][split] = compute_classification_metrics(labels[index], probs.argmax(axis=1), probs[:, 1])

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "meme.joblib"
    joblib.dump({"vectorizer": vectorizer, "fusion": models["combined"]}, checkpoint)
    report = {
        "dataset": "Multi3Hate English, US annotations",
        "source_revision": "e3692a908ade3952bd70bb0cc1ca5eac8b23663c",
        "input_sha256": file_digest(data),
        "encoder_sha256": file_digest(encoder_dir / "pytorch_model.bin"),
        "checkpoint_sha256": file_digest(checkpoint),
        "splits": {name: {"n": len(indices), "templates": sorted(set(templates[indices])),
                          "meme_ids": [rows[i]["Meme ID"] for i in indices]}
                   for name, indices in splits.items()},
        "results": results,
    }
    write_json(output / "evaluation.json", report)
    return results


if __name__ == "__main__":
    torch.set_num_threads(2)
    results = train(Path("data/raw/multi3hate/memes.parquet"), Path("outputs/encoders/tinyclip"),
                    Path("outputs/meme-pilot-v1"))
    print({name: round(row["test"]["f1_macro"], 4) for name, row in results.items()})
