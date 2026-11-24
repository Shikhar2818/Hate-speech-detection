"""A small research classifier using frozen TinyCLIP image features and caption TF-IDF."""

from pathlib import Path

import joblib
import numpy as np
from PIL import ImageDraw, ImageOps
from scipy.sparse import csr_matrix, hstack

from hatexai.data.preprocess import normalize_text


def combine(text_features, image_features):
    return hstack((text_features, csr_matrix(image_features)), format="csr")


class VisionEncoder:
    def __init__(self, directory: Path):
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.processor = CLIPImageProcessor.from_pretrained(directory, local_files_only=True)
        self.model = CLIPVisionModelWithProjection.from_pretrained(directory, local_files_only=True).eval()

    def encode(self, images):
        import torch

        # Letterbox the entire meme so captions at the edges remain visible.
        inputs = [ImageOps.pad(image.convert("RGB"), (224, 224), color=(128, 128, 128)) for image in images]
        pixels = self.processor(images=inputs, return_tensors="pt")
        with torch.inference_mode():
            features = self.model(**pixels).image_embeds
        return torch.nn.functional.normalize(features, dim=1).numpy()


def occluded_images(image):
    width, height = image.size
    for row in range(3):
        for col in range(3):
            left, top = col * width // 3, row * height // 3
            right, bottom = (col + 1) * width // 3, (row + 1) * height // 3
            masked = image.copy()
            if left < right and top < bottom:
                ImageDraw.Draw(masked).rectangle((left, top, right - 1, bottom - 1), fill=(128, 128, 128))
            yield masked


class MemePilot:
    def __init__(self, checkpoint: Path, encoder_dir: Path):
        artifact = joblib.load(checkpoint)
        self.text = artifact["vectorizer"]
        self.fusion = artifact["fusion"]
        self.encoder = VisionEncoder(encoder_dir)

    def explain(self, image, caption):
        image = image.convert("RGB")
        batch = [image, *occluded_images(image)]
        text = self.text.transform([normalize_text(caption)] * len(batch))
        probs = self.fusion.predict_proba(combine(text, self.encoder.encode(batch)))[:, 1]
        return {
            "probability_hate": float(probs[0]),
            "decision": "hate" if probs[0] >= 0.5 else "non_hate",
            "region_deltas": (probs[0] - probs[1:]).tolist(),
        }
