from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")


class ImageTextEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        if not image_paths:
            return np.empty((0, 0), dtype="float32")
        images = [Image.open(Path(p)).convert("RGB") for p in image_paths]
        vectors = self.model.encode(
            images,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")