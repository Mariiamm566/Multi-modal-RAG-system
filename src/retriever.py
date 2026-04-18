from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np


def _cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_vec = query_vec.astype("float32")
    matrix = matrix.astype("float32")
    query_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
    matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True).T
    denom = np.clip(query_norm * matrix_norm, 1e-12, None)
    return (query_vec @ matrix.T) / denom


class Retriever:
    def __init__(self, text_embedder, image_embedder) -> None:
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.text_records: list[dict[str, Any]] = []
        self.image_records: list[dict[str, Any]] = []
        self.text_matrix: np.ndarray | None = None
        self.image_matrix: np.ndarray | None = None

    def fit(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        self.text_records = []
        self.image_records = []
        for c in chunks:
            item = dict(c)
            item["retrieval_text"] = item["text"]
            if item["modality"] == "image":
                self.image_records.append(item)
            else:
                self.text_records.append(item)

        self.text_matrix = self.text_embedder.encode([c["retrieval_text"] for c in self.text_records]) if self.text_records else None
        self.image_matrix = self.image_embedder.encode_images([c["meta"]["path"] for c in self.image_records]) if self.image_records else None

        return {
            "text_records": len(self.text_records),
            "image_records": len(self.image_records),
            "image_retrieval": "enabled" if self.image_matrix is not None else "disabled",
        }

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "text_records.json").write_text(json.dumps(self.text_records, ensure_ascii=False, indent=2), encoding="utf-8")
        (index_dir / "image_records.json").write_text(json.dumps(self.image_records, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.text_matrix is not None:
            np.save(index_dir / "text_vectors.npy", self.text_matrix)
        if self.image_matrix is not None:
            np.save(index_dir / "image_vectors.npy", self.image_matrix)

    def load(self, index_dir: Path) -> None:
        text_path = index_dir / "text_records.json"
        image_path = index_dir / "image_records.json"
        self.text_records = json.loads(text_path.read_text(encoding="utf-8")) if text_path.exists() else []
        self.image_records = json.loads(image_path.read_text(encoding="utf-8")) if image_path.exists() else []
        self.text_matrix = np.load(index_dir / "text_vectors.npy") if (index_dir / "text_vectors.npy").exists() else None
        self.image_matrix = np.load(index_dir / "image_vectors.npy") if (index_dir / "image_vectors.npy").exists() else None

    def _search_text(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.text_matrix is None or not self.text_records:
            return []
        query_vec = self.text_embedder.encode([query])
        scores = _cosine_scores(query_vec, self.text_matrix)[0]
        ranked = np.argsort(scores)[::-1][:top_k]
        out = []
        for idx in ranked:
            item = dict(self.text_records[idx])
            item["score"] = float(scores[idx])
            item["retrieval_channel"] = "text"
            out.append(item)
        return out

    def _search_images(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.image_matrix is None or not self.image_records:
            return []
        query_vec = self.image_embedder.encode_text([query])
        scores = _cosine_scores(query_vec, self.image_matrix)[0]
        ranked = np.argsort(scores)[::-1][:top_k]
        out = []
        for idx in ranked:
            item = dict(self.image_records[idx])
            item["score"] = float(scores[idx])
            item["retrieval_channel"] = "image"
            out.append(item)
        return out

    def _dedup(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dedup: dict[tuple[str, int, str, str], dict[str, Any]] = {}
        for item in items:
            key = (item["file_name"], item["page"], item["modality"], item["chunk_id"])
            if key not in dedup or item["score"] > dedup[key]["score"]:
                dedup[key] = item
        return sorted(dedup.values(), key=lambda x: x["score"], reverse=True)

    def search(
        self,
        query: str,
        top_k_text: int = 10,
        top_k_image: int = 10,
        final_top_k: int = 6,
        min_images: int = 3,
        min_score: float = 0.20,
    ) -> list[dict[str, Any]]:
        text_results = [item for item in self._dedup(self._search_text(query, top_k_text)) if float(item["score"]) >= min_score]
        image_results = [item for item in self._dedup(self._search_images(query, top_k_image)) if float(item["score"]) >= min_score]
        selected_images = image_results[:min_images]
        remaining_slots = max(0, final_top_k - len(selected_images))
        selected_texts = text_results[:remaining_slots]
        return (selected_images + selected_texts)[:final_top_k]