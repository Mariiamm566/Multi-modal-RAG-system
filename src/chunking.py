from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class Chunk:
    chunk_id: str
    file_name: str
    page: int
    modality: str
    text: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _sliding_windows(text: str, max_chars: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]
    windows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        windows.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return windows

def build_chunks(elements: list[dict], max_chars: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 0
    for element in elements:
        modality = element["modality"]
        text = element["text"]
        windows = [text] if modality == "table" else _sliding_windows(text, max_chars, overlap)
        for local_idx, window in enumerate(windows, start=1):
            if not window:
                continue
            counter += 1
            prefix = modality.upper()
            normalized = f"[{prefix}] {window}"
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{counter}",
                    file_name=element["file_name"],
                    page=element["page"],
                    modality=modality,
                    text=normalized,
                    meta={**element["meta"], "window": local_idx},
                )
            )
    return chunks
