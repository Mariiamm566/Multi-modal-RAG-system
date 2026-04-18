from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import fitz
import pdfplumber

@dataclass
class RawElement:
    file_name: str
    page: int
    modality: str
    text: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(text: str) -> str:
    return " ".join((text or "").replace("\x00", " ").split())


def _extract_page_text_candidates(page_dict: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines: list[str] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                lines.append(span.get("text", ""))
        merged = _normalize(" ".join(lines))
        if merged:
            candidates.append({"bbox": block.get("bbox", [0, 0, 0, 0]), "text": merged})
    return candidates


def extract_text_blocks(pdf_path: Path) -> list[RawElement]:
    doc = fitz.open(pdf_path)
    items: list[RawElement] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        blocks = page.get_text("blocks")
        for order, block in enumerate(blocks):
            x0, y0, x1, y1, text, *_ = block
            cleaned = _normalize(text)
            if not cleaned:
                continue
            items.append(
                RawElement(
                    file_name=pdf_path.name,
                    page=page_index + 1,
                    modality="text",
                    text=cleaned,
                    meta={"bbox": [x0, y0, x1, y1], "order": order},
                )
            )
    return items


def extract_tables(pdf_path: Path) -> list[RawElement]:
    items: list[RawElement] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for idx, table in enumerate(tables):
                if not table:
                    continue
                headers = ["" if h is None else _normalize(str(h)) for h in (table[0] or [])]
                rows = []
                for row in table[1:] if len(table) > 1 else table:
                    safe_row = ["" if cell is None else _normalize(str(cell)) for cell in row]
                    rows.append(" | ".join(safe_row))
                table_text = []
                if any(headers):
                    table_text.append("Headers: " + " | ".join(headers))
                table_text.extend(rows)
                text = "\n".join(x for x in table_text if x).strip()
                if not text:
                    continue
                items.append(
                    RawElement(
                        file_name=pdf_path.name,
                        page=page_index,
                        modality="table",
                        text=text,
                        meta={
                            "table_index": idx,
                            "rows": len(table),
                            "cols": max((len(r) for r in table), default=0),
                            "headers": headers,
                        },
                    )
                )
    return items


def extract_images(pdf_path: Path, output_dir: Path) -> list[RawElement]:
    doc = fitz.open(pdf_path)
    items: list[RawElement] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_dict = page.get_text("dict")
        page_text = _normalize(page.get_text())
        text_candidates = _extract_page_text_candidates(page_dict)
        for idx, img_info in enumerate(page.get_image_info(xrefs=True), start=1):
            xref = img_info.get("xref")
            if not xref:
                continue
            pix = fitz.Pixmap(doc, xref)
            if pix.colorspace is None:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            elif pix.alpha or pix.colorspace.n != 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_name = f"{pdf_path.stem}_p{page_index+1}_img{idx}.png"
            img_path = output_dir / img_name
            pix.save(str(img_path))
            img_bbox = img_info.get("bbox", (0, 0, 0, 0))
            caption_candidates = []
            for candidate in text_candidates:
                cb = candidate["bbox"]
                vertical_distance = min(abs(cb[1] - img_bbox[3]), abs(cb[3] - img_bbox[1]))
                if vertical_distance < 120:
                    caption_candidates.append((vertical_distance, candidate["text"]))
            caption_candidates.sort(key=lambda x: x[0])
            nearby_text = " ".join(t for _, t in caption_candidates[:3]).strip()
            surrogate = _normalize(
                f"Figure or image on page {page_index + 1}. Nearby caption/context: {nearby_text}. "
                f"Page context: {page_text[:500]}"
            )
            items.append(
                RawElement(
                    file_name=pdf_path.name,
                    page=page_index + 1,
                    modality="image",
                    text=surrogate,
                    meta={
                        "image_index": idx,
                        "path": str(img_path),
                        "xref": xref,
                        "bbox": list(img_bbox),
                        "nearby_text": nearby_text,
                    },
                )
            )
    return items


def ingest_pdf(pdf_path: Path, image_dir: Path) -> list[RawElement]:
    elements: list[RawElement] = []
    elements.extend(extract_text_blocks(pdf_path))
    elements.extend(extract_tables(pdf_path))
    elements.extend(extract_images(pdf_path, image_dir))
    return elements
