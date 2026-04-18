from __future__ import annotations
from pathlib import Path

from src.config import settings
from src.ingestion import ingest_pdf
from src.chunking import build_chunks
from src.embeddings import TextEmbedder, ImageTextEmbedder
from src.retriever import Retriever
from src.qa import answer_question


class RAGPipeline:
    def __init__(self) -> None:
        settings.ensure()
        self.text_embedder = TextEmbedder(settings.text_embedding_model)
        self.image_embedder = ImageTextEmbedder(settings.image_embedding_model)
        self.retriever = Retriever(self.text_embedder, self.image_embedder)

    def build(self, pdf_paths: list[Path]) -> dict:
        raw = []
        for pdf_path in pdf_paths:
            raw.extend([item.to_dict() for item in ingest_pdf(pdf_path, settings.images_dir)])
        chunks = [c.to_dict() for c in build_chunks(raw, settings.max_chunk_chars, settings.chunk_overlap)]
        fit_stats = self.retriever.fit(chunks)
        self.retriever.save(settings.index_dir)
        return {
            "documents": len(pdf_paths),
            "raw_elements": len(raw),
            "chunks": len(chunks),
            "text_chunks": sum(1 for c in chunks if c["modality"] == "text"),
            "table_chunks": sum(1 for c in chunks if c["modality"] == "table"),
            "image_chunks": sum(1 for c in chunks if c["modality"] == "image"),
            **fit_stats,
            "text_model": settings.text_embedding_model,
            "image_model": settings.image_embedding_model,
        }

    def load(self) -> None:
        self.retriever.load(settings.index_dir)

    def ask(self, query: str) -> tuple[str, list[dict]]:
        clean_query = query.strip()
        if not clean_query:
            return "Please enter a non-empty question.", []

        results = self.retriever.search(
            clean_query,
            top_k_text=settings.top_k_text,
            top_k_image=settings.top_k_image,
            final_top_k=settings.final_top_k,
            min_images=settings.min_images,
            min_score=settings.min_score,
        )

        if not results:
            return "I could not find grounded evidence in the indexed documents to answer this question.", []

        answer = answer_question(
            query=clean_query,
            contexts=results,
            llm_mode=settings.llm_mode,
            endpoint=settings.llm_endpoint,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
        return answer, results