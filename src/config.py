from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Settings:
    work_dir: Path = Path("workspace")
    images_dir: Path = Path("workspace/images")
    index_dir: Path = Path("workspace/index")
    max_chunk_chars: int = int(os.getenv("MAX_CHUNK_CHARS", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k_text: int = int(os.getenv("RAG_TOP_K_TEXT", "5"))
    top_k_image: int = int(os.getenv("RAG_TOP_K_IMAGE", "3"))
    final_top_k: int = int(os.getenv("RAG_FINAL_TOP_K", "6"))
    min_images: int = int(os.getenv("RAG_MIN_IMAGES", "3"))
    min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.20"))
    text_embedding_model: str = os.getenv("TEXT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    image_embedding_model: str = os.getenv("IMAGE_EMBEDDING_MODEL", "sentence-transformers/clip-ViT-B-32")
    llm_mode: str = os.getenv("LLM_MODE", "extractive")
    llm_endpoint: str = os.getenv("LLM_ENDPOINT", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")

    def ensure(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()