from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"

for _d in (DATA_DIR, UPLOAD_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "")

    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    clip_top_k: int = int(os.getenv("CLIP_TOP_K", "5"))
    clip_prompt_template: str = "a photo of a {label} houseplant"

    search_results: int = int(os.getenv("SEARCH_RESULTS", "8"))
    max_pages_to_scrape: int = int(os.getenv("MAX_PAGES_TO_SCRAPE", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))

    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "10"))

    def validate(self) -> list[str]:
        missing: list[str] = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.serpapi_api_key:
            missing.append("SERPAPI_API_KEY")
        return missing


settings = Settings()
