from __future__ import annotations

import logging

from langchain_core.documents import Document

from src.retrieval.chunking import build_splitter
from src.retrieval.vector_store import VectorStore
from src.retrieval.web_search import SearchResult, search_and_scrape

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self) -> None:
        self.splitter = build_splitter()
        self.vector_store = VectorStore()
        self.plant_name: str | None = None
        self.sources: list[str] = []

    def _to_documents(self, results: list[SearchResult]) -> list[Document]:
        docs: list[Document] = []
        for r in results:
            text = r.content or r.snippet
            if not text:
                continue
            for chunk in self.splitter.split_text(text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": r.link, "title": r.title},
                    )
                )
        return docs

    def build_knowledge(self, search_query: str, plant_name: str) -> int:
        self.plant_name = plant_name
        results = search_and_scrape(search_query)
        documents = self._to_documents(results)
        if not documents:
            logger.warning("Brak treści do zaindeksowania dla '%s'", search_query)
            return 0
        self.vector_store.build(documents)
        self.sources = sorted({d.metadata["source"] for d in documents if d.metadata.get("source")})
        logger.info("Zbudowano bazę wiedzy: %d fragmentów, %d źródeł", len(documents), len(self.sources))
        return len(documents)

    def retrieve_context(self, question: str) -> str:
        docs = self.vector_store.retrieve(question)
        if not docs:
            return ""
        parts = []
        for i, d in enumerate(docs, 1):
            title = d.metadata.get("title", "źródło")
            parts.append(f"[Fragment {i} — {title}]\n{d.page_content}")
        return "\n\n".join(parts)

    @property
    def is_ready(self) -> bool:
        return self.vector_store.is_ready
