from __future__ import annotations

import logging
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    logger.info("Ładowanie modelu embeddingów '%s'", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


class VectorStore:
    def __init__(self) -> None:
        self.embeddings = get_embeddings()
        self._store: FAISS | None = None

    def build(self, documents: list[Document]) -> "VectorStore":
        if not documents:
            raise ValueError("Brak dokumentów do zaindeksowania.")
        logger.info("Buduję indeks FAISS z %d fragmentów", len(documents))
        self._store = FAISS.from_documents(documents, self.embeddings)
        return self

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        if self._store is None:
            return []
        k = k or settings.retrieval_k
        return self._store.similarity_search(query, k=k)

    def as_retriever(self, k: int | None = None):
        if self._store is None:
            raise RuntimeError("Indeks nie został jeszcze zbudowany.")
        return self._store.as_retriever(search_kwargs={"k": k or settings.retrieval_k})

    @property
    def is_ready(self) -> bool:
        return self._store is not None
