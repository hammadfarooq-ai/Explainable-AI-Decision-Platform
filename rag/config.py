from __future__ import annotations

from dataclasses import dataclass

from backend.app.core.config import get_settings


@dataclass
class RAGSettings:
    """Settings specific to the RAG module."""

    faiss_index_dir: str
    llm_model_name: str


def get_rag_settings() -> RAGSettings:
    settings = get_settings()
    return RAGSettings(
        faiss_index_dir=settings.faiss_index_dir,
        llm_model_name=settings.llm_model_name,
    )

