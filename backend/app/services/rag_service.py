from __future__ import annotations

import logging
from typing import List, Tuple

from sqlalchemy.orm import Session

from backend.app.core.exceptions import RAGError
from backend.app.infrastructure.db.repositories import (
    DocumentRepository,
    RecommendationLogRepository,
)
from backend.app.infrastructure.llm_client import llm_client
from rag import chains, ingestion, vectorstore_faiss

logger = logging.getLogger("enterprise_ai.rag_service")


class RAGService:
    """Application service orchestrating the RAG pipeline."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._documents = DocumentRepository(db)
        self._logs = RecommendationLogRepository(db)

    def ingest_document(self, *, title: str, text: str, source: str | None, tags: str | None) -> Tuple[int, int]:
        """Persist a document, chunk it, and update the FAISS index."""

        document = self._documents.create_document(title=title, source=source, tags=tags)
        chunks = ingestion.chunk_text(text)
        chunk_records = [(idx, chunk) for idx, chunk in enumerate(chunks)]
        created_chunks = self._documents.add_chunks(document_id=document.id, chunks=chunk_records)

        # Update FAISS index
        vectorstore_faiss.add_document(
            document_id=document.id, chunks=[c.text for c in created_chunks]
        )

        logger.info(
            "Ingested document into RAG store",
            extra={"request_id": "n/a", "document_id": document.id, "n_chunks": len(created_chunks)},
        )
        return document.id, len(created_chunks)

    def list_documents(self) -> List[dict]:
        docs = self._documents.list_documents()
        return [
            {
                "id": d.id,
                "title": d.title,
                "source": d.source,
                "tags": d.tags,
            }
            for d in docs
        ]

    def recommend(self, *, question: str, model_id: int | None) -> tuple[str, list[int]]:
        """Generate a recommendation using the FAISS-backed retriever and LLM."""

        retriever = vectorstore_faiss.as_retriever()
        llm = chains.get_default_llm()
        chain = llm_client.build_recommendation_chain(retriever=retriever, llm=llm)

        try:
            result = chain.invoke({"query": question})
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "RAG recommendation failed", extra={"request_id": "n/a", "question": question}
            )
            raise RAGError("Failed to generate recommendation") from exc

        answer: str = result["result"]  # type: ignore[assignment]
        documents = result.get("source_documents") or []
        document_ids: list[int] = [int(doc.metadata.get("document_id", 0)) for doc in documents]

        summary = llm_client.summarize_sources(documents)
        self._logs.create(
            model_id=model_id,
            document_context_ids=document_ids,
            question=question,
            response_summary=summary,
        )

        return answer, document_ids

