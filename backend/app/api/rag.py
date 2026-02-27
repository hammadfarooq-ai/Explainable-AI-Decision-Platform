from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.exceptions import RAGError
from backend.app.infrastructure.db.base import get_db
from backend.app.schemas import rag as rag_schemas
from backend.app.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["RAG"])


def get_rag_service(db: Session = Depends(get_db)) -> RAGService:
    return RAGService(db=db)


@router.post("/documents", response_model=rag_schemas.DocumentIngestResponse)
def ingest_document(
    request: rag_schemas.DocumentIngestRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> rag_schemas.DocumentIngestResponse:
    """Ingest a textual document into the RAG pipeline."""

    document_id, n_chunks = rag_service.ingest_document(
        title=request.title,
        text=request.text,
        source=request.source,
        tags=request.tags,
    )

    return rag_schemas.DocumentIngestResponse(
        document=rag_schemas.DocumentSummary(
            id=document_id,
            title=request.title,
            source=request.source,
            tags=request.tags,
        ),
        n_chunks=n_chunks,
    )


@router.get("/documents", response_model=rag_schemas.ListDocumentsResponse)
def list_documents(
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> rag_schemas.ListDocumentsResponse:
    """List documents available to the RAG pipeline."""

    docs = rag_service.list_documents()
    return rag_schemas.ListDocumentsResponse(
        documents=[rag_schemas.DocumentSummary(**d) for d in docs]
    )


@router.post("/recommend", response_model=rag_schemas.RecommendationResponse)
def generate_recommendation(
    request: rag_schemas.RecommendationRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> rag_schemas.RecommendationResponse:
    """Generate business recommendations using the RAG pipeline."""

    try:
        answer, document_ids = rag_service.recommend(
            question=request.question,
            model_id=request.model_id,
        )
    except RAGError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return rag_schemas.RecommendationResponse(answer=answer, document_ids=document_ids)

