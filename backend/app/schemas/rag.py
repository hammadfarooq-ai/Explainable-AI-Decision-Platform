from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    title: str = Field(..., description="Human-readable title for the document.")
    text: str = Field(..., description="Raw text content of the document.")
    source: Optional[str] = Field(
        default=None, description="Origin of the document (URL, system name, etc.)."
    )
    tags: Optional[str] = Field(default=None, description="Comma-separated tags.")


class DocumentSummary(BaseModel):
    id: int
    title: str
    source: Optional[str]
    tags: Optional[str]


class DocumentIngestResponse(BaseModel):
    document: DocumentSummary
    n_chunks: int


class ListDocumentsResponse(BaseModel):
    documents: List[DocumentSummary]


class RecommendationRequest(BaseModel):
    question: str = Field(..., description="Business question to answer.")
    model_id: Optional[int] = Field(
        default=None,
        description="Optional ID of the predictive model whose context is relevant.",
    )


class RecommendationResponse(BaseModel):
    answer: str
    document_ids: List[int]

