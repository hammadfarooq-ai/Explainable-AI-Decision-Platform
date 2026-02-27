from __future__ import annotations

from typing import Any


class EnterpriseAIError(Exception):
    """Base class for domain-specific errors in the platform."""

    def __init__(self, message: str, *, details: Any | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class DatasetNotFoundError(EnterpriseAIError):
    """Raised when a dataset cannot be located by ID or path."""


class TrainingError(EnterpriseAIError):
    """Raised when model training fails for a domain-specific reason."""


class PredictionError(EnterpriseAIError):
    """Raised when generating predictions fails."""


class DriftComputationError(EnterpriseAIError):
    """Raised when data drift metrics cannot be computed."""


class RAGError(EnterpriseAIError):
    """Raised when the RAG pipeline encounters a failure."""

