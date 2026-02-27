from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Optional


ProblemType = Literal["classification", "regression"]


@dataclass
class DatasetMetadata:
    id: int
    name: str
    path: str
    target_column: str
    problem_type: ProblemType
    n_rows: int
    n_columns: int
    created_at: datetime


@dataclass
class ModelMetadata:
    id: int
    training_run_id: int
    algorithm: str
    mlflow_run_id: str
    is_production: bool
    metrics: dict[str, Any] | None
    created_at: datetime


@dataclass
class DriftReportDomain:
    id: int
    dataset_id: int
    report: dict[str, Any]
    created_at: datetime


@dataclass
class RecommendationRecord:
    id: int
    model_id: Optional[int]
    document_ids: list[int]
    question: str
    response_summary: str
    created_at: datetime

