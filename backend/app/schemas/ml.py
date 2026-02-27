from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DatasetUploadResponse(BaseModel):
    dataset_id: int
    name: str
    n_rows: int
    n_columns: int
    problem_type: Literal["classification", "regression"]


class EdaSummaryResponse(BaseModel):
    dataset_id: int
    basic_stats: Dict[str, Any]
    missingness: Dict[str, float]
    target_distribution: Dict[str, Any]


class TrainRequest(BaseModel):
    dataset_id: int = Field(..., description="Identifier of the dataset to train on.")
    target_column: str = Field(..., description="Name of the target column.")
    problem_type: Optional[Literal["classification", "regression"]] = Field(
        default=None,
        description="Optional override for problem type; if omitted, auto-detected.",
    )


class ModelSummary(BaseModel):
    model_id: int
    algorithm: str
    is_production: bool
    primary_metric: float | None
    metrics: Dict[str, Any] | None


class TrainResponse(BaseModel):
    training_run_id: int
    problem_type: Literal["classification", "regression"]
    best_model: ModelSummary
    all_models: List[ModelSummary]


class PredictRequest(BaseModel):
    model_id: Optional[int] = Field(
        default=None, description="Optional specific model ID; defaults to production model."
    )
    records: List[Dict[str, Any]] = Field(
        ..., description="Records for prediction as a list of feature dictionaries."
    )
    explain: bool = Field(
        default=False,
        description="If true, include SHAP explanations for the provided records.",
    )


class PredictionExplanation(BaseModel):
    shap_values: List[List[float]]
    expected_value: float | None
    feature_names: List[str]


class PredictResponse(BaseModel):
    model_id: int
    predictions: List[Any]
    probabilities: Optional[List[Any]] = None
    explanation: Optional[PredictionExplanation] = None


class DriftReportResponse(BaseModel):
    dataset_id: int
    report: Dict[str, Any]

