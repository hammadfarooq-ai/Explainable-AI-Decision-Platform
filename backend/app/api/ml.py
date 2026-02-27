from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.exceptions import DatasetNotFoundError, PredictionError, TrainingError
from backend.app.infrastructure.db.base import get_db
from backend.app.schemas import ml as ml_schemas
from backend.app.services.ml_service import MLService
from ml_pipeline import data_ingestion

logger = logging.getLogger("enterprise_ai.api.ml")

router = APIRouter(prefix="/ml", tags=["ML Pipeline"])


def get_ml_service(db: Session = Depends(get_db)) -> MLService:
    return MLService(db=db)


@router.post("/datasets/upload", response_model=ml_schemas.DatasetUploadResponse)
async def upload_dataset(
    *,
    file: UploadFile = File(...),
    target_column: str,
    ml_service: Annotated[MLService, Depends(get_ml_service)],
) -> ml_schemas.DatasetUploadResponse:
    """Upload a CSV dataset and persist basic metadata."""

    settings = get_settings()
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / file.filename
    contents = await file.read()
    file_path.write_bytes(contents)

    df = data_ingestion.load_csv(file_path)
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found in dataset")

    dataset_id, problem_type = ml_service.save_dataset(
        file_path=file_path, name=file.filename, target_column=target_column
    )

    return ml_schemas.DatasetUploadResponse(
        dataset_id=dataset_id,
        name=file.filename,
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        problem_type=problem_type,
    )


@router.get("/datasets/{dataset_id}/eda", response_model=ml_schemas.EdaSummaryResponse)
def get_eda_summary(
    dataset_id: int,
    ml_service: Annotated[MLService, Depends(get_ml_service)],
) -> ml_schemas.EdaSummaryResponse:
    """Return automated EDA summary for a dataset."""

    try:
        summary = ml_service.compute_eda_summary(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ml_schemas.EdaSummaryResponse(
        dataset_id=dataset_id,
        basic_stats=summary["basic_stats"],
        missingness=summary["missingness"],
        target_distribution=summary["target_distribution"],
    )


@router.post("/train", response_model=ml_schemas.TrainResponse)
def train_models(
    request: ml_schemas.TrainRequest,
    ml_service: Annotated[MLService, Depends(get_ml_service)],
) -> ml_schemas.TrainResponse:
    """Trigger model training, evaluation, and selection."""

    try:
        result = ml_service.train_models(request.dataset_id)
    except (DatasetNotFoundError, TrainingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    best = result["best_model"]
    all_models = result["all_models"]

    return ml_schemas.TrainResponse(
        training_run_id=result["training_run_id"],
        problem_type=result["problem_type"],
        best_model=ml_schemas.ModelSummary(**best),
        all_models=[ml_schemas.ModelSummary(**m) for m in all_models],
    )


@router.post("/predict", response_model=ml_schemas.PredictResponse)
def predict(
    request: ml_schemas.PredictRequest,
    ml_service: Annotated[MLService, Depends(get_ml_service)],
) -> ml_schemas.PredictResponse:
    """Generate predictions (and optional SHAP explanations) for input records."""

    try:
        model_id, predictions, probabilities, explanation = ml_service.predict(
            model_id=request.model_id,
            records=request.records,
            explain=request.explain,
        )
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    explanation_schema = None
    if explanation:
        explanation_schema = ml_schemas.PredictionExplanation(
            shap_values=explanation["shap_values"],
            expected_value=explanation.get("expected_value"),
            feature_names=explanation["feature_names"],
        )

    return ml_schemas.PredictResponse(
        model_id=model_id,
        predictions=predictions,
        probabilities=probabilities,
        explanation=explanation_schema,
    )

