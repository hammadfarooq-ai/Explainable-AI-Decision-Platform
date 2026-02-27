from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.exceptions import (
    DatasetNotFoundError,
    PredictionError,
    TrainingError,
)
from backend.app.infrastructure.db.repositories import (
    DatasetRepository,
    DriftReportRepository,
    ModelRepository,
    TrainingRunRepository,
)
from backend.app.infrastructure.redis_cache import redis_cache
from backend.app.infrastructure.mlflow_client import mlflow_client
from ml_pipeline import (
    data_ingestion,
    drift,
    eda,
    evaluation,
    explainability,
    problem_detection,
    training,
)

logger = logging.getLogger("enterprise_ai.ml_service")


ProblemType = Literal["classification", "regression"]


class MLService:
    """Application service orchestrating the ML pipeline."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._settings = get_settings()
        self._datasets = DatasetRepository(db)
        self._training_runs = TrainingRunRepository(db)
        self._models = ModelRepository(db)
        self._drift_reports = DriftReportRepository(db)

    # Dataset and EDA

    def save_dataset(
        self,
        *,
        file_path: Path,
        name: str,
        target_column: str,
    ) -> Tuple[int, ProblemType]:
        """Persist dataset metadata and infer problem type."""

        df = data_ingestion.load_csv(file_path)
        problem_type = problem_detection.detect_problem_type(df, target_column)
        dataset = self._datasets.create(
            name=name,
            path=str(file_path),
            target_column=target_column,
            problem_type=problem_type,
            n_rows=df.shape[0],
            n_columns=df.shape[1],
        )
        logger.info(
            "Saved dataset",
            extra={
                "request_id": "n/a",
                "dataset_id": dataset.id,
                "problem_type": problem_type,
            },
        )
        return dataset.id, problem_type

    def compute_eda_summary(self, dataset_id: int) -> Dict[str, Any]:
        """Return an EDA summary, using Redis caching when available."""

        cache_key = f"eda_summary:{dataset_id}"
        cached = redis_cache.get_json(cache_key)
        if cached is not None:
            return cached

        dataset = self._datasets.get(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"Dataset {dataset_id} not found")
        df = data_ingestion.load_csv(Path(dataset.path))
        summary = eda.compute_eda_summary(df, dataset.target_column)
        # Cache summary for subsequent calls
        redis_cache.set_json(cache_key, summary)
        return summary

    # Training and evaluation

    def train_models(self, dataset_id: int) -> Dict[str, Any]:
        dataset = self._datasets.get(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"Dataset {dataset_id} not found")
        df = data_ingestion.load_csv(Path(dataset.path))

        run = self._training_runs.create(
            dataset_id=dataset.id, problem_type=dataset.problem_type, status="running"
        )

        try:
            with mlflow_client.start_run(
                run_name=f"training_run_{run.id}",
                tags={"dataset_id": str(dataset.id), "problem_type": dataset.problem_type},
            ) as active_run:
                logger.info(
                    "Started training run",
                    extra={
                        "request_id": "n/a",
                        "training_run_id": run.id,
                        "mlflow_run_id": active_run.info.run_id,
                    },
                )

                (
                    model_results,
                    best_model_name,
                    best_model,
                    best_metric,
                ) = training.train_and_select_models(
                    df=df,
                    target_column=dataset.target_column,
                    problem_type=dataset.problem_type,  # type: ignore[arg-type]
                )

                # Log metrics and register models
                for name, result in model_results.items():
                    mlflow_client.log_metrics(
                        {f"{name}_{k}": float(v) for k, v in result.metrics.items()}
                    )

                mlflow_client.log_model(best_model, artifact_path="model")

                model_record = self._models.create(
                    training_run_id=run.id,
                    mlflow_run_id=active_run.info.run_id,
                    algorithm=best_model_name,
                    metrics=model_results[best_model_name].metrics,
                    is_production=True,
                )

                self._training_runs.update_status(
                    run.id, status="completed", primary_metric=best_metric
                )

                logger.info(
                    "Completed training run",
                    extra={
                        "request_id": "n/a",
                        "training_run_id": run.id,
                        "best_model_id": model_record.id,
                    },
                )

                return {
                    "training_run_id": run.id,
                    "problem_type": dataset.problem_type,
                    "best_model": {
                        "model_id": model_record.id,
                        "algorithm": model_record.algorithm,
                        "is_production": model_record.is_production,
                        "primary_metric": model_record.training_run.primary_metric,  # type: ignore[union-attr]
                        "metrics": model_record.metrics,
                    },
                    "all_models": [
                        {
                            "model_id": model_record.id,
                            "algorithm": model_record.algorithm,
                            "is_production": model_record.is_production,
                            "primary_metric": model_record.training_run.primary_metric,  # type: ignore[union-attr]
                            "metrics": model_record.metrics,
                        }
                    ],
                }
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Training failed",
                extra={"request_id": "n/a", "training_run_id": run.id},
            )
            self._training_runs.update_status(run.id, status="failed")
            raise TrainingError("Training failed") from exc

    # Prediction and explainability

    def predict(
        self,
        *,
        model_id: Optional[int],
        records: List[Dict[str, Any]],
        explain: bool,
    ) -> Tuple[int, List[Any], Optional[List[Any]], Optional[Dict[str, Any]]]:
        """Generate predictions for a list of records."""

        if model_id is not None:
            model_record = self._models.get(model_id)
            if not model_record:
                raise PredictionError(f"Model {model_id} not found")
        else:
            model_record = self._models.get_production_model()
            if not model_record:
                raise PredictionError("No production model is available")

        model_uri = f"runs:/{model_record.mlflow_run_id}/model"
        model = mlflow_client.load_sklearn_model(model_uri)

        df = pd.DataFrame.from_records(records)
        predictions = model.predict(df).tolist()
        probabilities: Optional[List[Any]] = None
        explanation: Optional[Dict[str, Any]] = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df).tolist()  # type: ignore[assignment]

        if explain:
            explanation = explainability.compute_shap_for_records(
                model=model,
                df=df,
            )

        return model_record.id, predictions, probabilities, explanation

    # Drift detection

    def compute_drift(self, dataset_id: int, new_data: pd.DataFrame) -> Dict[str, Any]:
        dataset = self._datasets.get(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"Dataset {dataset_id} not found")

        reference_df = data_ingestion.load_csv(Path(dataset.path))
        report = drift.compute_drift_report(reference_df, new_data)
        self._drift_reports.create(dataset_id=dataset.id, report=report)
        return report

