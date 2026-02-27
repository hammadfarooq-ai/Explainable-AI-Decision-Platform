from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import mlflow

from backend.app.core.config import get_settings


class MLflowClientWrapper:
    """Thin wrapper around MLflow to centralize configuration and usage patterns."""

    def __init__(self) -> None:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        if settings.mlflow_artifacts_uri:
            mlflow.set_registry_uri(settings.mlflow_artifacts_uri)

    @contextmanager
    def start_run(
        self, run_name: str, tags: Optional[Dict[str, str]] = None
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Context manager to start and end an MLflow run."""

        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            yield run

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log a model object (typically a sklearn or PyTorch model)."""

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )

    def get_run(self, run_id: str) -> mlflow.entities.Run:
        return mlflow.get_run(run_id)

    def load_sklearn_model(self, model_uri: str) -> Any:
        return mlflow.sklearn.load_model(model_uri)


mlflow_client = MLflowClientWrapper()

