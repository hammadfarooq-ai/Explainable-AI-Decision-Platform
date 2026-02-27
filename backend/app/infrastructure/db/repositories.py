from __future__ import annotations

from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models


class DatasetRepository:
    """Repository for interacting with Dataset entities."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        *,
        name: str,
        path: str,
        target_column: str,
        problem_type: str,
        n_rows: int,
        n_columns: int,
    ) -> models.Dataset:
        dataset = models.Dataset(
            name=name,
            path=path,
            target_column=target_column,
            problem_type=problem_type,
            n_rows=n_rows,
            n_columns=n_columns,
        )
        self._db.add(dataset)
        self._db.commit()
        self._db.refresh(dataset)
        return dataset

    def get(self, dataset_id: int) -> models.Dataset | None:
        return self._db.get(models.Dataset, dataset_id)


class TrainingRunRepository:
    """Repository for interacting with TrainingRun entities."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        *,
        dataset_id: int,
        problem_type: str,
        status: str,
    ) -> models.TrainingRun:
        run = models.TrainingRun(
            dataset_id=dataset_id,
            problem_type=problem_type,
            status=status,
        )
        self._db.add(run)
        self._db.commit()
        self._db.refresh(run)
        return run

    def update_status(
        self, run_id: int, *, status: str, primary_metric: float | None = None
    ) -> models.TrainingRun:
        run = self._db.get(models.TrainingRun, run_id)
        if not run:
            raise ValueError(f"TrainingRun {run_id} not found")
        run.status = status
        if primary_metric is not None:
            run.primary_metric = primary_metric
        self._db.commit()
        self._db.refresh(run)
        return run


class ModelRepository:
    """Repository for interacting with Model entities."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        *,
        training_run_id: int,
        mlflow_run_id: str,
        algorithm: str,
        metrics: dict | None,
        is_production: bool = False,
    ) -> models.Model:
        model = models.Model(
            training_run_id=training_run_id,
            mlflow_run_id=mlflow_run_id,
            algorithm=algorithm,
            metrics=metrics,
            is_production=is_production,
        )
        self._db.add(model)
        self._db.commit()
        self._db.refresh(model)
        return model

    def list_for_run(self, training_run_id: int) -> Sequence[models.Model]:
        stmt = select(models.Model).where(models.Model.training_run_id == training_run_id)
        return self._db.execute(stmt).scalars().all()

    def get_production_model(self) -> models.Model | None:
        stmt = select(models.Model).where(models.Model.is_production.is_(True))
        return self._db.execute(stmt).scalars().first()


class DriftReportRepository:
    """Repository for interacting with DriftReport entities."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, *, dataset_id: int, report: dict) -> models.DriftReport:
        drift_report = models.DriftReport(dataset_id=dataset_id, report=report)
        self._db.add(drift_report)
        self._db.commit()
        self._db.refresh(drift_report)
        return drift_report


class DocumentRepository:
    """Repository for documents and chunks used by the RAG pipeline."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create_document(
        self, *, title: str, source: str | None, tags: str | None
    ) -> models.Document:
        document = models.Document(title=title, source=source, tags=tags)
        self._db.add(document)
        self._db.commit()
        self._db.refresh(document)
        return document

    def add_chunks(
        self, document_id: int, chunks: Iterable[tuple[int, str]]
    ) -> list[models.DocumentChunk]:
        created_chunks: list[models.DocumentChunk] = []
        for idx, text in chunks:
            chunk = models.DocumentChunk(document_id=document_id, chunk_index=idx, text=text)
            self._db.add(chunk)
            created_chunks.append(chunk)
        self._db.commit()
        for chunk in created_chunks:
            self._db.refresh(chunk)
        return created_chunks

    def list_documents(self) -> Sequence[models.Document]:
        stmt = select(models.Document)
        return self._db.execute(stmt).scalars().all()


class RecommendationLogRepository:
    """Repository for logging RAG recommendations."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        *,
        model_id: int | None,
        document_context_ids: list[int] | None,
        question: str,
        response_summary: str,
    ) -> models.RecommendationLog:
        log = models.RecommendationLog(
            model_id=model_id,
            document_context_ids=document_context_ids,
            question=question,
            response_summary=response_summary,
        )
        self._db.add(log)
        self._db.commit()
        self._db.refresh(log)
        return log

