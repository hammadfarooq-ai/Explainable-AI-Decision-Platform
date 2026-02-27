from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import evaluation, feature_engineering, models_lightgbm, models_sklearn, models_xgboost
from .config import DEFAULT_TRAINING_CONFIG, ProblemType, TrainingConfig


def _build_candidate_models(problem_type: ProblemType) -> Dict[str, Any]:
    """Return a dictionary of model name -> estimator without hyperparameter tuning."""

    candidates: Dict[str, Any] = {
        "logistic_regression": models_sklearn.build_logistic_regression()
        if problem_type == "classification"
        else models_sklearn.build_linear_regression(),
        "random_forest": models_sklearn.build_random_forest(problem_type),
        "xgboost": models_xgboost.build_xgboost(problem_type),
        "lightgbm": models_lightgbm.build_lightgbm(problem_type),
    }
    return candidates


def train_and_select_models(
    *,
    df: pd.DataFrame,
    target_column: str,
    problem_type: ProblemType,
    training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
) -> Tuple[Dict[str, evaluation.ModelResult], str, Pipeline, float]:
    """Train multiple models and select the best one on a validation split."""

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=training_config.random_state,
        stratify=y if problem_type == "classification" else None,
    )

    preprocessor, _ = feature_engineering.build_preprocessing_pipeline(df, target_column)
    candidates = _build_candidate_models(problem_type)

    results: Dict[str, evaluation.ModelResult] = {}
    best_name = ""
    best_score: float | None = None
    best_pipeline: Pipeline | None = None

    for name, model in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_valid)
        y_proba = None
        if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_valid)

        metric_dict = evaluation.compute_metrics(
            y_true=y_valid, y_pred=y_pred, y_proba=y_proba, problem_type=problem_type
        )
        results[name] = evaluation.ModelResult(name=name, metrics=metric_dict)

        primary_metric_name = "f1" if problem_type == "classification" else "rmse"
        primary_metric = metric_dict[primary_metric_name]

        if best_score is None:
            best_score = primary_metric
            best_name = name
            best_pipeline = pipeline
        else:
            if problem_type == "classification":
                if primary_metric > best_score:
                    best_score = primary_metric
                    best_name = name
                    best_pipeline = pipeline
            else:
                if primary_metric < best_score:
                    best_score = primary_metric
                    best_name = name
                    best_pipeline = pipeline

    assert best_pipeline is not None
    assert best_score is not None

    return results, best_name, best_pipeline, float(best_score)

