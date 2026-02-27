from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import optuna
from sklearn.model_selection import cross_val_score

from .config import ProblemType, TrainingConfig


def tune_model(
    *,
    model_builder: Callable[[optuna.Trial], Any],
    X,
    y,
    problem_type: ProblemType,
    training_config: TrainingConfig,
) -> Tuple[Dict[str, Any], float]:
    """Run Optuna-based hyperparameter tuning for a model builder."""

    scoring = "f1_weighted" if problem_type == "classification" else "neg_root_mean_squared_error"

    def objective(trial: optuna.Trial) -> float:
        model = model_builder(trial)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=training_config.n_folds,
            scoring=scoring,
            n_jobs=-1,
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize" if problem_type == "classification" else "minimize"
    )
    study.optimize(
        objective,
        n_trials=training_config.n_trials_per_model,
        timeout=training_config.timeout_per_model,
        show_progress_bar=False,
    )

    return study.best_params, study.best_value

