from __future__ import annotations

from typing import Literal

from lightgbm import LGBMClassifier, LGBMRegressor

ProblemType = Literal["classification", "regression"]


def build_lightgbm(problem_type: ProblemType):
    common_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    if problem_type == "classification":
        return LGBMClassifier(**common_params)
    return LGBMRegressor(**common_params)

