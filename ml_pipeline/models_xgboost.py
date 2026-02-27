from __future__ import annotations

from typing import Literal

from xgboost import XGBClassifier, XGBRegressor

ProblemType = Literal["classification", "regression"]


def build_xgboost(problem_type: ProblemType):
    common_params = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    if problem_type == "classification":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **common_params,
        )
    return XGBRegressor(
        objective="reg:squarederror",
        **common_params,
    )

