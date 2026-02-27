from __future__ import annotations

from typing import Literal

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

ProblemType = Literal["classification", "regression"]


def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, n_jobs=-1)


def build_random_forest(problem_type: ProblemType):
    if problem_type == "classification":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


def build_linear_regression() -> LinearRegression:
    return LinearRegression()

