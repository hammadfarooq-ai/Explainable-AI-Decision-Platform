from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

import numpy as np
from sklearn import metrics

ProblemType = Literal["classification", "regression"]


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]


def compute_metrics(
    y_true,
    y_pred,
    y_proba=None,
    problem_type: ProblemType = "classification",
) -> Dict[str, float]:
    """Compute standard metrics for classification or regression."""

    if problem_type == "classification":
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average="weighted")
        metric_dict: Dict[str, float] = {"accuracy": acc, "f1": f1}
        if y_proba is not None and y_proba.shape[1] == 2:
            auc = metrics.roc_auc_score(y_true, y_proba[:, 1])
            metric_dict["auc"] = auc
        return metric_dict

    # Regression
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

