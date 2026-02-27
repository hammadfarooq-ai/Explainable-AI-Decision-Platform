from __future__ import annotations

from typing import Literal

import pandas as pd

ProblemType = Literal["classification", "regression"]


def detect_problem_type(df: pd.DataFrame, target_column: str) -> ProblemType:
    """Infer whether a problem is classification or regression based on the target.

    Heuristic:
    - If the target is numeric and has many unique values relative to dataset size,
      treat as regression.
    - Otherwise treat as classification.
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    target = df[target_column]
    unique_ratio = target.nunique() / max(len(target), 1)

    if pd.api.types.is_numeric_dtype(target) and unique_ratio > 0.2:
        return "regression"
    return "classification"

