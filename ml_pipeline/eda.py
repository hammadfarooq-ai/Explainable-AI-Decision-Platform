from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def compute_eda_summary(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Return a lightweight automated EDA summary for a dataset.

    The summary intentionally focuses on statistics that can be easily rendered
    on the frontend without server-side plotting.
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    basic_stats = df.describe(include="all").fillna(0).to_dict()

    missingness = df.isna().mean().to_dict()

    if pd.api.types.is_numeric_dtype(df[target_column]):
        target_distribution = df[target_column].describe().to_dict()
    else:
        value_counts = df[target_column].value_counts(normalize=True).to_dict()
        target_distribution = {"value_counts": value_counts}

    return {
        "basic_stats": basic_stats,
        "missingness": missingness,
        "target_distribution": target_distribution,
    }

