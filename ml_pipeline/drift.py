from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute a simple Population Stability Index (PSI) for one feature."""

    expected_perc, _ = np.histogram(expected, bins=bins, range=(expected.min(), expected.max()))
    actual_perc, _ = np.histogram(actual, bins=bins, range=(expected.min(), expected.max()))

    expected_perc = expected_perc / max(expected_perc.sum(), 1)
    actual_perc = actual_perc / max(actual_perc.sum(), 1)

    mask = (expected_perc > 0) & (actual_perc > 0)
    psi = np.sum((actual_perc[mask] - expected_perc[mask]) * np.log(actual_perc[mask] / expected_perc[mask]))
    return float(psi)


def compute_drift_report(reference_df: pd.DataFrame, new_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute a drift report comparing new data to reference data using PSI."""

    common_columns = [c for c in reference_df.columns if c in new_df.columns]
    report: Dict[str, Dict[str, float]] = {}

    for col in common_columns:
        if not pd.api.types.is_numeric_dtype(reference_df[col]):
            continue
        psi = _population_stability_index(
            reference_df[col].dropna().to_numpy(),
            new_df[col].dropna().to_numpy(),
        )
        report[col] = {"psi": psi}

    return report

