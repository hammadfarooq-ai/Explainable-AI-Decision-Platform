from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import shap


def compute_shap_for_records(model: Any, df: pd.DataFrame) -> Dict[str, Any]:
    """Compute SHAP values for a subset of records using TreeExplainer when possible."""

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:  # noqa: BLE001
        explainer = shap.Explainer(model.predict, df.values)

    shap_values = explainer(df)
    if hasattr(shap_values, "values"):
        values = np.array(shap_values.values).tolist()
    else:
        values = np.array(shap_values).tolist()

    expected_value = None
    if hasattr(shap_values, "base_values"):
        base_values = np.array(shap_values.base_values)
        expected_value = float(base_values.mean()) if base_values.size > 0 else None

    feature_names = list(df.columns)
    return {
        "shap_values": values,
        "expected_value": expected_value,
        "feature_names": feature_names,
    }

