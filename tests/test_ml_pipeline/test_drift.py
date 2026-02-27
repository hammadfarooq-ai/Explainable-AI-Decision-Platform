from __future__ import annotations

import pandas as pd

from ml_pipeline.drift import compute_drift_report


def test_compute_drift_report_contains_numeric_columns():
    reference = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": ["x", "y", "z"]})
    new = pd.DataFrame({"a": [2, 3, 4], "b": [0.2, 0.3, 0.4], "c": ["x", "y", "z"]})

    report = compute_drift_report(reference, new)
    assert "a" in report
    assert "b" in report
    assert "c" not in report

