from __future__ import annotations

import pandas as pd

from ml_pipeline.problem_detection import detect_problem_type


def test_detect_problem_type_classification():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    assert detect_problem_type(df, "target") == "classification"


def test_detect_problem_type_regression():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    assert detect_problem_type(df, "target") == "regression"

