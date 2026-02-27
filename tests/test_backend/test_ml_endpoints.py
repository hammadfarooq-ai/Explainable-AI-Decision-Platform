from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def _write_temp_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "feature_num_1": [1.0, 2.0, 3.0, 4.0],
            "feature_num_2": [10.0, 20.0, 30.0, 40.0],
            "feature_cat": ["A", "B", "A", "B"],
            "target": [0, 1, 0, 1],
        }
    )
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_health_check():
    response = client.get("/api/system/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

