from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelTrainedEvent:
    training_run_id: int
    model_id: int
    occurred_at: datetime


@dataclass
class DriftDetectedEvent:
    dataset_id: int
    occurred_at: datetime

