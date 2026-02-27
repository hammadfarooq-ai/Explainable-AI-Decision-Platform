from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ProblemType = Literal["classification", "regression"]


@dataclass
class TrainingConfig:
    """Configuration options for model training and evaluation."""

    n_folds: int = 3
    random_state: int = 42
    n_trials_per_model: int = 10
    timeout_per_model: int | None = None  # seconds


DEFAULT_TRAINING_CONFIG = TrainingConfig()

