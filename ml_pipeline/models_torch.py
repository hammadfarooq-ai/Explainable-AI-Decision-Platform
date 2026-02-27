from __future__ import annotations

from typing import Literal

import torch
from torch import nn

ProblemType = Literal["classification", "regression"]


class TabularMLP(nn.Module):
    """Simple feedforward network for tabular data."""

    def __init__(self, input_dim: int, output_dim: int, problem_type: ProblemType) -> None:
        super().__init__()
        self.problem_type = problem_type
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

