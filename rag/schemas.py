from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecommendationContext:
    question: str
    model_id: int | None

