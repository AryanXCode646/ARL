"""Evaluation scenario schemas for repeatable benchmarks."""

from __future__ import annotations

from typing import Any, Dict
from pydantic import BaseModel, Field


class EvaluationScenario(BaseModel):
    """Declarative specification for an evaluation scenario."""

    name: str = Field(..., description="Scenario name or difficulty tag")
    seed: int = Field(..., description="Deterministic random seed for the scenario")
    environment_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment parameters specifically overridden for this scenario"
    )
