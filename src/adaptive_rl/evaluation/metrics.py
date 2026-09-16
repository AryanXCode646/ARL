"""Evaluation metrics data structures."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Container for standardized reinforcement learning evaluation results."""

    episodes: int = Field(..., gt=0, description="Total evaluation episodes executed")
    mean_reward: float = Field(..., description="Mean cumulative episodic reward")
    std_reward: float = Field(0.0, description="Standard deviation of episodic reward")
    success_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Fraction of episodes reaching target"
    )
    collision_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Fraction of episodes ending in collision"
    )
    mean_episode_length: float = Field(..., ge=0.0, description="Mean step count per episode")
    additional_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment-specific metrics (e.g. energy consumption, path length)",
    )
