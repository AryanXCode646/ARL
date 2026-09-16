"""Metadata schemas for environments registered with AdaptiveRL."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class EnvironmentMetadata(BaseModel):
    """Metadata describing a registered environment's characteristics and specifications."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique environment identifier, e.g. 'gridworld'")
    description: str = Field("", description="Human-readable description of the environment")
    observation_type: str = Field("unknown", description="Type of observation space: 'discrete', 'box', 'dict', etc.")
    action_type: str = Field("unknown", description="Type of action space: 'discrete', 'box', etc.")
    version: str = Field("0.1.0", description="Environment implementation version")
    max_episode_steps: Optional[int] = Field(None, gt=0, description="Default episode truncation step limit")
    reward_range: Tuple[float, float] = Field(
        (-float("inf"), float("inf")),
        description="Minimum and maximum possible rewards"
    )
    tags: List[str] = Field(default_factory=list, description="Categorization tags, e.g. ['discrete', 'navigation']")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Default parameter dictionary")
