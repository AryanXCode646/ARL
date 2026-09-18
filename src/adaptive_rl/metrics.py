"""Canonical episode metrics data structure and extraction contract.

Defines the single source of truth for episodic performance outcomes across
reinforcement learning algorithms, classical planners, and evaluation routines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Canonical precedence order for episode outcome fields
SUCCESS_KEYS: Sequence[str] = ("episode_success", "success", "is_success")
COLLISION_KEYS: Sequence[str] = ("collision", "is_collision", "had_collision")

_EXCLUDED_OUTCOME_KEYS = frozenset(SUCCESS_KEYS).union(COLLISION_KEYS)


def _extract_flag(
    info: Optional[Mapping[str, Any]],
    keys: Sequence[str],
) -> Optional[bool]:
    """Extract a boolean outcome flag following strict precedence.

    Rules:
    - Checks candidate keys in strict sequence order.
    - If a key is present and its value is not None, returns bool(value).
    - Explicit False remains False.
    - If no candidate key is present (or all are None), returns None.
    """
    if not info:
        return None
    for key in keys:
        if key in info and info[key] is not None:
            return bool(info[key])
    return None


@dataclass(frozen=True)
class EpisodeMetrics:
    """Immutable, typed container representing the canonical outcome of a single episode.

    Attributes:
        reward: Total cumulative episodic reward (return).
        length: Total timesteps elapsed in the episode.
        success: True if episode objective reached, False if failed, or None if not applicable.
        collision: True if collision occurred, False if collision-free, or None if not applicable.
        terminated: True if episode ended via natural task termination (Gymnasium return flag).
        truncated: True if episode ended via artificial truncation/timeout (Gymnasium return flag).
        additional_metrics: Environment-specific terminal metrics (e.g. queue length, energy consumed).
    """

    reward: float
    length: int
    success: Optional[bool]
    collision: Optional[bool]
    terminated: bool
    truncated: bool
    additional_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize field types for strict immutability and schema conformity."""
        if not isinstance(self.reward, float):
            object.__setattr__(self, "reward", float(self.reward))
        if not isinstance(self.length, int):
            object.__setattr__(self, "length", int(self.length))
        if self.success is not None and not isinstance(self.success, bool):
            object.__setattr__(self, "success", bool(self.success))
        if self.collision is not None and not isinstance(self.collision, bool):
            object.__setattr__(self, "collision", bool(self.collision))
        if not isinstance(self.terminated, bool):
            object.__setattr__(self, "terminated", bool(self.terminated))
        if not isinstance(self.truncated, bool):
            object.__setattr__(self, "truncated", bool(self.truncated))
        if not isinstance(self.additional_metrics, Mapping):
            raise TypeError(
                f"additional_metrics must be a Mapping, got {type(self.additional_metrics).__name__}"
            )
        # Store a shallow dictionary copy to prevent external mutation of caller dictionaries
        object.__setattr__(self, "additional_metrics", dict(self.additional_metrics))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize episode metrics to a dictionary."""
        return {
            "reward": self.reward,
            "length": self.length,
            "success": self.success,
            "collision": self.collision,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "additional_metrics": dict(self.additional_metrics),
        }


def extract_episode_metrics(
    reward: float,
    length: int,
    terminated: bool,
    truncated: bool,
    info: Optional[Mapping[str, Any]] = None,
    *,
    additional_metrics: Optional[Mapping[str, Any]] = None,
) -> EpisodeMetrics:
    """Canonical extractor producing an EpisodeMetrics instance.

    Conforms to strict architectural requirements:
    - Success precedence: `episode_success` -> `success` -> `is_success`
    - Collision precedence: `collision` -> `is_collision` -> `had_collision`
    - Missing metrics remain None (never replaced with False).
    - Explicit False remains False.
    - Invariant: an episode that ended in collision is never a success. If collision
      occurred (collision is True), any conflicting positive success flag is overridden to False.
    - No heuristics: does not infer success/collision from reward, termination, or length.
    - `terminated` and `truncated` are taken directly from Gymnasium step-return flags.
    - Environment-specific terminal values from `info` are preserved in `additional_metrics`.

    Args:
        reward: Cumulative episodic reward.
        length: Total episode length in timesteps.
        terminated: Gymnasium step terminated flag.
        truncated: Gymnasium step truncated flag.
        info: Terminal step info dictionary from environment.
        additional_metrics: Optional explicit environment-specific metrics mapping.

    Returns:
        EpisodeMetrics: Immutable, canonical episode metrics instance.
    """
    success = _extract_flag(info, SUCCESS_KEYS)
    collision = _extract_flag(info, COLLISION_KEYS)

    # Invariant: An episode that ended in collision is never a success.
    # If collision occurred, it overrides any conflicting positive success flag.
    if collision is True and success is True:
        success = False

    if additional_metrics is not None:
        extra: Dict[str, Any] = dict(additional_metrics)
    elif info is not None:
        extra = {k: v for k, v in info.items() if k not in _EXCLUDED_OUTCOME_KEYS}
    else:
        extra = {}

    return EpisodeMetrics(
        reward=reward,
        length=length,
        success=success,
        collision=collision,
        terminated=terminated,
        truncated=truncated,
        additional_metrics=extra,
    )
