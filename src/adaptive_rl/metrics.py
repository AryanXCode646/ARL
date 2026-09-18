"""Canonical episode metrics data structure, accumulator, and extraction contract.

Defines the single source of truth for episodic performance outcomes across
reinforcement learning algorithms, classical planners, and evaluation routines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, SupportsFloat

# Canonical precedence order for episode outcome fields
SUCCESS_KEYS: Sequence[str] = ("episode_success", "success", "is_success")
COLLISION_KEYS: Sequence[str] = ("collision", "is_collision", "had_collision")
OVERFLOW_KEYS: Sequence[str] = ("overflow", "step_overflow", "had_overflow")
TRAFFIC_KEYS: Sequence[str] = (
    "queue_lengths",
    "total_queue",
    "step_overflow",
    "had_overflow",
    "step_controlled",
    "phase_switched",
    "step_departures",
    "step_arrivals",
    "premature_switch",
)

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
        collision: True if collision occurred during the episode, False if collision-free, or None if not applicable.
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


class EpisodeMetricsAccumulator:
    """Accumulates step-level transitions across an entire episode to construct canonical EpisodeMetrics.

    Preserves episode-level semantics across multi-step episodes:
    - Multi-step collision tracking: If a collision occurred on ANY step (intermediate or terminal),
      the episode is marked collision=True. If collision keys are monitored and no collision occurred,
      collision=False. If no collision info was present on any step, collision=None.
    - Multi-step success tracking:
      * Standard / navigation environments: Success is achieved if the agent reached the goal on ANY step.
        If collision occurred at any step, collision overrides success to False (Universal Invariant).
      * Traffic environments: Follows the traffic success contract. An episode is successful iff no queue
        overflow occurred across the entire episode (not had_overflow), the episode did not terminate early
        due to failure/overflow (not terminated; horizon completion is truncated), and the terminal step
        achieved controlled queues (terminal step success is True). Any episode with queue overflow or
        premature termination is strictly marked success=False.
    - Preserves terminal environment telemetry in additional_metrics.
    """

    def __init__(self, is_traffic: Optional[bool] = None) -> None:
        self.is_traffic: Optional[bool] = is_traffic
        self.reward: float = 0.0
        self.length: int = 0
        self.terminated: bool = False
        self.truncated: bool = False

        self.has_collision_info: bool = False
        self.had_collision: bool = False

        self.has_success_info: bool = False
        self.had_success: bool = False

        self.has_overflow_info: bool = False
        self.had_overflow: bool = False

        self.last_info: Dict[str, Any] = {}
        self.step_infos: List[Mapping[str, Any]] = []

    def record_step(
        self,
        reward: float | SupportsFloat = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record a single step transition in the episode."""
        self.reward += float(reward)
        self.length += 1
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)

        if info:
            info_dict = dict(info)
            self.last_info = info_dict
            self.step_infos.append(info_dict)

            # Auto-detect traffic environment if not explicitly set
            if self.is_traffic is None:
                if any(k in info_dict for k in TRAFFIC_KEYS):
                    self.is_traffic = True

            # Track collision across steps
            col_flag = _extract_flag(info_dict, COLLISION_KEYS)
            if col_flag is not None:
                self.has_collision_info = True
                if col_flag:
                    self.had_collision = True

            # Track overflow across steps
            ovf_flag = _extract_flag(info_dict, OVERFLOW_KEYS)
            if ovf_flag is not None:
                self.has_overflow_info = True
                if ovf_flag:
                    self.had_overflow = True

            # Track success across steps
            succ_flag = _extract_flag(info_dict, SUCCESS_KEYS)
            if succ_flag is not None:
                self.has_success_info = True
                if succ_flag:
                    self.had_success = True

    def finish(
        self,
        *,
        additional_metrics: Optional[Mapping[str, Any]] = None,
    ) -> EpisodeMetrics:
        """Finalize accumulated transitions into an immutable EpisodeMetrics instance."""
        # 1. Resolve collision state
        collision: Optional[bool]
        if not self.has_collision_info:
            collision = None
        else:
            collision = self.had_collision

        # 2. Resolve success state according to environment-specific semantics
        success: Optional[bool]
        if self.is_traffic:
            # Traffic success contract:
            # Episode succeeded iff:
            # - No queue overflow occurred at any step (not had_overflow)
            # - Episode did not terminate prematurely (not terminated)
            # - Final queue was controlled (terminal step success is True)
            if self.has_success_info or self.has_overflow_info:
                if self.had_overflow or self.terminated:
                    success = False
                else:
                    # In traffic, success is evaluated at terminal/horizon step
                    term_succ = _extract_flag(self.last_info, SUCCESS_KEYS)
                    success = bool(term_succ) if term_succ is not None else self.had_success
            else:
                success = None
        else:
            # Goal-directed navigation / standard contract:
            # Preserves success signals occurring before the terminal step when not repeated.
            # However, if terminal step explicitly reports success=False, that terminal outcome
            # is authoritative (Invariant 1: Intermediate step success does not latch episode success).
            if not self.has_success_info:
                success = None
            else:
                term_succ = _extract_flag(self.last_info, SUCCESS_KEYS)
                if term_succ is not None:
                    success = term_succ
                else:
                    success = self.had_success

        # 3. Universal invariant: collision overrides success
        if collision is True and success is True:
            success = False

        if additional_metrics is not None:
            extra: Dict[str, Any] = dict(additional_metrics)
        else:
            extra = {k: v for k, v in self.last_info.items() if k not in _EXCLUDED_OUTCOME_KEYS}

        return EpisodeMetrics(
            reward=self.reward,
            length=self.length,
            success=success,
            collision=collision,
            terminated=self.terminated,
            truncated=self.truncated,
            additional_metrics=extra,
        )


def extract_episode_metrics(
    reward: float | SupportsFloat,
    length: int,
    terminated: bool,
    truncated: bool,
    info: Optional[Mapping[str, Any]] = None,
    *,
    additional_metrics: Optional[Mapping[str, Any]] = None,
    step_infos: Optional[Sequence[Mapping[str, Any]]] = None,
    had_collision: Optional[bool] = None,
    had_success: Optional[bool] = None,
    had_overflow: Optional[bool] = None,
    is_traffic: Optional[bool] = None,
) -> EpisodeMetrics:
    """Canonical extractor producing an EpisodeMetrics instance.

    Conforms to strict architectural requirements:
    - Success precedence: `episode_success` -> `success` -> `is_success`
    - Collision precedence: `collision` -> `is_collision` -> `had_collision`
    - Missing metrics remain None (never replaced with False).
    - Explicit False remains False.
    - Multi-step accumulation: supports full step traces via `step_infos` or accumulated
      flags (`had_collision`, `had_success`, `had_overflow`).
    - Traffic success contract: In traffic environments (or when overflow is tracked),
      queue overflow or premature termination strictly forces success to False.
    - Universal Invariant: An episode with collision is never a success. If collision
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
        step_infos: Optional sequence of step info dictionaries accumulated across the episode.
        had_collision: Optional override indicating whether collision occurred during the episode.
        had_success: Optional override indicating whether success was achieved during the episode.
        had_overflow: Optional override indicating whether queue overflow occurred during the episode.
        is_traffic: Optional flag explicitly designating environment as a traffic environment.

    Returns:
        EpisodeMetrics: Immutable, canonical episode metrics instance.
    """
    acc = EpisodeMetricsAccumulator(is_traffic=is_traffic)

    if step_infos:
        for s_info in step_infos:
            acc.record_step(info=s_info)

    if info is not None:
        acc.record_step(info=info)

    acc.reward = float(reward)
    acc.length = int(length)
    acc.terminated = bool(terminated)
    acc.truncated = bool(truncated)

    if had_collision is not None:
        acc.has_collision_info = True
        acc.had_collision = bool(had_collision)

    if had_success is not None:
        acc.has_success_info = True
        acc.had_success = bool(had_success)

    if had_overflow is not None:
        acc.has_overflow_info = True
        acc.had_overflow = bool(had_overflow)

    return acc.finish(additional_metrics=additional_metrics)


__all__ = [
    "COLLISION_KEYS",
    "EpisodeMetrics",
    "EpisodeMetricsAccumulator",
    "OVERFLOW_KEYS",
    "SUCCESS_KEYS",
    "TRAFFIC_KEYS",
    "extract_episode_metrics",
]
