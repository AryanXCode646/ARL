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


class OutcomePolicy:
    """Base outcome policy governing episode success resolution."""

    def record_step(
        self,
        info: Mapping[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Process a step transition for domain-specific outcome tracking."""
        pass

    def resolve_success(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Optional[bool]:
        """Resolve final episode success outcome."""
        if not accumulator.has_success_info:
            return None
        term_succ = _extract_flag(accumulator.last_info, SUCCESS_KEYS)
        if term_succ is not None:
            return term_succ
        return accumulator.had_success

    def get_additional_metrics(self) -> Dict[str, Any]:
        """Return domain-specific metrics to merge into EpisodeMetrics.additional_metrics."""
        return {}


class DefaultOutcomePolicy(OutcomePolicy):
    """Standard goal-directed outcome policy (navigation, continuous control, gridworld).

    Rules:
    - If no success telemetry was monitored on any step, success is None.
    - If the terminal step explicitly reports success, that terminal outcome is authoritative.
    - If an intermediate step reported success and terminal step did not repeat/contradict it,
      had_success is preserved.
    - An episode with collision is overridden to False in the accumulator finish step (Universal Invariant).
    """

    pass


class TrafficOutcomePolicy(OutcomePolicy):
    """Traffic domain outcome policy governing queue overflow and traffic success semantics.

    Traffic Success Contract:
    An episode succeeded iff:
    - No queue overflow occurred across the entire episode (not had_overflow)
    - Episode did not terminate early due to failure/overflow (not terminated; horizon completion is truncated)
    - Final queue was controlled (terminal step success is True)
    Any episode with queue overflow or premature termination is strictly marked success=False.
    """

    def __init__(self) -> None:
        self.has_overflow_info: bool = False
        self.had_overflow: bool = False

    def record_step(
        self,
        info: Mapping[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> None:
        ovf_flag = _extract_flag(info, OVERFLOW_KEYS)
        if ovf_flag is not None:
            self.has_overflow_info = True
            if ovf_flag:
                self.had_overflow = True

    def resolve_success(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Optional[bool]:
        if accumulator.has_success_info or self.has_overflow_info:
            if self.had_overflow or accumulator.terminated:
                return False
            term_succ = _extract_flag(accumulator.last_info, SUCCESS_KEYS)
            return bool(term_succ) if term_succ is not None else accumulator.had_success
        return None

    def get_additional_metrics(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if self.has_overflow_info:
            extra["had_overflow"] = self.had_overflow
        return extra


class EpisodeMetricsAccumulator:
    """Accumulates step-level transitions across an entire episode to construct canonical EpisodeMetrics.

    Preserves episode-level semantics across multi-step episodes:
    - Multi-step collision tracking: If a collision occurred on ANY step (intermediate or terminal),
      the episode is marked collision=True. If collision keys are monitored and no collision occurred,
      collision=False. If no collision info was present on any step, collision=None.
    - Multi-step success tracking:
      * Standard / navigation environments: Uses DefaultOutcomePolicy where success is preserved from
        intermediate or terminal steps.
      * Traffic environments: Uses TrafficOutcomePolicy enforcing the traffic queue overflow contract.
    - Universal Invariant: collision=True overrides any conflicting positive success flag to False.
    - Preserves terminal environment telemetry in additional_metrics.
    """

    def __init__(
        self,
        outcome_policy: Optional[OutcomePolicy] = None,
        *,
        is_traffic: Optional[bool] = None,
    ) -> None:
        self._explicit_traffic: Optional[bool] = is_traffic
        if outcome_policy is not None:
            self.outcome_policy = outcome_policy
        elif is_traffic:
            self.outcome_policy = TrafficOutcomePolicy()
        else:
            self.outcome_policy = DefaultOutcomePolicy()

        self.reward: float = 0.0
        self.length: int = 0
        self.terminated: bool = False
        self.truncated: bool = False

        self.has_collision_info: bool = False
        self.had_collision: bool = False

        self.has_success_info: bool = False
        self.had_success: bool = False

        self.last_info: Dict[str, Any] = {}
        self.step_infos: List[Mapping[str, Any]] = []

    @property
    def is_traffic(self) -> bool:
        """Indicate whether the accumulator is configured with a TrafficOutcomePolicy."""
        return isinstance(self.outcome_policy, TrafficOutcomePolicy)

    @is_traffic.setter
    def is_traffic(self, val: Optional[bool]) -> None:
        self._explicit_traffic = val
        if val and not isinstance(self.outcome_policy, TrafficOutcomePolicy):
            self.outcome_policy = TrafficOutcomePolicy()
        elif val is False and isinstance(self.outcome_policy, TrafficOutcomePolicy):
            self.outcome_policy = DefaultOutcomePolicy()

    @property
    def has_overflow_info(self) -> bool:
        """Indicate whether queue overflow was monitored during the episode."""
        if isinstance(self.outcome_policy, TrafficOutcomePolicy):
            return self.outcome_policy.has_overflow_info
        return False

    @has_overflow_info.setter
    def has_overflow_info(self, val: bool) -> None:
        if not isinstance(self.outcome_policy, TrafficOutcomePolicy) and val:
            self.outcome_policy = TrafficOutcomePolicy()
        if isinstance(self.outcome_policy, TrafficOutcomePolicy):
            self.outcome_policy.has_overflow_info = val

    @property
    def had_overflow(self) -> bool:
        """Indicate whether queue overflow occurred during the episode."""
        if isinstance(self.outcome_policy, TrafficOutcomePolicy):
            return self.outcome_policy.had_overflow
        return False

    @had_overflow.setter
    def had_overflow(self, val: bool) -> None:
        if not isinstance(self.outcome_policy, TrafficOutcomePolicy) and val:
            self.outcome_policy = TrafficOutcomePolicy()
        if isinstance(self.outcome_policy, TrafficOutcomePolicy):
            self.outcome_policy.had_overflow = val
            if val:
                self.outcome_policy.has_overflow_info = True

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

            # Auto-detect traffic environment if using DefaultOutcomePolicy and not explicitly configured
            if (
                isinstance(self.outcome_policy, DefaultOutcomePolicy)
                and self._explicit_traffic is None
            ):
                if any(k in info_dict for k in TRAFFIC_KEYS):
                    traffic_policy = TrafficOutcomePolicy()
                    for s_info in self.step_infos:
                        traffic_policy.record_step(s_info, False, False)
                    self.outcome_policy = traffic_policy

            # Track collision across steps
            col_flag = _extract_flag(info_dict, COLLISION_KEYS)
            if col_flag is not None:
                self.has_collision_info = True
                if col_flag:
                    self.had_collision = True

            # Track success across steps
            succ_flag = _extract_flag(info_dict, SUCCESS_KEYS)
            if succ_flag is not None:
                self.has_success_info = True
                if succ_flag:
                    self.had_success = True

            # Delegate domain-specific step tracking to outcome policy
            self.outcome_policy.record_step(info_dict, self.terminated, self.truncated)

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

        # 2. Resolve success state via composed outcome policy
        success: Optional[bool] = self.outcome_policy.resolve_success(self)

        # 3. Universal invariant: collision overrides success
        if collision is True and success is True:
            success = False

        if additional_metrics is not None:
            extra: Dict[str, Any] = dict(additional_metrics)
        else:
            extra = {k: v for k, v in self.last_info.items() if k not in _EXCLUDED_OUTCOME_KEYS}

        # Merge domain policy additional metrics
        policy_extra = self.outcome_policy.get_additional_metrics()
        for k, v in policy_extra.items():
            if k not in extra:
                extra[k] = v

        return EpisodeMetrics(
            reward=self.reward,
            length=self.length,
            success=success,
            collision=collision,
            terminated=self.terminated,
            truncated=self.truncated,
            additional_metrics=extra,
        )


def compute_rate(values: Sequence[Optional[bool]]) -> Optional[float]:
    """Compute the rate of True outcomes among defined (non-None) episode outcomes.

    Canonical denominator semantics:
    - Only episodes where the metric is defined (value is not None) contribute to the denominator.
    - True contributes 1 to the numerator.
    - False contributes 0 to the numerator.
    - None is excluded from both numerator and denominator.
    - If no episodes have a defined outcome (empty or all None), returns None.

    Examples:
        >>> compute_rate([True, False, None])
        0.5
        >>> compute_rate([None, None])
        None
        >>> compute_rate([True, True])
        1.0
        >>> compute_rate([False, False])
        0.0
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(sum(1 for v in valid if v is True) / len(valid))


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
    outcome_policy: Optional[OutcomePolicy] = None,
) -> EpisodeMetrics:
    """Canonical convenience extractor producing an EpisodeMetrics instance.

    Thin convenience API wrapping EpisodeMetricsAccumulator. Preserves all
    canonical semantics without duplicating outcome derivation logic.

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
        outcome_policy: Optional explicit domain OutcomePolicy instance.

    Returns:
        EpisodeMetrics: Immutable, canonical episode metrics instance.
    """
    acc = EpisodeMetricsAccumulator(outcome_policy=outcome_policy, is_traffic=is_traffic)

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
        acc.had_overflow = bool(had_overflow)

    return acc.finish(additional_metrics=additional_metrics)


__all__ = [
    "COLLISION_KEYS",
    "DefaultOutcomePolicy",
    "EpisodeMetrics",
    "EpisodeMetricsAccumulator",
    "OVERFLOW_KEYS",
    "OutcomePolicy",
    "SUCCESS_KEYS",
    "TRAFFIC_KEYS",
    "TrafficOutcomePolicy",
    "compute_rate",
    "extract_episode_metrics",
]
