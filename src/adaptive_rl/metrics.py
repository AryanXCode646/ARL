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
    """Stateless strategy for interpreting episode-local facts into outcomes.

    EpisodeMetricsAccumulator owns all episode-local state (rewards, flags, infos).
    OutcomePolicy instances MUST NOT store per-episode facts. They may be shared
    across sequential episodes, vectorized environments, and concurrent accumulators.
    Constructor attributes, if any, may only hold immutable configuration.

    Policy selection is the accumulator's responsibility and is immutable for an
    episode after the first ``record_step`` (or after construction when the policy
    is supplied explicitly).
    """

    __slots__ = ()

    def resolve_collision(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Optional[bool]:
        """Resolve final episode collision outcome from accumulated facts."""
        if not accumulator.has_collision_info:
            return None
        return accumulator.had_collision

    def resolve_success(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Optional[bool]:
        """Resolve final episode success outcome from accumulated facts."""
        if accumulator.explicit_success is not None:
            return accumulator.explicit_success
        if not accumulator.has_success_info:
            return None
        term_succ = _extract_flag(accumulator.last_info, SUCCESS_KEYS)
        if term_succ is not None:
            return term_succ
        return accumulator.had_success

    def get_additional_metrics(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Dict[str, Any]:
        """Return domain-specific metrics to merge into EpisodeMetrics.additional_metrics."""
        return {}


class DefaultOutcomePolicy(OutcomePolicy):
    """Standard goal-directed outcome policy (navigation, continuous control, gridworld).

    Stateless. Safe to share across episodes and accumulators.

    Rules:
    - If no success telemetry was monitored on any step, success is None.
    - If the terminal step explicitly reports success, that terminal outcome is authoritative.
    - If an intermediate step reported success and terminal step did not repeat/contradict it,
      had_success is preserved.
    - Universal Invariant: An episode with collision is overridden to False in the accumulator finish step.
    """

    __slots__ = ()


class TrafficOutcomePolicy(OutcomePolicy):
    """Traffic domain outcome policy governing queue overflow and traffic success semantics.

    Traffic Success Contract:
    An episode succeeded iff:
    - No queue overflow occurred across the entire episode (not accumulator.had_overflow)
    - Episode did not terminate early due to failure/overflow (not accumulator.terminated; horizon completion is truncated)
    - Final queue was controlled (terminal step success is True)
    Any episode with queue overflow or premature termination is strictly marked success=False.
    Intermediate success signals are transient and NEVER latch to establish final episode success.
    If terminal success information is missing (None) and no failure/overflow occurred, success is None.

    Stateless. Safe to share across multiple accumulators, sequential episodes, and
    concurrent environments. All episodic facts are tracked in the episode-local
    EpisodeMetricsAccumulator.
    """

    __slots__ = ()

    def resolve_success(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Optional[bool]:
        # 1. Catastrophic overflow on ANY step of this episode forces success=False
        if accumulator.had_overflow:
            return False

        # 2. Premature failure termination in traffic forces success=False
        if accumulator.terminated:
            return False

        # 3. In traffic, episode outcome is strictly established at the terminal step
        term_succ = _extract_flag(accumulator.last_info, SUCCESS_KEYS)
        if term_succ is not None:
            return bool(term_succ)

        # 4. Check if an explicit episode-level override was provided
        if accumulator.explicit_success is not None:
            return accumulator.explicit_success

        # 5. Otherwise, if terminal step provides no success info, success is undefined (None)
        return None

    def get_additional_metrics(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if accumulator.has_overflow_info:
            extra["had_overflow"] = accumulator.had_overflow
        return extra


class EpisodeMetricsAccumulator:
    """Accumulates step-level transitions across an entire episode to construct canonical EpisodeMetrics.

    Architecture:
    - Accumulator: Collects raw episodic facts across steps (rewards, length, termination, truncation,
      observed telemetry for success, collision, overflow, and environment info mappings).
    - OutcomePolicy: Authoritative domain strategy that interprets the accumulated facts to determine
      success, collision, and domain-specific summary metrics.
    - EpisodeMetrics: Immutable canonical outcome container produced upon calling finish().

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
    - OutcomePolicy instances are stateless and may be shared; this accumulator never writes
      episode facts onto the policy. The bound policy cannot be replaced after the first
      record_step of this episode.
    """

    def __init__(
        self,
        outcome_policy: Optional[OutcomePolicy] = None,
        *,
        is_traffic: Optional[bool] = None,
    ) -> None:
        """Initialize episode metrics accumulator.

        Args:
            outcome_policy: Authoritative domain outcome policy governing outcome interpretation.
            is_traffic: Legacy compatibility parameter. If specified with outcome_policy,
                must match the outcome_policy type, or ValueError is raised.
        """
        if outcome_policy is not None and is_traffic is not None:
            if is_traffic is True and not isinstance(outcome_policy, TrafficOutcomePolicy):
                raise ValueError(
                    f"Conflicting configuration: outcome_policy={type(outcome_policy).__name__} "
                    "is incompatible with is_traffic=True."
                )
            if is_traffic is False and isinstance(outcome_policy, TrafficOutcomePolicy):
                raise ValueError(
                    f"Conflicting configuration: outcome_policy={type(outcome_policy).__name__} "
                    "is incompatible with is_traffic=False."
                )

        if outcome_policy is not None:
            self._outcome_policy: OutcomePolicy = outcome_policy
            self._explicit_policy: bool = True
        elif is_traffic is not None:
            self._outcome_policy = TrafficOutcomePolicy() if is_traffic else DefaultOutcomePolicy()
            self._explicit_policy = True
        else:
            self._outcome_policy = DefaultOutcomePolicy()
            self._explicit_policy = False

        self._policy_locked: bool = False

        self.reward: float = 0.0
        self.length: int = 0
        self.terminated: bool = False
        self.truncated: bool = False

        self.has_collision_info: bool = False
        self.had_collision: bool = False

        self.has_success_info: bool = False
        self.had_success: bool = False
        self.explicit_success: Optional[bool] = None

        self.has_overflow_info: bool = False
        self.had_overflow: bool = False

        self.last_info: Dict[str, Any] = {}
        self.step_infos: List[Mapping[str, Any]] = []

    def _ensure_policy_mutable(self) -> None:
        """Reject policy replacement after the first recorded step of this episode."""
        if self._policy_locked:
            raise RuntimeError(
                "Cannot change OutcomePolicy after the first record_step of an episode. "
                "Policy selection is immutable for the lifetime of an accumulator."
            )

    @property
    def outcome_policy(self) -> OutcomePolicy:
        """Domain policy bound to this episode. Immutable after the first record_step."""
        return self._outcome_policy

    @outcome_policy.setter
    def outcome_policy(self, policy: OutcomePolicy) -> None:
        if policy is self._outcome_policy:
            return
        self._ensure_policy_mutable()
        self._outcome_policy = policy

    @property
    def is_traffic(self) -> bool:
        """Compatibility property indicating whether accumulator is using TrafficOutcomePolicy."""
        return isinstance(self._outcome_policy, TrafficOutcomePolicy)

    @is_traffic.setter
    def is_traffic(self, val: Optional[bool]) -> None:
        """Compatibility setter to toggle traffic policy. Rejects contradictory explicit configuration."""
        self._ensure_policy_mutable()
        if self._explicit_policy:
            if val is True and not isinstance(self._outcome_policy, TrafficOutcomePolicy):
                raise ValueError(
                    "Cannot set is_traffic=True on an accumulator initialized with an explicit non-traffic OutcomePolicy."
                )
            if val is False and isinstance(self._outcome_policy, TrafficOutcomePolicy):
                raise ValueError(
                    "Cannot set is_traffic=False on an accumulator initialized with an explicit TrafficOutcomePolicy."
                )
        if val and not isinstance(self._outcome_policy, TrafficOutcomePolicy):
            self._outcome_policy = TrafficOutcomePolicy()
        elif val is False and isinstance(self._outcome_policy, TrafficOutcomePolicy):
            self._outcome_policy = DefaultOutcomePolicy()

    def _detect_domain_fallback(self, info_dict: Mapping[str, Any]) -> None:
        """Isolated deterministic fallback domain detection for unspecified configuration.

        Runs at most once, on the first record_step, and only when neither
        outcome_policy nor is_traffic was provided at initialization.
        After that step, policy selection is locked for the episode.
        """
        if any(k in info_dict for k in TRAFFIC_KEYS):
            self._outcome_policy = TrafficOutcomePolicy()

    def record_step(
        self,
        reward: float | SupportsFloat = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record a single step transition in the episode (collect raw facts)."""
        self.reward += float(reward)
        self.length += 1
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)

        info_dict = dict(info) if info is not None else {}
        self.last_info = info_dict
        self.step_infos.append(info_dict)

        if not self._policy_locked:
            # First-step fallback only; subsequent steps cannot change the policy.
            if (
                info_dict
                and not self._explicit_policy
                and isinstance(self._outcome_policy, DefaultOutcomePolicy)
            ):
                self._detect_domain_fallback(info_dict)
            self._policy_locked = True

        if info_dict:
            # Track collision facts across steps
            col_flag = _extract_flag(info_dict, COLLISION_KEYS)
            if col_flag is not None:
                self.has_collision_info = True
                if col_flag:
                    self.had_collision = True

            # Track success facts across steps
            succ_flag = _extract_flag(info_dict, SUCCESS_KEYS)
            if succ_flag is not None:
                self.has_success_info = True
                if succ_flag:
                    self.had_success = True

            # Track overflow facts across steps
            ovf_flag = _extract_flag(info_dict, OVERFLOW_KEYS)
            if ovf_flag is not None:
                self.has_overflow_info = True
                if ovf_flag:
                    self.had_overflow = True

    def finish(
        self,
        *,
        additional_metrics: Optional[Mapping[str, Any]] = None,
    ) -> EpisodeMetrics:
        """Finalize accumulated transitions into an immutable EpisodeMetrics instance."""
        # 1. Resolve collision state via outcome policy
        collision: Optional[bool] = self.outcome_policy.resolve_collision(self)

        # 2. Resolve success state via outcome policy
        success: Optional[bool] = self.outcome_policy.resolve_success(self)

        # 3. Universal invariant: collision overrides success
        if collision is True and success is True:
            success = False

        if additional_metrics is not None:
            extra: Dict[str, Any] = dict(additional_metrics)
        else:
            extra = {k: v for k, v in self.last_info.items() if k not in _EXCLUDED_OUTCOME_KEYS}

        # Merge domain policy additional metrics (stateless; facts come from this accumulator)
        policy_extra = self.outcome_policy.get_additional_metrics(self)
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


def _has_traffic_telemetry(info: Mapping[str, Any]) -> bool:
    return any(key in info for key in TRAFFIC_KEYS)


def _resolve_episode_step_infos(
    info: Optional[Mapping[str, Any]],
    step_infos: Optional[Sequence[Mapping[str, Any]]],
) -> List[Mapping[str, Any]]:
    """Build the ordered per-step info sequence for extract_episode_metrics.

    Contract:
    - ``step_infos`` is the ordered sequence of per-step Gymnasium info dicts.
    - ``info`` is the terminal step info dict.
    - If only ``info`` is provided, it is the sole recorded step (terminal-only extraction).
    - If only ``step_infos`` is provided, it is the complete episode including the terminal step.
    - If both are provided, ``step_infos`` may be either the prefix *before* the terminal
      step or the complete episode already ending with ``info``. The terminal mapping is
      recorded exactly once (identity or equal dict as the last ``step_infos`` entry is
      treated as already included).
    """
    recorded: List[Mapping[str, Any]] = list(step_infos) if step_infos else []
    if info is None:
        return recorded
    if recorded and dict(recorded[-1]) == dict(info):
        return recorded
    recorded.append(info)
    return recorded


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

    ``info`` vs ``step_infos``:
        ``info`` is the terminal Gymnasium info dict. ``step_infos`` is the ordered
        per-step info sequence. Passing both does not double-count the terminal
        step: if ``step_infos`` already ends with ``info``, it is replayed as-is;
        otherwise ``info`` is appended as the final step. When the full episode
        trace is available, policy fallback inspects every step before recording
        so domain selection is fixed for the episode.

    Length, reward, terminated, and truncated always come from the explicit
    arguments, not from the number of replayed info dicts.

    Args:
        reward: Cumulative episodic reward.
        length: Total episode length in timesteps.
        terminated: Gymnasium step terminated flag.
        truncated: Gymnasium step truncated flag.
        info: Terminal step info dictionary from environment.
        additional_metrics: Optional explicit environment-specific metrics mapping.
        step_infos: Optional sequence of step info dictionaries. See contract above.
        had_collision: Optional override indicating whether collision occurred during the episode.
        had_success: Optional override indicating whether success was achieved during the episode.
        had_overflow: Optional override indicating whether queue overflow occurred during the episode.
        is_traffic: Optional flag explicitly designating environment as a traffic environment.
        outcome_policy: Optional explicit domain OutcomePolicy instance.

    Returns:
        EpisodeMetrics: Immutable, canonical episode metrics instance.
    """
    resolved_infos = _resolve_episode_step_infos(info, step_infos)

    resolved_is_traffic = is_traffic
    if outcome_policy is None and resolved_is_traffic is None:
        if any(_has_traffic_telemetry(step) for step in resolved_infos):
            resolved_is_traffic = True

    acc = EpisodeMetricsAccumulator(outcome_policy=outcome_policy, is_traffic=resolved_is_traffic)

    for s_info in resolved_infos:
        acc.record_step(info=s_info)

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
        acc.explicit_success = bool(had_success)

    if had_overflow is not None:
        acc.has_overflow_info = True
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
