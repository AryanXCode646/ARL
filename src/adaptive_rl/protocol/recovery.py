"""Recovery-endpoint computation for the preregistered AdaptiveRL shift protocol.

Implements the normative definitions of
``docs/research/adaptive_rl_hypothesis.md`` (§ Operational Definition):

* ``P_pre`` — mean of the ``K_PRE`` pre-shift (nominal) episode returns.
* ``P0`` — mean of the ``WINDOW`` shock-measurement episodes (post-shift 1..5).
* ``P(t)`` — mean of the causal trailing window of the ``WINDOW`` most recently
  completed post-shift episodes ending at episode ``t`` (first window: t=5,
  which equals ``P0`` exactly, so ``R(5) = 0`` and recovery can never be
  declared at t=5).
* Degradation ``delta = P_pre - P0`` is normalizable only when it exceeds the
  preregistered minimum ``delta_min = 2 * SE(delta)``; otherwise the run is
  flagged and contributes ``T_H = 0`` under an explicit status label.
* The recovery threshold ``R(t) >= 0.9`` is evaluated with the cross-multiplied
  float64 predicate ``10.0 * (p_t - p0) >= 9.0 * delta`` — no division, no
  clipping, no epsilon.
* Persistence: the earliest ``tau`` in ``{6, ..., 13}`` whose three consecutive
  windows (t, t+1, t+2) all meet the predicate; otherwise right-censored at
  ``T_H = HORIZON``.

All inputs are validated: wrong episode counts and non-finite returns raise
``ValueError`` (the protocol maps those conditions to the failure taxonomy
instead of producing silent numbers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from adaptive_rl.metrics import compute_rate
from adaptive_rl.protocol.constants import (
    HORIZON,
    K_PRE,
    MIN_DEGRADATION_SE_MULTIPLIER,
    PERSISTENCE,
    THRESHOLD_CROSS_DENOMINATOR,
    THRESHOLD_CROSS_NUMERATOR,
    WINDOW,
)

#: Recovery statuses. ``recovered`` and ``right_censored`` mean a measurable
#: degradation existed and recovery was (or was not) observed within H.
#: ``no_degradation`` and ``degradation_below_resolution`` mean no measurable
#: degradation: recovery was not required, ``T_H = 0`` by protocol convention,
#: and the status label must be preserved in reporting so this is never
#: conflated with demonstrated adaptation.
STATUS_RECOVERED = "recovered"
STATUS_RIGHT_CENSORED = "right_censored"
STATUS_NO_DEGRADATION = "no_degradation"
STATUS_DEGRADATION_BELOW_RESOLUTION = "degradation_below_resolution"

RECOVERY_STATUSES = (
    STATUS_RECOVERED,
    STATUS_RIGHT_CENSORED,
    STATUS_NO_DEGRADATION,
    STATUS_DEGRADATION_BELOW_RESOLUTION,
)

#: Statuses for which no recovery was measurable (T_H = 0 convention).
NON_RECOVERY_STATUSES = (STATUS_NO_DEGRADATION, STATUS_DEGRADATION_BELOW_RESOLUTION)


@dataclass(frozen=True)
class RecoveryResult:
    """Complete recovery-endpoint computation for one replicate and one arm.

    Attributes:
        p_pre: Mean pre-shift (nominal) return over K_PRE episodes.
        p0: Mean shock-window return over post-shift episodes 1..5.
        degradation: ``p_pre - p0``.
        standard_error: SE of the degradation from the 15 pre + 5 shock returns.
        minimum_degradation: ``2 * standard_error`` (normalization gate).
        recovery_required: Whether the normalization domain was met.
        status: One of ``RECOVERY_STATUSES``.
        tau: Earliest persistence-confirmed recovery episode (6..13) or None.
        truncated_recovery_time: ``T_H`` in {0} U {6..13} U {HORIZON}.
        window_returns: ``P(t)`` for t = 5..15 (length 11).
        recovery_ratios: ``R(t)`` for t = 5..15, None when not normalizable.
        threshold_met: Predicate outcome for each t = 5..15.
    """

    p_pre: float
    p0: float
    degradation: float
    standard_error: float
    minimum_degradation: float
    recovery_required: bool
    status: str
    tau: Optional[int]
    truncated_recovery_time: int
    window_returns: Tuple[float, ...]
    recovery_ratios: Tuple[Optional[float], ...]
    threshold_met: Tuple[bool, ...]


def _validate_returns(values: Sequence[float], label: str) -> List[float]:
    converted = [float(v) for v in values]
    for position, value in enumerate(converted):
        if not math.isfinite(value):
            raise ValueError(f"{label}[{position}] is not finite: {value!r}")
    return converted


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(values) / len(values))


def _sample_variance(values: Sequence[float]) -> float:
    """Unbiased sample variance (ddof=1); 0.0 for fewer than two observations."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = _mean(values)
    return float(math.fsum((v - mean) ** 2 for v in values) / (n - 1))


def trailing_window_means(post_shift_returns: Sequence[float]) -> Tuple[float, ...]:
    """Compute ``P(t)`` for t = WINDOW..HORIZON (length ``HORIZON - WINDOW + 1``).

    Entry ``i`` corresponds to ``t = WINDOW + i`` and is the mean of episodes
    ``t - WINDOW + 1 .. t`` (the causal trailing window).
    """
    returns = _validate_returns(post_shift_returns, "post_shift_returns")
    if len(returns) != HORIZON:
        raise ValueError(f"post_shift_returns must have {HORIZON} entries, got {len(returns)}")
    return tuple(_mean(returns[t - WINDOW : t]) for t in range(WINDOW, HORIZON + 1))


def recovery_threshold_met(p_t: float, p0: float, degradation: float) -> bool:
    """Canonical recovery predicate: ``10.0 * (p_t - p0) >= 9.0 * degradation``.

    Mathematically equivalent to ``R(t) >= 0.9`` but computed without a
    division, so the boundary behavior on float64 inputs is fixed by a single
    multiplication order. ``degradation`` must be positive (the predicate is
    only defined inside the normalization domain).
    """
    if not (degradation > 0.0) or not math.isfinite(degradation):
        raise ValueError(f"degradation must be finite and > 0, got {degradation!r}")
    if not math.isfinite(p_t) or not math.isfinite(p0):
        raise ValueError(f"p_t and p0 must be finite, got p_t={p_t!r}, p0={p0!r}")
    return THRESHOLD_CROSS_NUMERATOR * (p_t - p0) >= THRESHOLD_CROSS_DENOMINATOR * degradation


def compute_recovery(
    pre_shift_returns: Sequence[float],
    post_shift_returns: Sequence[float],
) -> RecoveryResult:
    """Compute the full recovery endpoint for one replicate and one arm.

    ``P0`` is derived from ``post_shift_returns[:WINDOW]`` (post-shift
    episodes 1..5) so the shock window and the post-shift trajectory cannot
    disagree by construction.

    Args:
        pre_shift_returns: Exactly ``K_PRE`` pre-shift episode returns.
        post_shift_returns: Exactly ``HORIZON`` returns for post-shift
            episodes 1..15.

    Returns:
        :class:`RecoveryResult` with status, tau, and T_H.

    Raises:
        ValueError: On wrong episode counts or non-finite returns (the
            protocol maps these to the failure taxonomy, never to numbers).
    """
    pre = _validate_returns(pre_shift_returns, "pre_shift_returns")
    post = _validate_returns(post_shift_returns, "post_shift_returns")

    if len(pre) != K_PRE:
        raise ValueError(f"pre_shift_returns must have {K_PRE} entries, got {len(pre)}")
    if len(post) != HORIZON:
        raise ValueError(f"post_shift_returns must have {HORIZON} entries, got {len(post)}")

    shock = post[:WINDOW]
    p_pre = _mean(pre)
    p0 = _mean(shock)
    degradation = p_pre - p0
    standard_error = math.sqrt(
        _sample_variance(pre) / len(pre) + _sample_variance(shock) / len(shock)
    )
    minimum_degradation = MIN_DEGRADATION_SE_MULTIPLIER * standard_error

    window_returns = trailing_window_means(post)
    # window_returns[i] corresponds to t = WINDOW + i.
    threshold_flags = tuple(
        degradation > 0.0 and recovery_threshold_met(p_t, p0, degradation) for p_t in window_returns
    )
    ratios: Tuple[Optional[float], ...]
    if degradation > 0.0:
        ratios = tuple((p_t - p0) / degradation for p_t in window_returns)
    else:
        ratios = tuple(None for _ in window_returns)

    if degradation <= 0.0:
        status = STATUS_NO_DEGRADATION
        recovery_required = False
        tau: Optional[int] = None
        t_h = 0
    elif degradation < minimum_degradation:
        status = STATUS_DEGRADATION_BELOW_RESOLUTION
        recovery_required = False
        tau = None
        t_h = 0
    else:
        recovery_required = True
        tau = None
        # Eligible starts: t in [WINDOW+1, HORIZON - PERSISTENCE + 1] = 6..13.
        # t = WINDOW is excluded because P(WINDOW) == p0 => R(t) == 0.
        last_start = HORIZON - PERSISTENCE + 1
        for t in range(WINDOW + 1, last_start + 1):
            offsets = [t2 - WINDOW for t2 in range(t, t + PERSISTENCE)]
            if all(threshold_flags[i] for i in offsets):
                tau = t
                break
        if tau is not None:
            status = STATUS_RECOVERED
            t_h = tau
        else:
            status = STATUS_RIGHT_CENSORED
            t_h = HORIZON

    return RecoveryResult(
        p_pre=p_pre,
        p0=p0,
        degradation=degradation,
        standard_error=standard_error,
        minimum_degradation=minimum_degradation,
        recovery_required=recovery_required,
        status=status,
        tau=tau,
        truncated_recovery_time=t_h,
        window_returns=window_returns,
        recovery_ratios=ratios,
        threshold_met=threshold_flags,
    )


def final_window_return(post_shift_returns: Sequence[float]) -> float:
    """Secondary endpoint: mean return over the final WINDOW post-shift episodes.

    Identical by construction to ``P(HORIZON)`` (episodes 11..15 when
    WINDOW=5 and HORIZON=15).
    """
    returns = _validate_returns(post_shift_returns, "post_shift_returns")
    if len(returns) != HORIZON:
        raise ValueError(f"post_shift_returns must have {HORIZON} entries, got {len(returns)}")
    return _mean(returns[len(returns) - WINDOW :])


def final_window_success_rate(success_flags: Sequence[Optional[bool]]) -> Optional[float]:
    """Secondary endpoint: success rate over the final WINDOW post-shift episodes.

    Uses the repository-canonical ``compute_rate`` semantics (``None`` excluded
    from numerator and denominator; ``None`` when no episode in the window has
    a defined outcome).
    """
    flags = list(success_flags)
    if len(flags) != HORIZON:
        raise ValueError(f"success_flags must have {HORIZON} entries, got {len(flags)}")
    return compute_rate(flags[len(flags) - WINDOW :])


__all__ = [
    "NON_RECOVERY_STATUSES",
    "RECOVERY_STATUSES",
    "STATUS_DEGRADATION_BELOW_RESOLUTION",
    "STATUS_NO_DEGRADATION",
    "STATUS_RECOVERED",
    "STATUS_RIGHT_CENSORED",
    "RecoveryResult",
    "compute_recovery",
    "final_window_return",
    "final_window_success_rate",
    "recovery_threshold_met",
    "trailing_window_means",
]
