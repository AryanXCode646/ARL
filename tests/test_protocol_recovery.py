"""Hand-verified tests for the recovery endpoint (Issue #98 / PR #168).

Every numbered test corresponds to a worked example in
``docs/research/adaptive_rl_hypothesis.md`` (§ Worked examples); the document
and this file must stay in exact agreement.
"""

from __future__ import annotations

import math

import pytest

from adaptive_rl.protocol import (
    HORIZON,
    K_PRE,
    STATUS_DEGRADATION_BELOW_RESOLUTION,
    STATUS_NO_DEGRADATION,
    STATUS_RECOVERED,
    STATUS_RIGHT_CENSORED,
    WINDOW,
    compute_recovery,
    final_window_return,
    final_window_success_rate,
    recovery_threshold_met,
    trailing_window_means,
)


def test_case_1a_immediate_no_degradation_equal_returns() -> None:
    result = compute_recovery([10.0] * K_PRE, [10.0] * HORIZON)
    assert result.status == STATUS_NO_DEGRADATION
    assert result.recovery_required is False
    assert result.tau is None
    assert result.truncated_recovery_time == 0
    assert result.degradation == 0.0
    assert all(ratio is None for ratio in result.recovery_ratios)


def test_case_1b_immediate_no_degradation_improved_returns() -> None:
    result = compute_recovery([8.0] * K_PRE, [10.0] * HORIZON)
    assert result.status == STATUS_NO_DEGRADATION
    assert result.degradation == -2.0
    assert result.truncated_recovery_time == 0


def test_case_2_recovery_at_earliest_eligible_t() -> None:
    post = [5.0] * 5 + [30.0] * 10
    result = compute_recovery([10.0] * K_PRE, post)
    assert result.status == STATUS_RECOVERED
    assert result.tau == 6
    assert result.truncated_recovery_time == 6
    # P(5) == P0 identically, so R(5) == 0 and t = 5 can never be eligible.
    assert result.window_returns[0] == result.p0
    assert result.recovery_ratios[0] == 0.0


def test_case_3_recovery_at_t13() -> None:
    post = [5.0] * 12 + [30.0] * 3
    result = compute_recovery([10.0] * K_PRE, post)
    assert result.status == STATUS_RECOVERED
    assert result.tau == 13
    assert result.truncated_recovery_time == 13


def test_case_4_never_recovers_right_censored_at_horizon() -> None:
    result = compute_recovery([10.0] * K_PRE, [5.0] * HORIZON)
    assert result.status == STATUS_RIGHT_CENSORED
    assert result.tau is None
    assert result.truncated_recovery_time == HORIZON == 15


def test_case_5_threshold_exactly_point_nine() -> None:
    # P(t) = 9, P0 = 0, degradation = 10 => R = 0.9 exactly at t = 10..12.
    post = [0.0] * 5 + [9.0] * 10
    result = compute_recovery([10.0] * K_PRE, post)
    assert result.status == STATUS_RECOVERED
    assert result.tau == 10
    for t in (10, 11, 12):
        assert result.recovery_ratios[t - WINDOW] == 0.9
        assert result.threshold_met[t - WINDOW] is True


def test_case_6_threshold_boundary_floating_point_tolerance() -> None:
    assert recovery_threshold_met(9.0, 0.0, 10.0) is True
    assert recovery_threshold_met(9.000000000000002, 0.0, 10.0) is True
    assert recovery_threshold_met(8.999999999999998, 0.0, 10.0) is False


def test_case_7_recovery_ratio_can_be_negative() -> None:
    post = [10.0] * 5 + [5.0] * 10
    result = compute_recovery([20.0] * K_PRE, post)
    assert result.recovery_ratios[1] == pytest.approx(-0.1)
    assert result.recovery_ratios[1] < 0.0
    assert result.status == STATUS_RIGHT_CENSORED


def test_case_8_recovery_ratio_can_exceed_one() -> None:
    post = [5.0] * 5 + [20.0] * 10
    result = compute_recovery([10.0] * K_PRE, post)
    assert result.recovery_ratios[-1] == pytest.approx(3.0)
    assert result.recovery_ratios[-1] > 1.0
    assert result.status == STATUS_RECOVERED


def test_case_9_zero_denominator_is_no_degradation() -> None:
    result = compute_recovery([10.0] * K_PRE, [10.0] * HORIZON)
    assert result.degradation == 0.0
    assert result.status == STATUS_NO_DEGRADATION
    assert result.truncated_recovery_time == 0


def test_case_10_denominator_below_minimum_measurable_degradation() -> None:
    # degradation = 1.0; var(shock)=2.5 (ddof=1) => SE = sqrt(2.5/5 + 0/15)
    # = sqrt(0.5) => delta_min = 2*sqrt(0.5) = sqrt(2) > 1.0.
    post = [7.0, 8.0, 9.0, 10.0, 11.0] + [5.0] * 10
    result = compute_recovery([10.0] * K_PRE, post)
    assert result.degradation == 1.0
    assert result.minimum_degradation == pytest.approx(2.0 * math.sqrt(0.5))
    assert result.degradation < result.minimum_degradation
    assert result.status == STATUS_DEGRADATION_BELOW_RESOLUTION
    assert result.recovery_required is False
    assert result.truncated_recovery_time == 0


def test_case_10b_small_but_resolvable_denominator_is_normalizable() -> None:
    post = [8.8, 9.0, 9.0, 9.0, 9.2] + [5.0] * 10
    result = compute_recovery([10.0] * K_PRE, post)
    assert result.degradation == 1.0
    assert result.degradation >= result.minimum_degradation
    assert result.recovery_required is True


def test_case_11_persistence_broken_at_t_plus_one() -> None:
    post = [0.0] * 5 + [45.0, -50.0] + [100.0] * 8
    result = compute_recovery([10.0] * K_PRE, post)
    assert result.threshold_met[6 - WINDOW] is True  # t = 6 passes
    assert result.threshold_met[7 - WINDOW] is False  # t = 7 breaks persistence
    assert result.status == STATUS_RECOVERED
    assert result.tau == 8
    assert result.truncated_recovery_time == 8


def test_case_12_persistence_achieved_for_first_time_later() -> None:
    post = [0.0] * 5 + [40.0, 50.0, 0.0, -50.0] + [100.0] * 6
    result = compute_recovery([10.0] * K_PRE, post)
    # Isolated passes at t = 7 and t = 8 are broken at t = 9; first full
    # three-window confirmation starts at t = 10.
    assert result.threshold_met[7 - WINDOW] is True
    assert result.threshold_met[8 - WINDOW] is True
    assert result.threshold_met[9 - WINDOW] is False
    assert result.tau == 10
    assert result.truncated_recovery_time == 10


def test_tau_always_inside_preregistered_domain() -> None:
    cases = [
        [5.0] * 5 + [30.0] * 10,
        [5.0] * 12 + [30.0] * 3,
        [5.0] * 15,
        [0.0] * 5 + [9.0] * 10,
        [10.0] * 15,
    ]
    for post in cases:
        result = compute_recovery([10.0] * K_PRE, post)
        assert result.truncated_recovery_time in {0, *range(6, 14), 15}
        if result.tau is not None:
            assert 6 <= result.tau <= 13
        if result.status == STATUS_RECOVERED:
            assert result.truncated_recovery_time == result.tau


def test_window_means_definition() -> None:
    post = [float(i) for i in range(1, HORIZON + 1)]
    windows = trailing_window_means(post)
    assert len(windows) == HORIZON - WINDOW + 1 == 11
    assert windows[0] == pytest.approx(sum(post[0:5]) / 5)
    assert windows[-1] == pytest.approx(sum(post[10:15]) / 5)


def test_final_window_return_equals_p_of_horizon() -> None:
    post = [float(i) for i in range(1, HORIZON + 1)]
    result = compute_recovery([20.0] * K_PRE, post)
    assert final_window_return(post) == result.window_returns[-1]


def test_final_window_success_rate_semantics() -> None:
    assert final_window_success_rate([True] * HORIZON) == 1.0
    assert final_window_success_rate([False] * HORIZON) == 0.0
    assert final_window_success_rate([None] * HORIZON) is None
    flags = [None] * 10 + [True, False, None, None, None]
    assert final_window_success_rate(flags) == 0.5
    with pytest.raises(ValueError, match="success_flags"):
        final_window_success_rate([True] * (HORIZON - 1))


def test_wrong_episode_counts_raise() -> None:
    with pytest.raises(ValueError, match="pre_shift_returns"):
        compute_recovery([10.0] * (K_PRE - 1), [10.0] * HORIZON)
    with pytest.raises(ValueError, match="post_shift_returns"):
        compute_recovery([10.0] * K_PRE, [10.0] * (HORIZON - 1))


def test_non_finite_returns_raise() -> None:
    pre = [10.0] * K_PRE
    pre[3] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        compute_recovery(pre, [10.0] * HORIZON)
    post = [10.0] * HORIZON
    post[7] = float("inf")
    with pytest.raises(ValueError, match="not finite"):
        compute_recovery([10.0] * K_PRE, post)


def test_threshold_predicate_requires_positive_degradation() -> None:
    with pytest.raises(ValueError, match="degradation"):
        recovery_threshold_met(1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="degradation"):
        recovery_threshold_met(1.0, 0.0, -1.0)
