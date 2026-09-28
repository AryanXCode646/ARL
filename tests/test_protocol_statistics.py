"""Tests for the preregistered statistical analysis plan (Issue #98 / PR #168).

Validation strategy (no scipy in this repository):

* Student-t CDF/quantiles are checked against published t-table constants
  (t_{0.975,9}, t_{0.95,9}, t_{0.975,4}, t_{0.975,1}) at 1e-6 tolerance and
  against stdlib ``statistics`` for mean/SD arithmetic.
* Exact tests are checked against hand-enumerated binomial and sign-flip
  null distributions.
* Holm and the intersection-union family rule are checked against their
  worked examples in the protocol document.
"""

from __future__ import annotations

import math
import statistics as stdlib_statistics
from statistics import NormalDist

import pytest

from adaptive_rl.protocol import (
    ALPHA,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    IMPUTATION_DIRECTIONS,
    MIN_VALID_N,
    PRIMARY_CELLS,
    bootstrap_percentile_ci,
    cohen_dz,
    decide_family,
    exact_sign_test,
    exact_wilcoxon_signed_rank,
    holm_adjust,
    impute_differences,
    paired_differences,
    paired_t_interval,
    paired_t_test,
    student_t_cdf,
    student_t_ppf,
)

# ---------------------------------------------------------------------------
# Student-t distribution
# ---------------------------------------------------------------------------


def test_cdf_matches_published_t_table_quantiles() -> None:
    assert student_t_cdf(2.262157, 9) == pytest.approx(0.975, abs=1e-6)
    assert student_t_cdf(1.833113, 9) == pytest.approx(0.95, abs=1e-6)
    assert student_t_cdf(2.776445, 4) == pytest.approx(0.975, abs=1e-6)
    assert student_t_cdf(12.706205, 1) == pytest.approx(0.975, abs=1e-6)


def test_ppf_inverts_published_t_table_quantiles() -> None:
    assert student_t_ppf(0.975, 9) == pytest.approx(2.262157, abs=1e-6)
    assert student_t_ppf(0.95, 9) == pytest.approx(1.833113, abs=1e-6)
    assert student_t_ppf(0.975, 4) == pytest.approx(2.776445, abs=1e-6)
    assert student_t_ppf(0.975, 1) == pytest.approx(12.706205, abs=1e-5)


def test_cdf_symmetry_bounds_and_monotonicity() -> None:
    for df in (1, 4, 9, 30):
        for t in (-3.5, -1.0, 0.0, 0.5, 2.25):
            value = student_t_cdf(t, df)
            assert 0.0 <= value <= 1.0
            assert student_t_cdf(-t, df) == pytest.approx(1.0 - value, abs=1e-12)
        assert student_t_cdf(-math.inf, df) == 0.0
        assert student_t_cdf(math.inf, df) == 1.0
        previous = -1.0
        for t in [x / 4.0 for x in range(-20, 21)]:
            value = student_t_cdf(t, df)
            assert value >= previous
            previous = value


def test_cdf_approaches_normal_limit() -> None:
    normal_cdf = NormalDist().cdf(1.96)
    assert student_t_cdf(1.96, 100000) == pytest.approx(normal_cdf, abs=1e-4)
    assert student_t_cdf(1.96, 1000) == pytest.approx(normal_cdf, abs=1e-3)


def test_distribution_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        student_t_cdf(1.0, 0)
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        student_t_ppf(0.975, 0)
    with pytest.raises(ValueError, match="p must be in"):
        student_t_ppf(0.0, 9)
    with pytest.raises(ValueError, match="p must be in"):
        student_t_ppf(1.0, 9)
    with pytest.raises(ValueError, match="NaN"):
        student_t_cdf(float("nan"), 9)


# ---------------------------------------------------------------------------
# Primary test
# ---------------------------------------------------------------------------

DEMO_DIFFERENCES = [-1.5, -2.0, -0.5, -3.0, -1.0, -2.5, -0.5, -1.5]


def test_paired_t_test_matches_stdlib_arithmetic() -> None:
    result = paired_t_test(DEMO_DIFFERENCES)
    expected_mean = stdlib_statistics.mean(DEMO_DIFFERENCES)
    expected_sd = stdlib_statistics.stdev(DEMO_DIFFERENCES)
    expected_se = expected_sd / math.sqrt(len(DEMO_DIFFERENCES))
    expected_t = expected_mean / expected_se
    assert result.n == 8
    assert result.degrees_of_freedom == 7  # n - 1, never hard-coded to 9
    assert result.mean == pytest.approx(expected_mean, abs=1e-12)
    assert result.std_dev == pytest.approx(expected_sd, abs=1e-12)
    assert result.standard_error == pytest.approx(expected_se, abs=1e-12)
    assert result.t_statistic == pytest.approx(expected_t, abs=1e-10)
    assert result.p_value == pytest.approx(student_t_cdf(expected_t, 7), abs=1e-12)
    # All differences negative and significant for H1: mean < 0.
    assert result.t_statistic < 0.0
    assert result.p_value < ALPHA


def test_paired_t_test_rejects_below_minimum_valid_n() -> None:
    with pytest.raises(ValueError, match="not evaluable"):
        paired_t_test([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="not evaluable"):
        paired_t_test([-1.0] * (MIN_VALID_N - 1))
    # Exactly MIN_VALID_N is evaluable.
    assert paired_t_test([-1.0] * MIN_VALID_N).n == MIN_VALID_N


def test_paired_t_test_degenerate_variance_rules() -> None:
    below = paired_t_test([-1.0] * MIN_VALID_N)
    assert below.t_statistic == -math.inf
    assert below.p_value == 0.0
    assert below.standard_error == 0.0

    above = paired_t_test([2.0] * MIN_VALID_N)
    assert above.t_statistic == math.inf
    assert above.p_value == 1.0

    zero = paired_t_test([0.0] * MIN_VALID_N)
    assert zero.t_statistic == 0.0
    assert zero.p_value == 0.5


def test_paired_t_test_rejects_non_finite_values() -> None:
    values = [-1.0] * MIN_VALID_N
    values[2] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        paired_t_test(values)
    values[2] = float("inf")
    with pytest.raises(ValueError, match="not finite"):
        paired_t_test(values)


def test_paired_t_interval_uses_actual_degrees_of_freedom() -> None:
    low, high = paired_t_interval(DEMO_DIFFERENCES)
    n = len(DEMO_DIFFERENCES)
    mean = stdlib_statistics.mean(DEMO_DIFFERENCES)
    se = stdlib_statistics.stdev(DEMO_DIFFERENCES) / math.sqrt(n)
    critical = student_t_ppf(0.975, n - 1)
    assert low == pytest.approx(mean - critical * se, abs=1e-12)
    assert high == pytest.approx(mean + critical * se, abs=1e-12)
    assert low < mean < high

    # n = 9 must use df = 8, not the full-sample df = 9.
    nine = [-1.0, -2.0, -1.5, -2.5, -1.0, -2.0, -1.5, -2.5, -1.25]
    low9, high9 = paired_t_interval(nine)
    critical_8 = student_t_ppf(0.975, 8)
    se_8 = stdlib_statistics.stdev(nine) / math.sqrt(9)
    mean_9 = stdlib_statistics.mean(nine)
    assert high9 == pytest.approx(mean_9 + critical_8 * se_8, abs=1e-12)
    assert low9 == pytest.approx(mean_9 - critical_8 * se_8, abs=1e-12)

    with pytest.raises(ValueError, match="confidence"):
        paired_t_interval(DEMO_DIFFERENCES, confidence=1.0)
    with pytest.raises(ValueError, match="not evaluable"):
        paired_t_interval([1.0, 2.0, 3.0])


def test_cohen_dz() -> None:
    result_mean = stdlib_statistics.mean(DEMO_DIFFERENCES)
    result_sd = stdlib_statistics.stdev(DEMO_DIFFERENCES)
    assert cohen_dz(DEMO_DIFFERENCES) == pytest.approx(result_mean / result_sd)
    assert cohen_dz([0.0] * MIN_VALID_N) == 0.0
    assert cohen_dz([3.0] * MIN_VALID_N) is None


# ---------------------------------------------------------------------------
# Pairwise-complete differences and imputation
# ---------------------------------------------------------------------------


def test_paired_differences_is_pairwise_complete() -> None:
    fixed = [8.0, 8.0, 6.0, None, None, 5.0, 5.0, 5.0, 5.0, 13.0]
    adaptive = [6.0, 5.0, 6.0, 7.0, None, 5.0, 6.0, 4.0, 5.0, 10.0]
    differences = paired_differences(fixed, adaptive)
    assert len(differences) == 8  # replicates 4 (one arm missing) and 5 (both missing) dropped
    assert differences[0] == pytest.approx(-2.0)
    assert differences[1] == pytest.approx(-3.0)
    assert differences[2] == pytest.approx(0.0)


def test_paired_differences_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_differences([1.0, 2.0], [1.0])


def test_imputation_directions_are_adversarial_and_bounded() -> None:
    fixed = [8.0, 8.0, 6.0, None, None, 5.0, 5.0, 5.0, 5.0, 13.0]
    adaptive = [6.0, 5.0, 6.0, 7.0, None, 5.0, 6.0, 4.0, 5.0, 10.0]
    horizon = float(15)
    worst = impute_differences(fixed, adaptive, horizon, "worst_for_adaptive")
    best = impute_differences(fixed, adaptive, horizon, "best_for_adaptive")
    assert len(worst) == len(best) == len(fixed) == 10
    # Replicate 4 (only Fixed failed): worst f = 0 => D = 7 - 0; best f = 15 => 7 - 15.
    assert worst[3] == pytest.approx(7.0)
    assert best[3] == pytest.approx(-8.0)
    # Replicate 5 (both failed): worst D = 15 - 0 = 15; best D = 0 - 15.
    assert worst[4] == pytest.approx(15.0)
    assert best[4] == pytest.approx(-15.0)
    for worst_d, best_d in zip(worst, best):
        assert worst_d >= best_d
        assert -horizon <= best_d <= worst_d <= horizon
    assert set(IMPUTATION_DIRECTIONS) == {"worst_for_adaptive", "best_for_adaptive"}

    with pytest.raises(ValueError, match="direction"):
        impute_differences(fixed, adaptive, horizon, "upside_down")
    with pytest.raises(ValueError, match="equal length"):
        impute_differences([1.0], [1.0, 2.0], horizon, "worst_for_adaptive")
    with pytest.raises(ValueError, match="horizon"):
        impute_differences(fixed, adaptive, -1.0, "worst_for_adaptive")


# ---------------------------------------------------------------------------
# Sensitivity analyses (distinct targets)
# ---------------------------------------------------------------------------


def test_sign_test_hand_enumeration() -> None:
    # All 8 negative: p = P(K >= 8) for K ~ Bin(8, 0.5) = 1/256.
    result = exact_sign_test([-1.0] * MIN_VALID_N)
    assert result.n_nonzero == 8
    assert result.n_negative == 8
    assert result.p_value == pytest.approx(1.0 / 256.0)

    # Zeros dropped: [-1, 1] => n' = 2, k = 1 => p = P(K >= 1) = 3/4.
    zeros = [-1.0, 1.0] + [0.0] * (MIN_VALID_N - 2)
    result = exact_sign_test(zeros, min_valid_n=2)
    assert result.n_nonzero == 2
    assert result.n_negative == 1
    assert result.p_value == pytest.approx(0.75)

    # All positive (Adaptive slower): p = P(K >= 0) = 1.
    result = exact_sign_test([1.0] * MIN_VALID_N)
    assert result.p_value == 1.0

    # All zero: no evidence either way.
    result = exact_sign_test([0.0] * MIN_VALID_N)
    assert result.n_nonzero == 0
    assert result.p_value == 1.0

    with pytest.raises(ValueError, match="not evaluable"):
        exact_sign_test([1.0, -1.0, 1.0])


def test_wilcoxon_hand_enumeration() -> None:
    # All 8 negative: p = P(W+ = 0) = 1/256.
    result = exact_wilcoxon_signed_rank([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0])
    assert result.rank_sum_positive == 0.0
    assert result.p_value == pytest.approx(1.0 / 256.0)

    # d = [-1, 2, 3, -4] (n' = 4): ranks {1,2,3,4}, W+ = 2 + 3 = 5,
    # 9 of 16 sign patterns give W+ <= 5 => p = 9/16.
    small = [-1.0, 2.0, 3.0, -4.0, 0.0, 0.0, 0.0, 0.0]
    result = exact_wilcoxon_signed_rank(small, min_valid_n=4)
    assert result.n_nonzero == 4
    assert result.rank_sum_positive == 5.0
    assert result.p_value == pytest.approx(9.0 / 16.0)

    # All positive (Adaptive slower): W+ at its maximum, lower-tail p = 1.0.
    all_positive = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    result = exact_wilcoxon_signed_rank(all_positive, min_valid_n=4)
    assert result.rank_sum_positive == 10.0
    assert result.p_value == 1.0

    # Tie in magnitudes: average ranks must be used. Ranks are
    # {1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5}; W+ = 1.5 + 3.5 + 5.5 + 7.5 = 18,
    # and 137 of the 256 sign patterns give W+ <= 18 => p = 137/256.
    ties = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
    result = exact_wilcoxon_signed_rank(ties)
    assert result.rank_sum_positive == 18.0
    assert result.p_value == pytest.approx(137.0 / 256.0)


def test_bootstrap_is_deterministic_and_bounded() -> None:
    first = bootstrap_percentile_ci(DEMO_DIFFERENCES, reps=2000)
    second = bootstrap_percentile_ci(DEMO_DIFFERENCES, reps=2000)
    assert first == second  # frozen seed => reproducible
    low, high = first
    assert low <= high
    assert low >= min(DEMO_DIFFERENCES)
    assert high <= max(DEMO_DIFFERENCES)
    # A different seed must (overwhelmingly) give a different interval.
    other = bootstrap_percentile_ci(DEMO_DIFFERENCES, reps=2000, seed=BOOTSTRAP_SEED + 1)
    assert other != first
    with pytest.raises(ValueError, match="reps"):
        bootstrap_percentile_ci(DEMO_DIFFERENCES, reps=0)
    with pytest.raises(ValueError, match="not evaluable"):
        bootstrap_percentile_ci([1.0, 2.0, 3.0])


def test_bootstrap_defaults_are_frozen() -> None:
    assert BOOTSTRAP_REPS == 10000
    assert BOOTSTRAP_SEED == 168098


# ---------------------------------------------------------------------------
# Multiplicity and family decision
# ---------------------------------------------------------------------------


def test_holm_adjust_worked_example() -> None:
    raw = [0.01, 0.04, 0.03, 0.5, 0.02, 0.6]
    adjusted = holm_adjust(raw)
    assert adjusted == pytest.approx([0.06, 0.12, 0.12, 1.0, 0.10, 1.0], abs=1e-12)
    assert holm_adjust([0.01]) == [0.01]
    assert holm_adjust([0.5, 0.5, 0.5]) == pytest.approx([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="non-empty"):
        holm_adjust([])
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        holm_adjust([0.01, 1.5])


def test_decide_family_rules() -> None:
    all_significant = {cell: 0.01 for cell in PRIMARY_CELLS}
    assert decide_family(all_significant) == "SUPPORTED"

    one_not = dict(all_significant)
    one_not["drone_disturbed/sac"] = 0.2
    assert decide_family(one_not) == "NOT_SUPPORTED"

    # Strict inequality: p == alpha is not sufficient.
    boundary = dict(all_significant)
    boundary["gridworld/ppo"] = ALPHA
    assert decide_family(boundary) == "NOT_SUPPORTED"

    missing = dict(all_significant)
    missing["navigation_2d/sac"] = None
    assert decide_family(missing) == "INCONCLUSIVE"

    with pytest.raises(ValueError, match="missing"):
        decide_family({"gridworld/ppo": 0.01})
    with pytest.raises(ValueError, match="extra"):
        extra = dict(all_significant)
        extra["bogus/cell"] = 0.01
        decide_family(extra)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        bad = dict(all_significant)
        bad["traffic_signal/ppo"] = -0.1
        decide_family(bad)
    with pytest.raises(ValueError, match="alpha"):
        decide_family(all_significant, alpha=0.0)


def test_primary_cells_are_frozen() -> None:
    assert list(PRIMARY_CELLS) == [
        "gridworld/ppo",
        "traffic_signal/ppo",
        "drone_disturbed/ppo",
        "drone_disturbed/sac",
        "navigation_2d/ppo",
        "navigation_2d/sac",
    ]
