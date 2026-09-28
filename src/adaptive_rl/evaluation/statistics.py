"""Statistical summaries for independent-seed evaluation results."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class MetricStatistics:
    """Across-seed statistics for one scalar metric."""

    mean: float | None
    std: float | None
    ci95_lower: float | None
    ci95_upper: float | None
    sample_count: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the regularized beta function."""
    max_iterations = 300
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result

    raise ArithmeticError("Incomplete beta continued fraction did not converge.")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"Incomplete beta x must be in [0, 1], got {x}.")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive.")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)
    return 1.0 - tail if value > 0 else tail


def student_t_critical_value(confidence: float, degrees_of_freedom: int) -> float:
    """Return the positive two-sided Student-t critical value.

    The inverse is computed by bisection over the Student-t CDF, implemented
    using the regularized incomplete beta function; no statistical dependency
    or normal approximation is used.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}.")
    if degrees_of_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be positive.")

    target = (1.0 + confidence) / 2.0
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < target:
        upper *= 2.0

    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, degrees_of_freedom) < target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def summarize_seed_values(values: Sequence[float | None]) -> MetricStatistics:
    """Summarize independent seed-level values using sample standard deviation.

    Missing values are excluded metric-by-metric. CI bounds are unavailable
    unless at least two finite seed-level observations are present.
    """
    observed: list[float] = []
    for value in values:
        if value is None:
            continue
        if not math.isfinite(value):
            raise ValueError(f"Metric values must be finite, got {value!r}.")
        observed.append(float(value))

    count = len(observed)
    if count == 0:
        return MetricStatistics(None, None, None, None, 0)

    mean = math.fsum(observed) / count
    if count < 2:
        return MetricStatistics(mean, None, None, None, count)

    variance = math.fsum((value - mean) ** 2 for value in observed) / (count - 1)
    std = math.sqrt(variance)
    critical = student_t_critical_value(0.95, count - 1)
    margin = critical * std / math.sqrt(count)
    return MetricStatistics(mean, std, mean - margin, mean + margin, count)


__all__ = ["MetricStatistics", "student_t_critical_value", "summarize_seed_values"]
