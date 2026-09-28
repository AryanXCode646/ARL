"""Evaluation engine and metrics for AdaptiveRL."""

from adaptive_rl.evaluation.evaluator import (
    EpisodeEvaluationRecord,
    Evaluator,
    MultiSeedEvaluationResult,
    SeedEvaluationSummary,
    compare_policies,
    evaluate_ppo_policy,
    evaluate_random_policy,
    run_obstacle_density_experiment,
)
from adaptive_rl.evaluation.metrics import EvaluationMetrics
from adaptive_rl.evaluation.statistics import MetricStatistics, student_t_critical_value

__all__ = [
    "EpisodeEvaluationRecord",
    "EvaluationMetrics",
    "Evaluator",
    "MetricStatistics",
    "MultiSeedEvaluationResult",
    "SeedEvaluationSummary",
    "compare_policies",
    "evaluate_ppo_policy",
    "evaluate_random_policy",
    "run_obstacle_density_experiment",
    "student_t_critical_value",
]
