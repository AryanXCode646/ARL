"""Evaluation benchmarks, metrics, and scenario interfaces for AdaptiveRL."""

from adaptive_rl.evaluation.evaluator import BaseEvaluator, Evaluator
from adaptive_rl.evaluation.generalization import (
    GeneralizationDistribution,
    GeneralizationEvaluator,
    GeneralizationReport,
)
from adaptive_rl.evaluation.metrics import (
    EpisodeMetrics,
    EvaluationMetrics,
    StandardizedExperimentMetrics,
    extract_episode_metrics,
)
from adaptive_rl.evaluation.scenarios import EvaluationScenario
from adaptive_rl.evaluation.seeding import (
    derive_evaluation_seed,
    derive_planner_seed,
    generate_evaluation_seeds,
)

__all__ = [
    "BaseEvaluator",
    "EpisodeMetrics",
    "Evaluator",
    "EvaluationMetrics",
    "EvaluationScenario",
    "GeneralizationDistribution",
    "GeneralizationEvaluator",
    "GeneralizationReport",
    "StandardizedExperimentMetrics",
    "derive_evaluation_seed",
    "derive_planner_seed",
    "extract_episode_metrics",
    "generate_evaluation_seeds",
]
