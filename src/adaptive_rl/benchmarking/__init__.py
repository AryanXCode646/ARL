"""Benchmarking and experimental evaluation modules for AdaptiveRL."""

from importlib import import_module
from typing import Any

from adaptive_rl.benchmarking.learning_curve import (
    LearningCurveBenchmarkResult,
    LearningCurvePoint,
    plot_learning_curve,
    run_learning_curve_benchmark,
    validate_budgets,
)

_ABLATION_EXPORTS = {
    "REWARD_ABLATION_VARIANTS",
    "ConvergenceEvaluationCallback",
    "RewardAblationVariant",
    "export_ablation_csv",
    "export_ablation_json",
    "get_ablation_variant",
    "run_reward_ablation_experiment",
}


def __getattr__(name: str) -> Any:
    if name in _ABLATION_EXPORTS:
        return getattr(import_module(".ablation", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "REWARD_ABLATION_VARIANTS",
    "ConvergenceEvaluationCallback",
    "RewardAblationVariant",
    "export_ablation_csv",
    "export_ablation_json",
    "get_ablation_variant",
    "run_reward_ablation_experiment",
    "LearningCurveBenchmarkResult",
    "LearningCurvePoint",
    "plot_learning_curve",
    "run_learning_curve_benchmark",
    "validate_budgets",
]
