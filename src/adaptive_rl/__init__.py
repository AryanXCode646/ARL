"""AdaptiveRL — Reinforcement Learning for Simulated 3D Drone Navigation.

AdaptiveRL trains, evaluates, and demonstrates an RL agent navigating
a simulated 3D drone through obstacles toward a target waypoint.
"""

from adaptive_rl.benchmarking import (
    LearningCurveBenchmarkResult,
    LearningCurvePoint,
    plot_learning_curve,
    run_learning_curve_benchmark,
    validate_budgets,
)
from adaptive_rl.config import (
    AlgorithmConfig,
    BenchmarkConfig,
    ConfigError,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
    load_config,
    save_config,
)
from adaptive_rl.metrics import (
    DefaultOutcomePolicy,
    EpisodeMetrics,
    EpisodeMetricsAccumulator,
    OutcomePolicy,
    compute_rate,
    extract_episode_metrics,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AlgorithmConfig",
    "BenchmarkConfig",
    "ConfigError",
    "DefaultOutcomePolicy",
    "EnvironmentConfig",
    "EpisodeMetrics",
    "EpisodeMetricsAccumulator",
    "EvaluationConfig",
    "ExperimentConfig",
    "LearningCurveBenchmarkResult",
    "LearningCurvePoint",
    "OutcomePolicy",
    "TrainingConfig",
    "compute_rate",
    "extract_episode_metrics",
    "load_config",
    "plot_learning_curve",
    "run_learning_curve_benchmark",
    "save_config",
    "validate_budgets",
]
