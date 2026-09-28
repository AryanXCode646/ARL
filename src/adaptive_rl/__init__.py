"""AdaptiveRL — Reinforcement Learning for Simulated 3D Drone Navigation.

AdaptiveRL trains, evaluates, and demonstrates an RL agent navigating
a simulated 3D drone through obstacles toward a target waypoint.
"""

from adaptive_rl.config import (
    AlgorithmConfig,
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
    "ConfigError",
    "DefaultOutcomePolicy",
    "EnvironmentConfig",
    "EpisodeMetrics",
    "EpisodeMetricsAccumulator",
    "EvaluationConfig",
    "ExperimentConfig",
    "OutcomePolicy",
    "TrainingConfig",
    "compute_rate",
    "extract_episode_metrics",
    "load_config",
    "save_config",
]
