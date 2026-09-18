"""AdaptiveRL — Multi-Environment Reinforcement Learning Platform.

AdaptiveRL is a modular reinforcement learning framework designed to train,
evaluate, and benchmark agents across multiple environments.
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
from adaptive_rl.metrics import EpisodeMetrics, extract_episode_metrics

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AlgorithmConfig",
    "ConfigError",
    "EnvironmentConfig",
    "EpisodeMetrics",
    "EvaluationConfig",
    "ExperimentConfig",
    "TrainingConfig",
    "extract_episode_metrics",
    "load_config",
    "save_config",
]

