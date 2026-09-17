"""Experiment runners and lifecycle management for AdaptiveRL."""

from __future__ import annotations

# Import lightweight metadata module eagerly (no circular deps)
from adaptive_rl.experiments.metadata import (
    EpisodeRecord,
    ExperimentMetadata,
    save_episodes_csv,
)
from adaptive_rl.experiments.runner import BaseExperimentRunner

__all__ = [
    "BaseExperimentRunner",
    "EpisodeRecord",
    "ExperimentMetadata",
    "GeneralizationExperimentRunner",
    "save_episodes_csv",
]


def __getattr__(name: str) -> object:
    """Lazy-load heavy modules to avoid circular imports at package init time."""
    if name == "GeneralizationExperimentRunner":
        from adaptive_rl.experiments.generalization_runner import (
            GeneralizationExperimentRunner,
        )
        return GeneralizationExperimentRunner
    raise AttributeError(f"module 'adaptive_rl.experiments' has no attribute {name!r}")
