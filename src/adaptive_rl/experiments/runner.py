"""Experiment orchestration interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from adaptive_rl.config import ExperimentConfig


class BaseExperimentRunner(ABC):
    """Abstract interface for orchestrating end-to-end training and evaluation runs."""

    @abstractmethod
    def run(self, config: ExperimentConfig) -> Any:
        """Execute the experiment defined by the configuration."""
        pass
