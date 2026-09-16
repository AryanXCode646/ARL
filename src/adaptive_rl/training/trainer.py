"""Training engine interfaces for AdaptiveRL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):
    """Abstract interface for RL training workflows in AdaptiveRL.

    Coordinates environment instantiation, algorithm optimization,
    callback scheduling, and model checkpointing.
    Concrete training engine implemented in Phase 4.
    """

    @abstractmethod
    def fit(self) -> Any:
        """Execute the training process."""
        pass
