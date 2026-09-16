"""Training lifecycle callbacks for AdaptiveRL."""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional


class BaseCallback(ABC):
    """Abstract base class for monitoring, logging, and evaluation callbacks during training."""

    def on_training_start(self, locals_dict: Optional[Dict[str, Any]] = None) -> None:
        """Called before the first training step."""
        pass

    def on_step(self, step: int, locals_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Called after every environment step.

        Returns:
            bool: True to continue training, False to abort early.
        """
        return True

    def on_episode_end(self, episode: int, episode_reward: float, episode_length: int) -> None:
        """Called upon completion of an environment episode."""
        pass

    def on_training_end(self) -> None:
        """Called after the final training step has executed."""
        pass
