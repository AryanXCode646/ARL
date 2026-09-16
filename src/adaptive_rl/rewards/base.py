"""Base interfaces for modular reward functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseRewardFunction(ABC):
    """Abstract base class for modular reward functions in AdaptiveRL environments."""

    @abstractmethod
    def compute_reward(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute the scalar reward given transition dynamics."""
        pass
