"""Base rendering interfaces for environment state visualization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseRenderer(ABC):
    """Abstract interface for decoupled environment and trajectory visualization."""

    @abstractmethod
    def render(self, state: Any, mode: str = "ansi") -> Optional[Any]:
        """Render the given environment state.

        Args:
            state: Current state representation.
            mode: Rendering mode (e.g., 'ansi', 'rgb_array').
        """
        pass
