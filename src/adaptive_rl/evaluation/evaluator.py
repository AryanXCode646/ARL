"""Evaluation engine interfaces for AdaptiveRL."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaptive_rl.evaluation.metrics import EvaluationMetrics


class BaseEvaluator(ABC):
    """Abstract interface for agent/environment evaluation routines.

    Coordinates multi-episode evaluation runs, standard metric collection,
    and structured output generation. Concrete evaluation engine implemented in Phase 5.
    """

    @abstractmethod
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> EvaluationMetrics:
        """Run evaluation benchmark over the specified number of episodes."""
        pass
