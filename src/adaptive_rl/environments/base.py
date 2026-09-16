"""Base environment interface for AdaptiveRL environments, conforming to Gymnasium."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, TypeVar

import gymnasium as gym
from gymnasium import spaces

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class AdaptiveRLEnv(gym.Env[ObsType, ActType]):
    """Abstract base class for all environments implemented within AdaptiveRL.

    Strictly conforms to the Farama Gymnasium environment interface:
    - Defines `observation_space` and `action_space` (instances of `gymnasium.spaces.Space`).
    - Implements `reset(seed=..., options=...) -> tuple[obs, info]`.
    - Implements `step(action) -> tuple[obs, reward, terminated, truncated, info]`.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        # Concrete subclasses must assign observation_space and action_space
        self.observation_space: spaces.Space[ObsType]
        self.action_space: spaces.Space[ActType]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the environment to an initial state.

        Subclasses should call `super().reset(seed=seed)` to initialize the random number generator,
        and must return a `(initial_observation, info_dictionary)` tuple.
        """
        super().reset(seed=seed, options=options)
        return getattr(self, "_state", None), {}  # type: ignore[return-value]

    @abstractmethod
    def step(
        self,
        action: ActType,
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment.

        Args:
            action: Action compatible with `self.action_space`.

        Returns:
            Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
                (observation, reward, terminated, truncated, info)
        """
        raise NotImplementedError
