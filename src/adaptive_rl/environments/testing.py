"""Testing dummy environment used strictly for registry and contract verification.

IMPORTANT NOTICE:
This is a minimal dummy environment intended ONLY for unit tests and quality
gates of the environment abstraction and registry interfaces. It is NOT a
production reinforcement learning domain.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
from gymnasium import spaces
from adaptive_rl.environments.base import AdaptiveRLEnv


class DummyTestEnv(AdaptiveRLEnv[np.ndarray, int]):
    """Minimal discrete-action, continuous-observation dummy environment.

    Used ONLY to verify:
    - Farama Gymnasium API contract
    - AdaptiveRL EnvironmentRegistry integration
    - Reset and step determinism
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, step_limit: int = 10, reward_step: float = 1.0) -> None:
        super().__init__()
        self.step_limit = step_limit
        self.reward_step = reward_step

        # Observation space: 2D vector bounded in [-1.0, 1.0]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Action space: 2 discrete actions (0 or 1)
        self.action_space = spaces.Discrete(2)

        self._current_step = 0
        self._state = np.zeros(2, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state using seeded RNG."""
        super().reset(seed=seed)
        self._current_step = 0
        # Deterministic generation if seeded
        self._state = self.np_random.uniform(low=-0.5, high=0.5, size=(2,)).astype(np.float32)
        return self._state.copy(), {"step": self._current_step}

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one environment step."""
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid for action space {self.action_space}")

        self._current_step += 1

        # Simple state update
        delta = 0.1 if action == 1 else -0.1
        self._state = np.clip(self._state + delta, -1.0, 1.0).astype(np.float32)

        terminated = bool(self._current_step >= self.step_limit)
        truncated = False
        reward = float(self.reward_step)

        info = {
            "step": self._current_step,
            "action_taken": action,
        }
        return self._state.copy(), reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Return simple text representation of state."""
        return f"DummyTestEnv(step={self._current_step}, state={self._state})"
