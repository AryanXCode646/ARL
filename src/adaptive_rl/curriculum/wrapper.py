"""Gymnasium environment wrapper injecting dynamic curriculum stage parameters."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym

from adaptive_rl.curriculum.curriculum import Curriculum


class CurriculumEnvWrapper(gym.Wrapper):
    """Dynamic Gymnasium environment wrapper synchronizing environment state with curriculum stages."""

    def __init__(self, env: gym.Env, curriculum: Curriculum) -> None:
        """Initialize CurriculumEnvWrapper.

        Args:
            env: Base Gymnasium environment instance.
            curriculum: Active Curriculum instance controlling progression.
        """
        super().__init__(env)
        self.curriculum = curriculum
        self.apply_stage()

    def apply_stage(self) -> None:
        """Propagate current curriculum stage parameters into underlying environment."""
        stage = self.curriculum.current_stage
        target_env = self.env.unwrapped

        for param_key, param_value in stage.environment_parameters.items():
            # Update wrapper env and unwrapped environment attributes
            if hasattr(self.env, param_key):
                setattr(self.env, param_key, param_value)
            if hasattr(target_env, param_key):
                setattr(target_env, param_key, param_value)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment while applying current curriculum stage parameters."""
        self.apply_stage()
        obs, info = self.env.reset(seed=seed, options=options)

        info["curriculum_stage_id"] = self.curriculum.current_stage.stage_id
        info["curriculum_stage_name"] = self.curriculum.current_stage.name
        info["curriculum_complete"] = self.curriculum.is_complete

        return obs, info

    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute simulation step and enrich info dictionary with curriculum telemetry."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        info["curriculum_stage_id"] = self.curriculum.current_stage.stage_id
        info["curriculum_stage_name"] = self.curriculum.current_stage.name
        info["curriculum_complete"] = self.curriculum.is_complete

        return obs, float(reward), terminated, truncated, info
