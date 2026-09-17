"""Training callbacks coordinating automated curriculum stage progression."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

from adaptive_rl.curriculum.curriculum import Curriculum
from adaptive_rl.curriculum.wrapper import CurriculumEnvWrapper
from adaptive_rl.training.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """Callback evaluating rolling performance metrics and advancing curriculum stages."""

    def __init__(
        self,
        curriculum: Curriculum,
        env_wrapper: Optional[CurriculumEnvWrapper] = None,
        verbose: int = 1,
    ) -> None:
        """Initialize CurriculumCallback.

        Args:
            curriculum: Curriculum instance governing progression stages.
            env_wrapper: Optional CurriculumEnvWrapper to notify upon stage change.
            verbose: Verbosity level (0: silent, 1: log stage advancement).
        """
        self.curriculum = curriculum
        self.env_wrapper = env_wrapper
        self.verbose = verbose

        self.window_size = curriculum.eval_window
        self.recent_rewards: deque[float] = deque(maxlen=self.window_size)
        self.recent_successes: deque[float] = deque(maxlen=self.window_size)

        self.stage_episodes = 0
        self.stage_timesteps = 0
        self.total_timesteps = 0

    def on_step(self, step: int, locals_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Track step execution in current stage and evaluate max_timesteps threshold."""
        self.stage_timesteps += 1
        self.total_timesteps = step

        # Check timestep-based timeout trigger
        stage = self.curriculum.current_stage
        if stage.max_timesteps is not None and self.stage_timesteps >= stage.max_timesteps:
            rolling_metrics = {
                "success_rate": (
                    float(sum(self.recent_successes) / len(self.recent_successes))
                    if self.recent_successes
                    else 0.0
                ),
                "mean_reward": (
                    float(sum(self.recent_rewards) / len(self.recent_rewards))
                    if self.recent_rewards
                    else 0.0
                ),
                "reason": "max_timesteps_reached",
            }
            new_stage = self.curriculum.advance(
                timesteps=self.total_timesteps,
                metrics=rolling_metrics,
            )
            if new_stage is not None:
                if self.verbose > 0:
                    print(
                        f"\n[Curriculum] >>> ADVANCED to Stage {new_stage.stage_id}: "
                        f"'{new_stage.name}' at timestep {self.total_timesteps} (max_timesteps reached) <<<"
                    )
                self.stage_episodes = 0
                self.stage_timesteps = 0
                self.recent_rewards.clear()
                self.recent_successes.clear()
                if self.env_wrapper is not None:
                    self.env_wrapper.apply_stage()

        return True

    def on_episode_end(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update rolling metrics and evaluate stage graduation criteria upon episode completion."""
        self.stage_episodes += 1
        self.recent_rewards.append(episode_reward)

        step_info = info or {}
        is_success = 1.0 if step_info.get("success", False) else 0.0
        self.recent_successes.append(is_success)

        mean_reward = float(sum(self.recent_rewards) / len(self.recent_rewards))
        success_rate = (
            float(sum(self.recent_successes) / len(self.recent_successes))
            if self.recent_successes
            else 0.0
        )

        rolling_metrics = {
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "window_size": len(self.recent_rewards),
        }

        # Check for stage advancement
        if self.curriculum.check_advance(
            rolling_metrics=rolling_metrics,
            stage_episodes=self.stage_episodes,
            stage_timesteps=self.stage_timesteps,
        ):
            new_stage = self.curriculum.advance(
                timesteps=self.total_timesteps,
                metrics=rolling_metrics,
            )

            if new_stage is not None:
                if self.verbose > 0:
                    print(
                        f"\n[Curriculum] >>> ADVANCED to Stage {new_stage.stage_id}: "
                        f"'{new_stage.name}' at timestep {self.total_timesteps} "
                        f"(SR: {success_rate * 100:.1f}%, Return: {mean_reward:.2f}) <<<"
                    )

                # Reset stage metrics for the new difficulty tier
                self.stage_episodes = 0
                self.stage_timesteps = 0
                self.recent_rewards.clear()
                self.recent_successes.clear()

                if self.env_wrapper is not None:
                    self.env_wrapper.apply_stage()
