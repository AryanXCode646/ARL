"""Curriculum manager orchestrating staged progression across environment milestones."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from adaptive_rl.curriculum.stage import CurriculumStage


class Curriculum:
    """Manages progression across an ordered sequence of curriculum stages."""

    def __init__(
        self,
        name: str,
        stages: List[CurriculumStage],
        eval_window: int = 20,
    ) -> None:
        """Initialize Curriculum with ordered stages.

        Args:
            name: Curriculum identifier (e.g. 'navigation_curriculum').
            stages: Non-empty list of ordered CurriculumStage instances.
            eval_window: Rolling window size for performance evaluation.
        """
        if not stages:
            raise ValueError("Curriculum must contain at least one CurriculumStage.")

        self.name = name
        self.stages = stages
        self.eval_window = eval_window
        self.current_stage_index = 0
        self.history: List[Dict[str, Any]] = []

    @property
    def current_stage(self) -> CurriculumStage:
        """Return currently active CurriculumStage."""
        return self.stages[self.current_stage_index]

    @property
    def is_complete(self) -> bool:
        """Return whether curriculum has reached and completed final stage."""
        return self.current_stage_index >= len(self.stages) - 1

    @property
    def total_stages(self) -> int:
        """Return total number of stages in curriculum."""
        return len(self.stages)

    def get_stage_parameters(self) -> Dict[str, Any]:
        """Return active environment parameters for current stage."""
        return dict(self.current_stage.environment_parameters)

    def check_advance(
        self,
        rolling_metrics: Dict[str, Any],
        stage_episodes: int,
        stage_timesteps: int,
    ) -> bool:
        """Check if current stage graduation criteria are satisfied.

        Args:
            rolling_metrics: Performance metrics dictionary.
            stage_episodes: Episodes completed in current stage.
            stage_timesteps: Steps completed in current stage.

        Returns:
            bool: True if agent should transition to next stage.
        """
        if self.is_complete:
            return False

        return self.current_stage.can_advance(
            rolling_metrics=rolling_metrics,
            stage_episodes=stage_episodes,
            stage_timesteps=stage_timesteps,
        )

    def advance(
        self,
        timesteps: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[CurriculumStage]:
        """Advance curriculum to the next stage and record transition event.

        Args:
            timesteps: Cumulative environment steps at time of advancement.
            metrics: Snapshot of metrics when advancing.

        Returns:
            Optional[CurriculumStage]: New active stage, or None if already at final stage.
        """
        if self.is_complete:
            return None

        from_stage = self.current_stage
        self.current_stage_index += 1
        to_stage = self.current_stage

        transition = {
            "from_stage_id": from_stage.stage_id,
            "from_stage_name": from_stage.name,
            "to_stage_id": to_stage.stage_id,
            "to_stage_name": to_stage.name,
            "timesteps": timesteps,
            "metrics": dict(metrics or {}),
        }
        self.history.append(transition)
        return to_stage

    def reset(self) -> CurriculumStage:
        """Reset curriculum back to initial stage."""
        self.current_stage_index = 0
        self.history.clear()
        return self.current_stage

    def to_dict(self) -> Dict[str, Any]:
        """Serialize curriculum state and configuration to a dictionary."""
        return {
            "name": self.name,
            "eval_window": self.eval_window,
            "current_stage_index": self.current_stage_index,
            "is_complete": self.is_complete,
            "stages": [asdict(s) for s in self.stages],
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Curriculum:
        """Instantiate Curriculum from serialized dictionary."""
        stages = [
            CurriculumStage(
                stage_id=s["stage_id"],
                name=s["name"],
                environment_parameters=dict(s.get("environment_parameters", {})),
                success_threshold=s.get("success_threshold"),
                mean_reward_threshold=s.get("mean_reward_threshold"),
                max_timesteps=s.get("max_timesteps"),
                min_episodes=s.get("min_episodes", 10),
                description=s.get("description", ""),
            )
            for s in data["stages"]
        ]
        curriculum = cls(
            name=data["name"],
            stages=stages,
            eval_window=data.get("eval_window", 20),
        )
        curriculum.current_stage_index = data.get("current_stage_index", 0)
        curriculum.history = list(data.get("history", []))
        return curriculum
