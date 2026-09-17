"""Curriculum learning module for AdaptiveRL."""

from adaptive_rl.curriculum.callbacks import CurriculumCallback
from adaptive_rl.curriculum.curriculum import Curriculum
from adaptive_rl.curriculum.presets import (
    CURRICULUM_PRESETS,
    create_gridworld_curriculum,
    create_navigation_curriculum,
    get_curriculum_preset,
)
from adaptive_rl.curriculum.stage import CurriculumStage
from adaptive_rl.curriculum.trainer import CurriculumTrainer
from adaptive_rl.curriculum.wrapper import CurriculumEnvWrapper

__all__ = [
    "CURRICULUM_PRESETS",
    "Curriculum",
    "CurriculumCallback",
    "CurriculumEnvWrapper",
    "CurriculumStage",
    "CurriculumTrainer",
    "create_gridworld_curriculum",
    "create_navigation_curriculum",
    "get_curriculum_preset",
]
