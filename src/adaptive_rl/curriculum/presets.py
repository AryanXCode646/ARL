"""Predefined standard curriculum schedules for benchmark environments."""

from __future__ import annotations

from typing import Any, Dict

from adaptive_rl.curriculum.curriculum import Curriculum
from adaptive_rl.curriculum.stage import CurriculumStage


def create_navigation_curriculum(eval_window: int = 20) -> Curriculum:
    """Create a progressive 4-stage curriculum for Continuous 2D Navigation.

    Stage 0: Clear Corridor — 0 obstacles, learns pure goal-directed translation.
    Stage 1: Sparse Clutter — 2 obstacles, learns basic detour behaviors.
    Stage 2: Standard Density — 5 obstacles, standard benchmark difficulty.
    Stage 3: Dense Hazard Field — 8 obstacles with larger radii, dense avoidance.
    """
    stages = [
        CurriculumStage(
            stage_id=0,
            name="Clear Corridor",
            environment_parameters={"num_obstacles": 0, "arena_width": 20.0, "arena_height": 20.0},
            success_threshold=0.80,
            min_episodes=10,
            description="Empty arena with direct line-of-sight navigation to goal.",
        ),
        CurriculumStage(
            stage_id=1,
            name="Sparse Clutter",
            environment_parameters={"num_obstacles": 2, "obstacle_radius": 1.0},
            success_threshold=0.70,
            min_episodes=10,
            description="Sparse obstacles requiring initial LiDAR rangefinder steering.",
        ),
        CurriculumStage(
            stage_id=2,
            name="Standard Density",
            environment_parameters={"num_obstacles": 5, "obstacle_radius": 1.0},
            success_threshold=0.60,
            min_episodes=10,
            description="Standard navigation environment with multi-obstacle avoidance.",
        ),
        CurriculumStage(
            stage_id=3,
            name="Dense Hazard Field",
            environment_parameters={"num_obstacles": 8, "obstacle_radius": 1.2},
            success_threshold=0.50,
            min_episodes=10,
            description="High obstacle density requiring tight, agile path finding.",
        ),
    ]
    return Curriculum(
        name="navigation_curriculum",
        stages=stages,
        eval_window=eval_window,
    )


def create_gridworld_curriculum(eval_window: int = 20) -> Curriculum:
    """Create a progressive 4-stage curriculum for Discrete GridWorld.

    Stage 0: Open Grid — 4x4 grid with 0 obstacles.
    Stage 1: Light Clutter — 5x5 grid with 2 obstacles.
    Stage 2: Standard Grid — 6x5 grid with 4 obstacles.
    Stage 3: Dense Labyrinth — 7x7 grid with 6 obstacles.
    """
    stages = [
        CurriculumStage(
            stage_id=0,
            name="Open Grid",
            environment_parameters={"width": 4, "height": 4, "num_obstacles": 0},
            success_threshold=0.80,
            min_episodes=10,
            description="Small grid without obstacles to establish cardinal action grounding.",
        ),
        CurriculumStage(
            stage_id=1,
            name="Light Clutter",
            environment_parameters={"width": 5, "height": 5, "num_obstacles": 2},
            success_threshold=0.70,
            min_episodes=10,
            description="Moderate grid with simple obstacle detours.",
        ),
        CurriculumStage(
            stage_id=2,
            name="Standard Grid",
            environment_parameters={"width": 6, "height": 5, "num_obstacles": 4},
            success_threshold=0.60,
            min_episodes=10,
            description="Standard procedurally generated GridWorld.",
        ),
        CurriculumStage(
            stage_id=3,
            name="Dense Labyrinth",
            environment_parameters={"width": 7, "height": 7, "num_obstacles": 6},
            success_threshold=0.50,
            min_episodes=10,
            description="Larger grid with dense obstacles requiring long-horizon routing.",
        ),
    ]
    return Curriculum(
        name="gridworld_curriculum",
        stages=stages,
        eval_window=eval_window,
    )


CURRICULUM_PRESETS: Dict[str, Any] = {
    "navigation": create_navigation_curriculum,
    "navigation_2d": create_navigation_curriculum,
    "gridworld": create_gridworld_curriculum,
}


def get_curriculum_preset(name: str, eval_window: int = 20) -> Curriculum:
    """Resolve and build curriculum from preset identifier.

    Args:
        name: Name of preset ('navigation', 'navigation_2d', 'gridworld').
        eval_window: Rolling evaluation window size.

    Returns:
        Curriculum: Configured curriculum instance.
    """
    key = name.lower().strip()
    if key not in CURRICULUM_PRESETS:
        raise ValueError(
            f"Unknown curriculum preset '{name}'. Available presets: {list(CURRICULUM_PRESETS.keys())}"
        )
    return CURRICULUM_PRESETS[key](eval_window=eval_window)  # type: ignore[no-any-return]
