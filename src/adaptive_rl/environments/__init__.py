"""Environment interfaces, metadata, and registration for AdaptiveRL."""

from adaptive_rl.environments.base import AdaptiveRLEnv
from adaptive_rl.environments.gridworld.grid import GridWorldEnv
from adaptive_rl.environments.metadata import EnvironmentMetadata
from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv
from adaptive_rl.environments.registry import (
    EnvironmentRegistry,
    RegistryError,
    create_environment,
    get,
    get_metadata,
    list_all_metadata,
    list_environments,
    make_env,
    register,
    registry,
)
from adaptive_rl.environments.testing import DummyTestEnv
from adaptive_rl.environments.traffic.intersection import TrafficSignalEnv


def register_default_environments() -> None:
    """Register built-in environments into the global registry if not already present."""
    if "gridworld" not in list_environments():
        register(
            "gridworld",
            lambda **kwargs: GridWorldEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="gridworld",
                description="Procedurally generated 2D grid navigation with obstacle avoidance.",
                observation_type="box",
                action_type="discrete",
                version="0.1.0",
                max_episode_steps=100,
                reward_range=(-100.0, 100.0),
                tags=["discrete", "procedural", "navigation", "grid"],
            ),
        )

    if "navigation" not in list_environments():
        register(
            "navigation",
            lambda **kwargs: ContinuousNavigation2DEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="navigation",
                description="Continuous 2D navigation environment with multi-directional LiDAR rangefinders.",
                observation_type="box",
                action_type="continuous",
                version="0.1.0",
                max_episode_steps=200,
                reward_range=(-100.0, 100.0),
                tags=["continuous", "navigation", "lidar", "obstacle_avoidance"],
            ),
        )

    if "navigation_2d" not in list_environments():
        register(
            "navigation_2d",
            lambda **kwargs: ContinuousNavigation2DEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="navigation_2d",
                description="Continuous 2D navigation environment with multi-directional LiDAR rangefinders.",
                observation_type="box",
                action_type="continuous",
                version="0.1.0",
                max_episode_steps=200,
                reward_range=(-100.0, 100.0),
                tags=["continuous", "navigation", "lidar", "obstacle_avoidance"],
            ),
        )

    if "traffic" not in list_environments():
        register(
            "traffic",
            lambda **kwargs: TrafficSignalEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="traffic",
                description="4-way signalized intersection queue length and delay optimization.",
                observation_type="box",
                action_type="discrete",
                version="0.1.0",
                max_episode_steps=100,
                reward_range=(-1000.0, 100.0),
                tags=["discrete", "traffic", "queuing", "signal_control", "optimization"],
            ),
        )

    if "traffic_signal" not in list_environments():
        register(
            "traffic_signal",
            lambda **kwargs: TrafficSignalEnv(**kwargs),
            metadata=EnvironmentMetadata(
                name="traffic_signal",
                description="4-way signalized intersection queue length and delay optimization.",
                observation_type="box",
                action_type="discrete",
                version="0.1.0",
                max_episode_steps=100,
                reward_range=(-1000.0, 100.0),
                tags=["discrete", "traffic", "queuing", "signal_control", "optimization"],
            ),
        )


# Automatically register standard environments with metadata
register_default_environments()

__all__ = [
    "AdaptiveRLEnv",
    "ContinuousNavigation2DEnv",
    "DummyTestEnv",
    "EnvironmentMetadata",
    "EnvironmentRegistry",
    "GridWorldEnv",
    "RegistryError",
    "TrafficSignalEnv",
    "create_environment",
    "get",
    "get_metadata",
    "list_all_metadata",
    "list_environments",
    "make_env",
    "register",
    "register_default_environments",
    "registry",
]
