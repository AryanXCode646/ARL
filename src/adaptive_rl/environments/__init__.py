"""Environment interfaces, metadata, and registration for AdaptiveRL."""

from adaptive_rl.environments.base import AdaptiveRLEnv
from adaptive_rl.environments.gridworld.grid import GridWorldEnv
from adaptive_rl.environments.metadata import EnvironmentMetadata
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


# Automatically register standard GridWorld environment with metadata
register_default_environments()

__all__ = [
    "AdaptiveRLEnv",
    "DummyTestEnv",
    "EnvironmentMetadata",
    "EnvironmentRegistry",
    "GridWorldEnv",
    "RegistryError",
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
