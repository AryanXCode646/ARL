"""Environment interfaces, metadata, and registration for AdaptiveRL."""

from adaptive_rl.environments.base import AdaptiveRLEnv
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

__all__ = [
    "AdaptiveRLEnv",
    "DummyTestEnv",
    "EnvironmentMetadata",
    "EnvironmentRegistry",
    "RegistryError",
    "create_environment",
    "get",
    "get_metadata",
    "list_all_metadata",
    "list_environments",
    "make_env",
    "register",
    "registry",
]
