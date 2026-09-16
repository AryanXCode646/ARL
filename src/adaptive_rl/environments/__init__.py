"""Environment interfaces and registration for AdaptiveRL."""

from adaptive_rl.environments.registry import (
    EnvironmentRegistry,
    RegistryError,
    get,
    list_environments,
    register,
    registry,
)

__all__ = [
    "EnvironmentRegistry",
    "RegistryError",
    "get",
    "list_environments",
    "register",
    "registry",
]
