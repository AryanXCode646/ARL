"""Environment registry and factory system for AdaptiveRL."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym

from adaptive_rl.environments.metadata import EnvironmentMetadata

EnvironmentFactory = Callable[..., gym.Env]


class RegistryError(Exception):
    """Exception raised for environment registry operations."""

    pass


class EnvironmentRegistry:
    """Registry maintaining available environments, factories, and associated metadata."""

    def __init__(self) -> None:
        self._factories: Dict[str, EnvironmentFactory] = {}
        self._metadata: Dict[str, EnvironmentMetadata] = {}

    def register(
        self,
        name: str,
        factory: EnvironmentFactory,
        metadata: Optional[Union[EnvironmentMetadata, Dict[str, Any]]] = None,
    ) -> None:
        """Register an environment factory and its metadata.

        Args:
            name: Unique identifier for the environment (e.g. 'gridworld', 'navigation').
            factory: Callable producing an instantiated `gymnasium.Env`.
            metadata: Optional `EnvironmentMetadata` instance or dictionary.

        Raises:
            RegistryError: If name is invalid or already registered.
        """
        if not name or not isinstance(name, str):
            raise RegistryError("Environment name must be a non-empty string.")

        clean_name = name.strip()
        if clean_name in self._factories:
            raise RegistryError(f"Environment '{clean_name}' is already registered.")

        if not callable(factory):
            raise RegistryError(f"Environment factory for '{clean_name}' must be callable.")

        self._factories[clean_name] = factory

        if isinstance(metadata, EnvironmentMetadata):
            self._metadata[clean_name] = metadata
        elif isinstance(metadata, dict):
            # Ensure name matches if provided in dict
            meta_dict = dict(metadata)
            meta_dict.setdefault("name", clean_name)
            self._metadata[clean_name] = EnvironmentMetadata.model_validate(meta_dict)
        else:
            self._metadata[clean_name] = EnvironmentMetadata(name=clean_name)

    def get(self, name: str) -> EnvironmentFactory:
        """Retrieve the factory callable for a registered environment.

        Args:
            name: Environment identifier.

        Returns:
            EnvironmentFactory: Factory callable.

        Raises:
            RegistryError: If environment is not registered.
        """
        clean_name = name.strip()
        if clean_name not in self._factories:
            available = ", ".join(sorted(self._factories.keys())) or "none"
            raise RegistryError(
                f"Unknown environment '{clean_name}'. Available registered environments: {available}"
            )
        return self._factories[clean_name]

    def get_metadata(self, name: str) -> EnvironmentMetadata:
        """Retrieve metadata for a registered environment.

        Args:
            name: Environment identifier.

        Returns:
            EnvironmentMetadata: Metadata container.

        Raises:
            RegistryError: If environment is not registered.
        """
        clean_name = name.strip()
        if clean_name not in self._metadata:
            available = ", ".join(sorted(self._metadata.keys())) or "none"
            raise RegistryError(
                f"Unknown environment '{clean_name}'. Available registered environments: {available}"
            )
        return self._metadata[clean_name]

    def list_environments(self) -> List[str]:
        """Return a sorted list of registered environment names."""
        return sorted(self._factories.keys())

    def list_all_metadata(self) -> Dict[str, EnvironmentMetadata]:
        """Return a mapping of all registered environment names to their metadata."""
        return {k: self._metadata[k] for k in sorted(self._metadata.keys())}

    def create(self, name: str, **kwargs: Any) -> gym.Env:
        """Instantiate an environment by name using registered factory or Gymnasium fallback.

        Args:
            name: Environment identifier (e.g. 'gridworld', or standard Gymnasium 'CartPole-v1').
            **kwargs: Configuration arguments forwarded to the environment factory.

        Returns:
            gym.Env: Instantiated Gymnasium-compatible environment.

        Raises:
            RegistryError: If the environment cannot be found or fails to initialize.
        """
        clean_name = name.strip()

        # 1. Check AdaptiveRL internal registry first
        if clean_name in self._factories:
            try:
                env = self._factories[clean_name](**kwargs)
            except TypeError as err:
                raise RegistryError(
                    f"Failed to instantiate registered environment '{clean_name}' with provided kwargs: {err}"
                ) from err
            except Exception as err:
                raise RegistryError(
                    f"Error during environment '{clean_name}' creation: {err}"
                ) from err

            # Contract verification
            if not isinstance(env, gym.Env):
                raise RegistryError(
                    f"Factory for '{clean_name}' returned {type(env).__name__}, which is not an instance of gymnasium.Env"
                )
            if not hasattr(env, "observation_space") or not hasattr(env, "action_space"):
                raise RegistryError(
                    f"Environment '{clean_name}' is missing observation_space or action_space attribute."
                )
            return env

        # 2. Fall back to standard Gymnasium environment registry
        try:
            return gym.make(clean_name, **kwargs)
        except Exception as gym_err:
            available = ", ".join(sorted(self._factories.keys())) or "none"
            raise RegistryError(
                f"Environment '{clean_name}' is neither registered in AdaptiveRL nor resolvable by Gymnasium.\n"
                f"Available registered AdaptiveRL environments: {available}\n"
                f"Gymnasium error: {gym_err}"
            ) from gym_err

    def clear(self) -> None:
        """Clear all registered environments (primarily used for test isolation)."""
        self._factories.clear()
        self._metadata.clear()


# Global default registry instance
registry = EnvironmentRegistry()

# Public convenience API
register = registry.register
get = registry.get
get_metadata = registry.get_metadata
list_environments = registry.list_environments
list_all_metadata = registry.list_all_metadata
create_environment = registry.create
make_env = registry.create
