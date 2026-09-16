"""Environment registry interface and factory management for AdaptiveRL."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class RegistryError(Exception):
    """Exception raised for environment registry operations."""
    pass


class EnvironmentRegistry:
    """Registry maintaining available environments and their creation factories."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an environment factory by name.

        Args:
            name: Identifier for the environment.
            factory: Callable producing an environment instance.
            metadata: Optional dictionary with environment attributes.

        Raises:
            RegistryError: If the environment name is already registered.
        """
        if not name or not isinstance(name, str):
            raise RegistryError("Environment name must be a non-empty string.")
        if name in self._registry:
            raise RegistryError(f"Environment '{name}' is already registered.")
        self._registry[name] = factory
        self._metadata[name] = metadata or {}

    def get(self, name: str) -> Callable[..., Any]:
        """Retrieve the factory for a registered environment.

        Args:
            name: Identifier for the environment.

        Returns:
            Callable[..., Any]: The factory callable.

        Raises:
            RegistryError: If the environment is not registered.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "none"
            raise RegistryError(
                f"Unknown environment '{name}'. Available registered environments: {available}"
            )
        return self._registry[name]

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Retrieve metadata for a registered environment."""
        if name not in self._metadata:
            raise RegistryError(f"Unknown environment '{name}'.")
        return self._metadata[name]

    def list_environments(self) -> List[str]:
        """Return a sorted list of registered environment names."""
        return sorted(self._registry.keys())

    def clear(self) -> None:
        """Clear all registered environments (primarily used for test isolation)."""
        self._registry.clear()
        self._metadata.clear()


# Global default registry instance
registry = EnvironmentRegistry()

register = registry.register
get = registry.get
list_environments = registry.list_environments
