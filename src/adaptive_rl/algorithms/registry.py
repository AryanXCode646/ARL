"""Algorithm registry for AdaptiveRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AlgorithmMetadata:
    """Metadata record for a registered RL algorithm."""

    name: str
    description: str = ""
    action_space: str = "continuous"
    trainable: bool = True
    class_name: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class AlgorithmRegistryError(Exception):
    """Exception raised for algorithm registry operation failures."""

    pass


class AlgorithmRegistry:
    """Registry maintaining available algorithms, factories, and associated metadata."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._metadata: Dict[str, AlgorithmMetadata] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        metadata: Optional[AlgorithmMetadata] = None,
    ) -> None:
        if not name or not isinstance(name, str):
            raise AlgorithmRegistryError("Algorithm name must be a non-empty string.")

        clean = name.strip().lower()
        if clean in self._factories:
            raise AlgorithmRegistryError(f"Algorithm '{clean}' is already registered.")
        if not callable(factory):
            raise AlgorithmRegistryError(f"Factory for algorithm '{clean}' must be callable.")

        self._factories[clean] = factory
        if metadata is not None:
            self._metadata[clean] = metadata
        else:
            self._metadata[clean] = AlgorithmMetadata(
                name=clean, class_name=getattr(factory, "__name__", str(factory))
            )

    def get_factory(self, name: str) -> Callable[..., Any]:
        clean = name.strip().lower()
        if clean not in self._factories:
            available = ", ".join(sorted(self._factories.keys())) or "none"
            raise AlgorithmRegistryError(
                f"Unknown algorithm '{clean}'. Available registered algorithms: {available}"
            )
        return self._factories[clean]

    def get_metadata(self, name: str) -> AlgorithmMetadata:
        clean = name.strip().lower()
        if clean not in self._metadata:
            available = ", ".join(sorted(self._metadata.keys())) or "none"
            raise AlgorithmRegistryError(
                f"Unknown algorithm '{clean}'. Available registered algorithms: {available}"
            )
        return self._metadata[clean]

    def list_algorithms(self) -> List[str]:
        return sorted(self._factories.keys())

    def list_all_metadata(self) -> Dict[str, AlgorithmMetadata]:
        return {name: self._metadata[name] for name in sorted(self._metadata.keys())}

    def is_trainable(self, name: str) -> bool:
        return True

    def clear(self) -> None:
        self._factories.clear()
        self._metadata.clear()


algorithm_registry = AlgorithmRegistry()


def _register_defaults() -> None:
    from adaptive_rl.algorithms.ppo import PPOAlgorithm

    if "ppo" not in algorithm_registry.list_algorithms():
        algorithm_registry.register(
            "ppo",
            PPOAlgorithm,
            AlgorithmMetadata(
                name="ppo",
                description="Proximal Policy Optimization (PPO) backed by Stable-Baselines3.",
                action_space="continuous",
                trainable=True,
                class_name="PPOAlgorithm",
                hyperparameters={
                    "learning_rate": 3e-4,
                    "n_steps": 1024,
                    "batch_size": 64,
                    "gamma": 0.99,
                },
                tags=["on-policy", "actor-critic", "sb3", "continuous"],
            ),
        )


_register_defaults()


def register_algorithm(
    name: str,
    factory: Callable[..., Any],
    metadata: Optional[AlgorithmMetadata] = None,
) -> None:
    algorithm_registry.register(name, factory, metadata)


def get_algorithm_factory(name: str) -> Callable[..., Any]:
    return algorithm_registry.get_factory(name)


def get_algorithm_metadata(name: str) -> AlgorithmMetadata:
    return algorithm_registry.get_metadata(name)


def list_algorithms() -> List[str]:
    return algorithm_registry.list_algorithms()


def list_all_algorithm_metadata() -> Dict[str, AlgorithmMetadata]:
    return algorithm_registry.list_all_metadata()


__all__ = [
    "AlgorithmMetadata",
    "AlgorithmRegistry",
    "AlgorithmRegistryError",
    "algorithm_registry",
    "get_algorithm_factory",
    "get_algorithm_metadata",
    "list_algorithms",
    "list_all_algorithm_metadata",
    "register_algorithm",
]
