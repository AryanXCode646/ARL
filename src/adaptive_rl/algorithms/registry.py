"""Algorithm registry and factory system for AdaptiveRL.

Provides a unified registry capable of resolving and inspecting both trainable
RL algorithms (PPO, SAC) and deterministic classical motion planners (A*, RRT*),
distinguishing between algorithm types via capability metadata and enforcing
registration integrity and implementation availability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AlgorithmKind(str, Enum):
    """Classification of algorithm types in the registry.

    Values:
        RL_POLICY: Stochastic policy gradient or actor-critic RL algorithm
            (trainable via SB3, produces a learned policy).
        PLANNER: Deterministic classical planner (A*, RRT*, etc.).
            Not trained; computes paths using world-model knowledge.
    """

    RL_POLICY = "rl_policy"
    PLANNER = "planner"


def _copy_container(val: Any) -> Any:
    """Recursively copy nested dictionaries, lists, and sets."""
    if isinstance(val, dict):
        return {k: _copy_container(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_copy_container(v) for v in val]
    if isinstance(val, set):
        return {_copy_container(v) for v in val}
    return val


@dataclass
class AlgorithmMetadata:
    """Metadata record for a registered algorithm or planner.

    Attributes:
        name: Canonical lowercase identifier used for registry lookup.
        kind: Whether this is an RL policy or deterministic planner.
        description: Human-readable description.
        action_space: Type of action space supported (e.g. 'discrete', 'continuous', 'discrete, continuous').
        trainable: True if the algorithm supports a training loop.
        class_name: Fully qualified or simple class name.
        hyperparameters: Default hyperparameter documentation.
        tags: Searchable tags.
    """

    name: str
    kind: AlgorithmKind
    description: str = ""
    action_space: str = "discrete, continuous"
    trainable: bool = True
    class_name: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def copy(self) -> AlgorithmMetadata:
        """Create a defensive copy of this metadata record."""
        return AlgorithmMetadata(
            name=self.name,
            kind=self.kind,
            description=self.description,
            action_space=self.action_space,
            trainable=self.trainable,
            class_name=self.class_name,
            hyperparameters=_copy_container(self.hyperparameters),
            tags=_copy_container(self.tags),
        )


@dataclass(frozen=True)
class RegisteredAlgorithm:
    """Internal registry entry pairing an algorithm factory with its metadata.

    Ensures that an algorithm's factory and capability metadata are stored and
    retrieved as an indivisible, coherent record.
    """

    factory: Callable[..., Any]
    metadata: AlgorithmMetadata


class AlgorithmRegistryError(Exception):
    """Exception raised for algorithm registry operation failures."""

    pass


class AlgorithmRegistry:
    """Registry maintaining available algorithms, factories, and associated metadata.

    Mirrors the design of :class:`EnvironmentRegistry` for consistency.
    Stores factory and metadata as a single authoritative :class:`RegisteredAlgorithm`
    record to prevent disconnects or inconsistent lookups.

    Example::

        from adaptive_rl.algorithms.registry import algorithm_registry

        # List all registered algorithms
        names = algorithm_registry.list_algorithms()

        # Retrieve metadata
        meta = algorithm_registry.get_metadata("ppo")

        # Instantiate an algorithm
        PPOAlgorithm = algorithm_registry.get_factory("ppo")
        algo = PPOAlgorithm(env=env)
    """

    def __init__(self) -> None:
        self._registry: Dict[str, RegisteredAlgorithm] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        metadata: Optional[AlgorithmMetadata] = None,
    ) -> None:
        """Register an algorithm factory and its metadata.

        Args:
            name: Unique lowercase identifier (e.g. 'ppo', 'sac', 'astar').
            factory: Callable returning an algorithm or planner instance.
            metadata: Optional metadata describing the algorithm.

        Raises:
            AlgorithmRegistryError: If name is empty, already registered,
                factory is not callable, or metadata name does not match registry key.
        """
        if not name or not isinstance(name, str):
            raise AlgorithmRegistryError("Algorithm name must be a non-empty string.")

        clean = name.strip().lower()
        if clean in self._registry:
            raise AlgorithmRegistryError(
                f"Algorithm '{clean}' is already registered. "
                "Use a unique name or clear the registry first."
            )
        if not callable(factory):
            raise AlgorithmRegistryError(f"Factory for algorithm '{clean}' must be callable.")

        if metadata is not None:
            if not isinstance(metadata, AlgorithmMetadata):
                raise AlgorithmRegistryError(
                    f"Metadata for algorithm '{clean}' must be an instance of AlgorithmMetadata."
                )
            if metadata.name.strip().lower() != clean:
                raise AlgorithmRegistryError(
                    f"Algorithm name mismatch: registration key '{clean}' does not match "
                    f"metadata name '{metadata.name}'."
                )
            meta_record = metadata.copy()
            meta_record.name = clean
        else:
            meta_record = self._infer_metadata(clean, factory)

        self._registry[clean] = RegisteredAlgorithm(
            factory=factory,
            metadata=meta_record,
        )

    def _infer_metadata(self, clean: str, factory: Callable[..., Any]) -> AlgorithmMetadata:
        """Safely infer metadata when unambiguous, or raise AlgorithmRegistryError."""
        if isinstance(factory, type):
            from adaptive_rl.algorithms.base import BaseAlgorithm
            from adaptive_rl.planning.base import BasePlanner, PlannerPolicy

            if issubclass(factory, (BasePlanner, PlannerPolicy)):
                return AlgorithmMetadata(
                    name=clean,
                    kind=AlgorithmKind.PLANNER,
                    description=f"Deterministic planner '{clean}'.",
                    action_space="discrete",
                    trainable=False,
                    class_name=factory.__name__,
                )
            elif issubclass(factory, BaseAlgorithm):
                return AlgorithmMetadata(
                    name=clean,
                    kind=AlgorithmKind.RL_POLICY,
                    description=f"RL policy algorithm '{clean}'.",
                    action_space="discrete, continuous",
                    trainable=True,
                    class_name=factory.__name__,
                )

        raise AlgorithmRegistryError(
            f"Cannot infer algorithm metadata for '{clean}'. "
            "Explicit AlgorithmMetadata must be provided during registration."
        )

    def get_factory(self, name: str) -> Callable[..., Any]:
        """Retrieve the factory callable for a registered algorithm.

        Args:
            name: Algorithm identifier.

        Returns:
            Callable factory for the algorithm.

        Raises:
            AlgorithmRegistryError: If name is not registered.
        """
        clean = name.strip().lower()
        if clean not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "none"
            raise AlgorithmRegistryError(
                f"Unknown algorithm '{clean}'. Available registered algorithms: {available}"
            )
        return self._registry[clean].factory

    def get_metadata(self, name: str) -> AlgorithmMetadata:
        """Retrieve metadata for a registered algorithm.

        Returns a defensive copy so external modification cannot mutate internal registry state.

        Args:
            name: Algorithm identifier.

        Returns:
            AlgorithmMetadata for the algorithm.

        Raises:
            AlgorithmRegistryError: If name is not registered.
        """
        clean = name.strip().lower()
        if clean not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "none"
            raise AlgorithmRegistryError(
                f"Unknown algorithm '{clean}'. Available registered algorithms: {available}"
            )
        return self._registry[clean].metadata.copy()

    def list_algorithms(self) -> List[str]:
        """Return sorted list of registered algorithm names."""
        return sorted(self._registry.keys())

    def list_by_kind(self, kind: AlgorithmKind | str) -> List[str]:
        """Return sorted list of registered algorithm names matching a kind.

        Operates directly on the authoritative registered records.

        Args:
            kind: AlgorithmKind to filter by.

        Returns:
            Sorted list of matching algorithm names.
        """
        if not isinstance(kind, AlgorithmKind):
            try:
                kind = AlgorithmKind(kind)
            except ValueError:
                return []
        return sorted(name for name, entry in self._registry.items() if entry.metadata.kind == kind)

    def list_all_metadata(self) -> Dict[str, AlgorithmMetadata]:
        """Return mapping of all registered algorithm names to metadata defensive copies."""
        return {name: entry.metadata.copy() for name, entry in sorted(self._registry.items())}

    def is_trainable(self, name: str) -> bool:
        """Return True if the named algorithm is a trainable RL policy.

        Args:
            name: Algorithm identifier.

        Returns:
            True if trainable, False for deterministic planners.
        """
        return self.get_metadata(name).trainable

    def is_planner(self, name: str) -> bool:
        """Return True if the named algorithm is a classical planner.

        Args:
            name: Algorithm identifier.

        Returns:
            True if planner, False otherwise.
        """
        return self.get_metadata(name).kind == AlgorithmKind.PLANNER

    def clear(self) -> None:
        """Clear all registered algorithms (primarily for test isolation)."""
        self._registry.clear()

    def reset_defaults(self) -> None:
        """Clear and re-register default built-in algorithms (PPO, SAC, and available planners).

        Primarily intended for test isolation and resetting registry state.
        """
        self.clear()
        _register_defaults(self)


def _safe_import_algorithm(
    primary_module: str,
    class_name: str,
    fallback_module: Optional[str] = None,
) -> Optional[type]:
    """Safely import an algorithm or planner class from preferred or fallback modules.

    Returns the imported class, or None if neither module is present in the environment.
    """
    import importlib
    import importlib.util

    for mod_path in (primary_module, fallback_module):
        if not mod_path:
            continue
        try:
            spec = importlib.util.find_spec(mod_path)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is not None:
            mod = importlib.import_module(mod_path)
            attr = getattr(mod, class_name, None)
            if isinstance(attr, type):
                return attr
    return None


# ---------------------------------------------------------------------------
# Global default registry instance
# ---------------------------------------------------------------------------
algorithm_registry = AlgorithmRegistry()


def _register_defaults(target_registry: Optional[AlgorithmRegistry] = None) -> None:
    """Register the built-in algorithms into the global registry with real implementations."""
    reg = target_registry if target_registry is not None else algorithm_registry
    import importlib.util

    rl_stack_installed = (
        importlib.util.find_spec("gymnasium") is not None
        and importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("stable_baselines3") is not None
    )

    if rl_stack_installed:
        from adaptive_rl.algorithms.ppo import PPOAlgorithm
        from adaptive_rl.algorithms.sac import SACAlgorithm

        ppo_factory: Callable[..., Any] = PPOAlgorithm
        sac_factory: Callable[..., Any] = SACAlgorithm
    else:

        def _uninstalled_ppo_factory(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(
                "PPO requires optional RL dependencies (gymnasium, stable-baselines3, torch). "
                "Install with: pip install 'adaptive-rl[rl]'"
            )

        def _uninstalled_sac_factory(*args: Any, **kwargs: Any) -> Any:
            raise ImportError(
                "SAC requires optional RL dependencies (gymnasium, stable-baselines3, torch). "
                "Install with: pip install 'adaptive-rl[rl]'"
            )

        ppo_factory = _uninstalled_ppo_factory
        sac_factory = _uninstalled_sac_factory

    if "ppo" not in reg.list_algorithms():
        reg.register(
            "ppo",
            ppo_factory,
            AlgorithmMetadata(
                name="ppo",
                kind=AlgorithmKind.RL_POLICY,
                description=(
                    "Proximal Policy Optimization (PPO) — on-policy actor-critic algorithm. "
                    "Supports discrete and continuous action spaces. Wraps Stable-Baselines3 PPO."
                ),
                action_space="discrete, continuous",
                trainable=True,
                class_name="PPOAlgorithm",
                hyperparameters={
                    "learning_rate": 3e-4,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                },
                tags=["on-policy", "actor-critic", "sb3", "discrete", "continuous"],
            ),
        )

    if "sac" not in reg.list_algorithms():
        reg.register(
            "sac",
            sac_factory,
            AlgorithmMetadata(
                name="sac",
                kind=AlgorithmKind.RL_POLICY,
                description=(
                    "Soft Actor-Critic (SAC) — off-policy maximum entropy actor-critic algorithm. "
                    "Supports continuous action spaces. Wraps Stable-Baselines3 SAC."
                ),
                action_space="continuous",
                trainable=True,
                class_name="SACAlgorithm",
                hyperparameters={
                    "learning_rate": 3e-4,
                    "buffer_size": 100_000,
                    "learning_starts": 100,
                    "batch_size": 256,
                    "tau": 0.005,
                    "gamma": 0.99,
                    "train_freq": 1,
                    "gradient_steps": 1,
                    "ent_coef": "auto",
                },
                tags=["off-policy", "actor-critic", "sb3", "continuous", "maximum-entropy"],
            ),
        )

    # Optional classical planners: resolved strictly via real implementations.
    # Never creates fake/stub classes. If unavailable, they are not registered.
    # Prefer adaptive_rl.planning.astar (richer implementation with heuristic/allow_diagonal
    # support) as the canonical source; fall back to adaptive_rl.planners.astar.
    astar_class = _safe_import_algorithm(
        "adaptive_rl.planning.astar",
        "AStarPlanner",
        fallback_module="adaptive_rl.planners.astar",
    )
    if astar_class is not None and "astar" not in reg.list_algorithms():
        reg.register(
            "astar",
            astar_class,
            AlgorithmMetadata(
                name="astar",
                kind=AlgorithmKind.PLANNER,
                description=(
                    "A* grid navigation planner. Deterministic shortest-path algorithm using "
                    "heuristic search. Supports discrete GridWorld environments."
                ),
                action_space="discrete",
                trainable=False,
                class_name="AStarPlanner",
                hyperparameters={"heuristic": "manhattan"},
                tags=["planner", "deterministic", "shortest-path", "gridworld", "classical"],
            ),
        )

    rrt_star_class = _safe_import_algorithm(
        "adaptive_rl.planning.rrt",
        "RRTStarPlanner",
        fallback_module="adaptive_rl.planners.rrt_star",
    )
    if rrt_star_class is not None and "rrt_star" not in reg.list_algorithms():
        reg.register(
            "rrt_star",
            rrt_star_class,
            AlgorithmMetadata(
                name="rrt_star",
                kind=AlgorithmKind.PLANNER,
                description=(
                    "RRT* continuous 2D motion planner. Sampling-based kinodynamic-free planner "
                    "with tree rewiring for optimal paths in continuous spaces. "
                    "Supports ContinuousNavigation2D environments."
                ),
                action_space="continuous",
                trainable=False,
                class_name="RRTStarPlanner",
                hyperparameters={
                    "step_size": 0.5,
                    "max_iterations": 1500,
                    "goal_bias": 0.1,
                    "search_radius": 1.5,
                },
                tags=["planner", "sampling-based", "continuous", "rrt-star", "navigation-2d"],
            ),
        )


_register_defaults()


# Public convenience API (mirrors environment registry pattern)
def register_algorithm(
    name: str,
    factory: Callable[..., Any],
    metadata: Optional[AlgorithmMetadata] = None,
) -> None:
    """Register an algorithm in the global algorithm registry."""
    algorithm_registry.register(name, factory, metadata)


def get_algorithm_factory(name: str) -> Callable[..., Any]:
    """Retrieve the factory callable for a registered algorithm from the global registry."""
    return algorithm_registry.get_factory(name)


def get_algorithm_metadata(name: str) -> AlgorithmMetadata:
    """Retrieve metadata for a registered algorithm from the global registry."""
    return algorithm_registry.get_metadata(name)


def list_algorithms() -> list[str]:
    """List names of all registered algorithms in the global registry."""
    return algorithm_registry.list_algorithms()


def list_algorithms_by_kind(kind: AlgorithmKind | str) -> list[str]:
    """List registered algorithms of a specific kind from the global registry."""
    return algorithm_registry.list_by_kind(kind)


def list_all_algorithm_metadata() -> dict[str, AlgorithmMetadata]:
    """Return a mapping of all registered algorithm names to metadata copies."""
    return algorithm_registry.list_all_metadata()


def reset_algorithm_defaults() -> None:
    """Reset the global algorithm registry to default built-in algorithms."""
    algorithm_registry.reset_defaults()


__all__ = [
    "AlgorithmKind",
    "AlgorithmMetadata",
    "AlgorithmRegistry",
    "AlgorithmRegistryError",
    "RegisteredAlgorithm",
    "algorithm_registry",
    "get_algorithm_factory",
    "get_algorithm_metadata",
    "list_algorithms",
    "list_algorithms_by_kind",
    "list_all_algorithm_metadata",
    "register_algorithm",
    "reset_algorithm_defaults",
]
