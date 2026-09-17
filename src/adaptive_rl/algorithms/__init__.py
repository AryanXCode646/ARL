"""Algorithm abstraction layer for AdaptiveRL."""

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.algorithms.registry import (
    AlgorithmKind,
    AlgorithmMetadata,
    AlgorithmRegistry,
    AlgorithmRegistryError,
    algorithm_registry,
    get_algorithm_factory,
    get_algorithm_metadata,
    list_algorithms,
    list_algorithms_by_kind,
    list_all_algorithm_metadata,
    register_algorithm,
    reset_algorithm_defaults,
)

try:
    from adaptive_rl.algorithms.ppo import PPOAlgorithm
except ImportError:
    PPOAlgorithm = None  # type: ignore

try:
    from adaptive_rl.algorithms.sac import SACAlgorithm
except ImportError:
    SACAlgorithm = None  # type: ignore

__all__ = [
    "AlgorithmKind",
    "AlgorithmMetadata",
    "AlgorithmRegistry",
    "AlgorithmRegistryError",
    "BaseAlgorithm",
    "PPOAlgorithm",
    "SACAlgorithm",
    "algorithm_registry",
    "get_algorithm_factory",
    "get_algorithm_metadata",
    "list_algorithms",
    "list_algorithms_by_kind",
    "list_all_algorithm_metadata",
    "register_algorithm",
    "reset_algorithm_defaults",
]
