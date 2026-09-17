"""Tests for Phase 13 — Algorithm Registry, Integrity, and Implementation Resolution."""

from __future__ import annotations

import sys
import types
from typing import Any, Callable

import pytest

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.registry import (
    AlgorithmKind,
    AlgorithmMetadata,
    AlgorithmRegistry,
    AlgorithmRegistryError,
    AlgorithmUnavailableError,
    RegisteredAlgorithm,
    _safe_import_algorithm,
    algorithm_registry,
    get_algorithm_factory,
    get_algorithm_metadata,
    list_algorithms,
    list_algorithms_by_kind,
    list_all_algorithm_metadata,
)
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.environments.gridworld.grid import GridWorldEnv
from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv
from adaptive_rl.planning.astar import AStarPlanner
from adaptive_rl.planning.rrt import RRTStarPlanner

# ---------------------------------------------------------------------------
# Unit tests with isolated registry instances
# ---------------------------------------------------------------------------


def _dummy_factory(**kwargs: Any) -> Any:
    return object()


class TestAlgorithmRegistryBasics:
    """Tests for core AlgorithmRegistry operations on isolated registry instances."""

    def setup_method(self) -> None:
        """Create a fresh registry for each test."""
        self.registry = AlgorithmRegistry()

    def test_register_and_lookup(self) -> None:
        """Basic registration and factory retrieval."""
        self.registry.register(
            "my_algo",
            _dummy_factory,
            AlgorithmMetadata(name="my_algo", kind=AlgorithmKind.RL_POLICY),
        )
        factory = self.registry.get_factory("my_algo")
        assert callable(factory)
        assert factory is _dummy_factory

    def test_list_algorithms(self) -> None:
        """list_algorithms returns sorted list of registered names."""
        self.registry.register("b_algo", _dummy_factory)
        self.registry.register("a_algo", _dummy_factory)
        names = self.registry.list_algorithms()
        assert names == ["a_algo", "b_algo"]

    def test_duplicate_registration_raises(self) -> None:
        """Registering the same name twice raises AlgorithmRegistryError."""
        self.registry.register("my_algo", _dummy_factory)
        with pytest.raises(AlgorithmRegistryError, match="already registered"):
            self.registry.register("my_algo", _dummy_factory)

    def test_empty_name_raises(self) -> None:
        """Empty or non-string algorithm name raises AlgorithmRegistryError."""
        with pytest.raises(AlgorithmRegistryError, match="non-empty string"):
            self.registry.register("", _dummy_factory)
        with pytest.raises(AlgorithmRegistryError, match="non-empty string"):
            self.registry.register(None, _dummy_factory)  # type: ignore

    def test_non_callable_factory_raises(self) -> None:
        """Non-callable factory raises AlgorithmRegistryError."""
        with pytest.raises(AlgorithmRegistryError, match="callable"):
            self.registry.register("my_algo", "not_callable")  # type: ignore

    def test_unknown_algorithm_raises(self) -> None:
        """Lookup of unregistered algorithm raises AlgorithmRegistryError with available list."""
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm 'nonexistent'"):
            self.registry.get_factory("nonexistent")
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm 'nonexistent'"):
            self.registry.get_metadata("nonexistent")

    def test_default_metadata_created(self) -> None:
        """When no metadata provided, sensible defaults are created."""
        self.registry.register("no_meta", _dummy_factory)
        meta = self.registry.get_metadata("no_meta")
        assert meta.name == "no_meta"
        assert meta.kind == AlgorithmKind.RL_POLICY
        assert meta.trainable is True

    def test_is_trainable(self) -> None:
        """is_trainable distinguishes RL algorithms from planners."""
        self.registry.register(
            "rl",
            _dummy_factory,
            AlgorithmMetadata(name="rl", kind=AlgorithmKind.RL_POLICY, trainable=True),
        )
        self.registry.register(
            "plan",
            _dummy_factory,
            AlgorithmMetadata(name="plan", kind=AlgorithmKind.PLANNER, trainable=False),
        )
        assert self.registry.is_trainable("rl") is True
        assert self.registry.is_trainable("plan") is False

    def test_clear(self) -> None:
        """clear() removes all registered algorithms."""
        self.registry.register("my_algo", _dummy_factory)
        assert len(self.registry) == 1
        self.registry.clear()
        assert self.registry.list_algorithms() == []
        assert len(self.registry) == 0

    def test_case_normalization(self) -> None:
        """Algorithm names are normalized to lowercase on registration and lookup."""
        self.registry.register("MyAlgo", _dummy_factory)
        assert "myalgo" in self.registry
        assert "MyAlgo" in self.registry
        factory1 = self.registry.get_factory("myalgo")
        factory2 = self.registry.get_factory("MyAlgo")
        assert factory1 is _dummy_factory
        assert factory2 is _dummy_factory


# ---------------------------------------------------------------------------
# Data integrity & Name consistency tests
# ---------------------------------------------------------------------------


class TestRegistryDataIntegrity:
    """Tests guaranteeing registry records cannot decouple or hold contradictory state."""

    def setup_method(self) -> None:
        self.registry = AlgorithmRegistry()

    def test_single_authoritative_record_structure(self) -> None:
        """Verify factory and metadata are encapsulated in a single RegisteredAlgorithm record."""
        meta = AlgorithmMetadata(name="integrated", kind=AlgorithmKind.RL_POLICY, trainable=True)
        self.registry.register("integrated", _dummy_factory, meta)

        entry = self.registry._registry["integrated"]
        assert isinstance(entry, RegisteredAlgorithm)
        assert entry.factory is _dummy_factory
        assert entry.metadata.name == "integrated"
        assert entry.metadata.kind == AlgorithmKind.RL_POLICY

    def test_name_mismatch_rejected(self) -> None:
        """Registration key and metadata name MUST match, preventing contradictory pairings."""
        meta = AlgorithmMetadata(name="sac", kind=AlgorithmKind.RL_POLICY)
        with pytest.raises(AlgorithmRegistryError, match="Algorithm name mismatch.*'ppo'.*'sac'"):
            self.registry.register("ppo", _dummy_factory, meta)

    def test_name_case_insensitive_match_accepted(self) -> None:
        """Case variations between key and metadata name normalize cleanly."""
        meta = AlgorithmMetadata(name="PPO", kind=AlgorithmKind.RL_POLICY)
        self.registry.register("ppo", _dummy_factory, meta)
        assert self.registry.get_metadata("ppo").name == "PPO"

    def test_invalid_metadata_type_rejected(self) -> None:
        """Non-AlgorithmMetadata objects are rejected during registration."""
        with pytest.raises(AlgorithmRegistryError, match="must be an instance of AlgorithmMetadata"):
            self.registry.register("my_algo", _dummy_factory, metadata={"name": "my_algo"})  # type: ignore


# ---------------------------------------------------------------------------
# list_by_kind() thorough tests
# ---------------------------------------------------------------------------


class TestListByKind:
    """Tests verifying list_by_kind filters authoritative records accurately."""

    def setup_method(self) -> None:
        self.registry = AlgorithmRegistry()

    def test_empty_registry_returns_empty_list(self) -> None:
        """An empty registry returns an empty list for any kind."""
        assert self.registry.list_by_kind(AlgorithmKind.RL_POLICY) == []
        assert self.registry.list_by_kind(AlgorithmKind.PLANNER) == []

    def test_matching_and_non_matching_kind(self) -> None:
        """Only matching kinds are returned; non-matching queries return empty list."""
        self.registry.register(
            "ppo",
            _dummy_factory,
            AlgorithmMetadata(name="ppo", kind=AlgorithmKind.RL_POLICY),
        )
        assert self.registry.list_by_kind(AlgorithmKind.RL_POLICY) == ["ppo"]
        assert self.registry.list_by_kind(AlgorithmKind.PLANNER) == []

    def test_multiple_algorithms_sorted(self) -> None:
        """Multiple algorithms of the same kind are returned sorted."""
        self.registry.register("sac", _dummy_factory, AlgorithmMetadata("sac", AlgorithmKind.RL_POLICY))
        self.registry.register("ppo", _dummy_factory, AlgorithmMetadata("ppo", AlgorithmKind.RL_POLICY))
        self.registry.register("rrt_star", _dummy_factory, AlgorithmMetadata("rrt_star", AlgorithmKind.PLANNER))
        self.registry.register("astar", _dummy_factory, AlgorithmMetadata("astar", AlgorithmKind.PLANNER))

        rl_algos = self.registry.list_by_kind(AlgorithmKind.RL_POLICY)
        planners = self.registry.list_by_kind(AlgorithmKind.PLANNER)

        assert rl_algos == ["ppo", "sac"]
        assert planners == ["astar", "rrt_star"]

    def test_metadata_consistency_with_list_by_kind(self) -> None:
        """Every algorithm returned by list_by_kind strictly matches the queried kind."""
        self.registry.register("sac", _dummy_factory, AlgorithmMetadata("sac", AlgorithmKind.RL_POLICY))
        self.registry.register("astar", _dummy_factory, AlgorithmMetadata("astar", AlgorithmKind.PLANNER))

        for name in self.registry.list_by_kind(AlgorithmKind.RL_POLICY):
            assert self.registry.get_metadata(name).kind == AlgorithmKind.RL_POLICY

        for name in self.registry.list_by_kind(AlgorithmKind.PLANNER):
            assert self.registry.get_metadata(name).kind == AlgorithmKind.PLANNER

    def test_invalid_kind_returns_empty_list(self) -> None:
        """Querying an unknown or invalid kind returns an empty list without raising."""
        assert self.registry.list_by_kind("invalid_kind") == []  # type: ignore


# ---------------------------------------------------------------------------
# Metadata protection and immutability tests
# ---------------------------------------------------------------------------


class TestMetadataProtection:
    """Tests proving callers cannot mutate the registry internal metadata state."""

    def setup_method(self) -> None:
        self.registry = AlgorithmRegistry()

    def test_metadata_defensive_copy_on_get(self) -> None:
        """Mutating returned metadata fields, tags, or hyperparameters does not pollute registry."""
        meta = AlgorithmMetadata(
            name="test_algo",
            kind=AlgorithmKind.RL_POLICY,
            trainable=True,
            hyperparameters={"lr": 0.001},
            tags=["fast", "actor-critic"],
        )
        self.registry.register("test_algo", _dummy_factory, meta)

        retrieved = self.registry.get_metadata("test_algo")
        retrieved.trainable = False
        retrieved.tags.append("mutated")
        retrieved.hyperparameters["lr"] = 999.0
        retrieved.hyperparameters["new_key"] = "hacked"

        # Subsequent retrieval must be pristine
        fresh = self.registry.get_metadata("test_algo")
        assert fresh.trainable is True
        assert fresh.tags == ["fast", "actor-critic"]
        assert fresh.hyperparameters == {"lr": 0.001}

    def test_metadata_defensive_copy_on_list_all(self) -> None:
        """Mutating dictionary or metadata objects from list_all_metadata does not pollute registry."""
        meta = AlgorithmMetadata(
            name="test_algo",
            kind=AlgorithmKind.RL_POLICY,
            tags=["original"],
        )
        self.registry.register("test_algo", _dummy_factory, meta)

        meta_map = self.registry.list_all_metadata()
        meta_map["test_algo"].tags.append("polluted")
        meta_map.clear()

        # Subsequent retrieval is unaffected
        assert len(self.registry.list_all_metadata()) == 1
        fresh = self.registry.get_metadata("test_algo")
        assert fresh.tags == ["original"]

    def test_metadata_defensive_copy_on_register(self) -> None:
        """Mutating the original metadata object after register() does not affect the registry."""
        meta = AlgorithmMetadata(name="test_algo", kind=AlgorithmKind.RL_POLICY, tags=["initial"])
        self.registry.register("test_algo", _dummy_factory, meta)

        meta.tags.append("post_register_mutation")
        assert self.registry.get_metadata("test_algo").tags == ["initial"]


# ---------------------------------------------------------------------------
# Optional algorithm behavior & error handling
# ---------------------------------------------------------------------------


class TestOptionalAlgorithmHandling:
    """Tests verifying optional planners and import mechanisms never fake availability."""

    def test_safe_import_returns_none_for_missing_module(self) -> None:
        """_safe_import_algorithm returns None when module specification is not on disk."""
        res = _safe_import_algorithm("nonexistent_package_12345.submod", "SomeClass")
        assert res is None

    def test_safe_import_resolves_fallback_module(self) -> None:
        """_safe_import_algorithm resolves class from fallback module if primary is missing."""
        cls = _safe_import_algorithm(
            "nonexistent_package_12345.submod",
            "AStarPlanner",
            fallback_module="adaptive_rl.planning.astar",
        )
        assert cls is AStarPlanner

    def test_safe_import_raises_attribute_error_if_class_missing(self) -> None:
        """_safe_import_algorithm raises AttributeError if module exists but attribute does not."""
        with pytest.raises(AttributeError, match="does not define 'NonexistentPlanner'"):
            _safe_import_algorithm("adaptive_rl.planning.astar", "NonexistentPlanner")

    def test_safe_import_does_not_swallow_genuine_import_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A module on disk that fails with a genuine internal ImportError must NOT be swallowed."""
        # Create a mock module that raises ImportError upon execution
        bad_module_name = "_test_broken_module"
        bad_module = types.ModuleType(bad_module_name)
        bad_module.__file__ = "/tmp/bad_module.py"

        # Simulating find_spec returning a valid spec
        fake_spec = types.SimpleNamespace(origin="/tmp/bad_module.py", submodule_search_locations=None)

        def mock_find_spec(name: str) -> Any:
            if name == bad_module_name:
                return fake_spec
            return None

        def mock_import_module(name: str) -> Any:
            if name == bad_module_name:
                raise ImportError("Simulated internal circular dependency or syntax defect!")
            return sys.modules[name]

        monkeypatch.setattr("importlib.util.find_spec", mock_find_spec)
        monkeypatch.setattr("importlib.import_module", mock_import_module)

        with pytest.raises(ImportError, match="Simulated internal circular dependency"):
            _safe_import_algorithm(bad_module_name, "SomeClass")

    def test_unavailable_algorithm_raises_explicit_error(self) -> None:
        """An unavailable algorithm is never resolved as a fake class."""
        registry = AlgorithmRegistry()

        # Case 1: Unregistered planner
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm 'uninstalled_planner'"):
            registry.get_factory("uninstalled_planner")

        # Case 2: Registered with explicit unavailability factory
        def missing_dep_factory(*args: Any, **kwargs: Any) -> Any:
            raise AlgorithmUnavailableError(
                "Algorithm 'custom_planner' is unavailable: dependency 'scipy' is not installed."
            )

        registry.register("custom_planner", missing_dep_factory)
        factory = registry.get_factory("custom_planner")
        with pytest.raises(AlgorithmUnavailableError, match="dependency 'scipy' is not installed"):
            factory()


# ---------------------------------------------------------------------------
# Global default registry & Real implementation integration tests
# ---------------------------------------------------------------------------


class TestGlobalAlgorithmRegistryIntegration:
    """Integration tests verifying real implementations registered in global algorithm_registry."""

    def test_default_algorithms_registered(self) -> None:
        """PPO, SAC, A*, and RRT* are registered in the global registry."""
        algos = list_algorithms()
        assert "ppo" in algos
        assert "sac" in algos
        assert "astar" in algos
        assert "rrt_star" in algos

    def test_real_ppo_algorithm_integration(self) -> None:
        """PPO factory produces the actual SB3-backed PPOAlgorithm class."""
        factory = get_algorithm_factory("ppo")
        assert factory is PPOAlgorithm
        meta = get_algorithm_metadata("ppo")
        assert meta.trainable is True
        assert meta.kind == AlgorithmKind.RL_POLICY
        assert meta.class_name == "PPOAlgorithm"

    def test_real_sac_algorithm_integration(self) -> None:
        """SAC factory produces the actual SB3-backed SACAlgorithm class."""
        factory = get_algorithm_factory("sac")
        assert factory is SACAlgorithm
        meta = get_algorithm_metadata("sac")
        assert meta.trainable is True
        assert meta.kind == AlgorithmKind.RL_POLICY
        assert meta.class_name == "SACAlgorithm"
        assert meta.action_space == "continuous"
        assert "buffer_size" in meta.hyperparameters

    def test_real_astar_planner_integration(self) -> None:
        """A* factory produces the real AStarPlanner class and can compute paths."""
        factory = get_algorithm_factory("astar")
        assert factory is AStarPlanner
        meta = get_algorithm_metadata("astar")
        assert meta.trainable is False
        assert meta.kind == AlgorithmKind.PLANNER
        assert meta.action_space == "discrete"

        # Verify real operational capability
        planner = factory(width=5, height=5)
        res = planner.plan((0, 0), (2, 2))
        assert res.success is True
        assert len(res.path) > 0

    def test_real_rrt_star_planner_integration(self) -> None:
        """RRT* factory produces the real RRTStarPlanner class."""
        factory = get_algorithm_factory("rrt_star")
        assert factory is RRTStarPlanner
        meta = get_algorithm_metadata("rrt_star")
        assert meta.trainable is False
        assert meta.kind == AlgorithmKind.PLANNER
        assert meta.action_space == "continuous"

        # Verify real operational capability
        planner = factory(bounds=[(0.0, 10.0), (0.0, 10.0)])
        assert planner is not None

    def test_ppo_action_space_contract(self) -> None:
        """PPO metadata explicitly declares discrete and continuous support, and can initialize both."""
        meta = get_algorithm_metadata("ppo")
        assert meta.action_space == "discrete, continuous"
        assert "any" not in meta.action_space
        assert "discrete" in meta.tags
        assert "continuous" in meta.tags

        # Verify PPO actually initializes on Discrete environment
        discrete_env = GridWorldEnv(width=4, height=4)
        ppo_discrete = PPOAlgorithm(env=discrete_env)
        assert ppo_discrete.model is not None
        discrete_env.close()

        # Verify PPO actually initializes on Continuous environment
        continuous_env = ContinuousNavigation2DEnv()
        ppo_continuous = PPOAlgorithm(env=continuous_env)
        assert ppo_continuous.model is not None
        continuous_env.close()

    def test_global_list_by_kind_partitions(self) -> None:
        """list_algorithms_by_kind partitions RL algorithms from classical planners."""
        rl_algos = list_algorithms_by_kind(AlgorithmKind.RL_POLICY)
        planners = list_algorithms_by_kind(AlgorithmKind.PLANNER)

        assert "ppo" in rl_algos
        assert "sac" in rl_algos
        assert "astar" not in rl_algos
        assert "rrt_star" not in rl_algos

        assert "astar" in planners
        assert "rrt_star" in planners
        assert "ppo" not in planners
        assert "sac" not in planners
