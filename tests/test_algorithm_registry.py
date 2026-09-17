"""Tests for Phase 13 — Algorithm Registry, Integrity, and Implementation Resolution."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.registry import (
    AlgorithmKind,
    AlgorithmMetadata,
    AlgorithmRegistry,
    AlgorithmRegistryError,
    _register_defaults,
    _safe_import_algorithm,
    algorithm_registry,
    get_algorithm_factory,
    get_algorithm_metadata,
    list_algorithms,
    list_algorithms_by_kind,
    reset_algorithm_defaults,
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
        self.registry.register(
            "b_algo",
            _dummy_factory,
            AlgorithmMetadata(name="b_algo", kind=AlgorithmKind.RL_POLICY),
        )
        self.registry.register(
            "a_algo",
            _dummy_factory,
            AlgorithmMetadata(name="a_algo", kind=AlgorithmKind.RL_POLICY),
        )
        names = self.registry.list_algorithms()
        assert names == ["a_algo", "b_algo"]

    def test_duplicate_registration_raises(self) -> None:
        """Registering the same name twice raises AlgorithmRegistryError."""
        meta = AlgorithmMetadata(name="my_algo", kind=AlgorithmKind.RL_POLICY)
        self.registry.register("my_algo", _dummy_factory, meta)
        with pytest.raises(AlgorithmRegistryError, match="already registered"):
            self.registry.register("my_algo", _dummy_factory, meta)

    def test_empty_name_raises(self) -> None:
        """Empty or non-string algorithm name raises AlgorithmRegistryError."""
        meta = AlgorithmMetadata(name="dummy", kind=AlgorithmKind.RL_POLICY)
        with pytest.raises(AlgorithmRegistryError, match="non-empty string"):
            self.registry.register("", _dummy_factory, meta)
        with pytest.raises(AlgorithmRegistryError, match="non-empty string"):
            self.registry.register(None, _dummy_factory, meta)  # type: ignore

    def test_non_callable_factory_raises(self) -> None:
        """Non-callable factory raises AlgorithmRegistryError."""
        meta = AlgorithmMetadata(name="my_algo", kind=AlgorithmKind.RL_POLICY)
        with pytest.raises(AlgorithmRegistryError, match="callable"):
            self.registry.register("my_algo", "not_callable", meta)  # type: ignore

    def test_unknown_algorithm_raises(self) -> None:
        """Lookup of unregistered algorithm raises AlgorithmRegistryError with available list."""
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm 'nonexistent'"):
            self.registry.get_factory("nonexistent")
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm 'nonexistent'"):
            self.registry.get_metadata("nonexistent")

    def test_default_metadata_created(self) -> None:
        """When no metadata is provided, unambiguous classes are inferred, ambiguous callables raise."""
        # Unambiguous RL policy
        self.registry.register("inferred_ppo", PPOAlgorithm)
        meta_ppo = self.registry.get_metadata("inferred_ppo")
        assert meta_ppo.name == "inferred_ppo"
        assert meta_ppo.kind == AlgorithmKind.RL_POLICY
        assert meta_ppo.trainable is True
        assert meta_ppo.class_name == "PPOAlgorithm"

        # Unambiguous Planner
        self.registry.register("inferred_astar", AStarPlanner)
        meta_astar = self.registry.get_metadata("inferred_astar")
        assert meta_astar.name == "inferred_astar"
        assert meta_astar.kind == AlgorithmKind.PLANNER
        assert meta_astar.trainable is False
        assert meta_astar.class_name == "AStarPlanner"

        # Ambiguous callable raises AlgorithmRegistryError
        with pytest.raises(AlgorithmRegistryError, match="Cannot infer algorithm metadata"):
            self.registry.register("ambiguous", _dummy_factory)

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
        self.registry.register(
            "my_algo",
            _dummy_factory,
            AlgorithmMetadata(name="my_algo", kind=AlgorithmKind.RL_POLICY),
        )
        assert len(self.registry.list_algorithms()) == 1
        self.registry.clear()
        assert self.registry.list_algorithms() == []

    def test_case_normalization(self) -> None:
        """Algorithm names are normalized to lowercase on registration and lookup."""
        self.registry.register(
            "MyAlgo",
            _dummy_factory,
            AlgorithmMetadata(name="MyAlgo", kind=AlgorithmKind.RL_POLICY),
        )
        assert "myalgo" in self.registry.list_algorithms()
        factory1 = self.registry.get_factory("myalgo")
        factory2 = self.registry.get_factory("MyAlgo")
        assert factory1 is _dummy_factory
        assert factory2 is _dummy_factory
        assert self.registry.get_metadata("myalgo").name == "myalgo"


# ---------------------------------------------------------------------------
# Data integrity & Name consistency tests
# ---------------------------------------------------------------------------


class TestRegistryDataIntegrity:
    """Tests guaranteeing registry records cannot decouple or hold contradictory state."""

    def setup_method(self) -> None:
        self.registry = AlgorithmRegistry()

    def test_single_authoritative_record_structure(self) -> None:
        """Verify factory and metadata are correctly bound upon registration."""
        meta = AlgorithmMetadata(name="integrated", kind=AlgorithmKind.RL_POLICY, trainable=True)
        self.registry.register("integrated", _dummy_factory, meta)

        assert self.registry.get_factory("integrated") is _dummy_factory
        retrieved_meta = self.registry.get_metadata("integrated")
        assert retrieved_meta.name == "integrated"
        assert retrieved_meta.kind == AlgorithmKind.RL_POLICY

    def test_name_mismatch_rejected(self) -> None:
        """Registration key and metadata name MUST match, preventing contradictory pairings."""
        meta = AlgorithmMetadata(name="sac", kind=AlgorithmKind.RL_POLICY)
        with pytest.raises(AlgorithmRegistryError, match="Algorithm name mismatch.*'ppo'.*'sac'"):
            self.registry.register("ppo", _dummy_factory, meta)

    def test_name_case_insensitive_match_accepted(self) -> None:
        """Case variations between key and metadata name normalize cleanly to canonical lowercase."""
        meta = AlgorithmMetadata(name="PPO", kind=AlgorithmKind.RL_POLICY)
        self.registry.register("ppo", _dummy_factory, meta)
        assert self.registry.get_metadata("ppo").name == "ppo"

    def test_whitespace_and_case_canonicalization(self) -> None:
        """Whitespace and mixed casing are stripped and canonicalized consistently."""
        meta = AlgorithmMetadata(name="  My_Algo  ", kind=AlgorithmKind.RL_POLICY)
        self.registry.register("  My_Algo  ", _dummy_factory, meta)
        assert "my_algo" in self.registry.list_algorithms()
        assert self.registry.get_factory("  my_algo  ") is _dummy_factory
        assert self.registry.get_factory("MY_ALGO") is _dummy_factory
        assert self.registry.get_metadata("my_algo").name == "my_algo"

    def test_invalid_metadata_type_rejected(self) -> None:
        """Non-AlgorithmMetadata objects are rejected during registration."""
        with pytest.raises(
            AlgorithmRegistryError, match="must be an instance of AlgorithmMetadata"
        ):
            self.registry.register("my_algo", _dummy_factory, metadata={"name": "my_algo"})  # type: ignore


# ---------------------------------------------------------------------------
# list_by_kind() tests
# ---------------------------------------------------------------------------


class TestListByKind:
    """Tests verifying list_by_kind filters authoritative records accurately."""

    def setup_method(self) -> None:
        self.registry = AlgorithmRegistry()

    def test_list_by_kind_filters_and_sorts(self) -> None:
        """Verify list_by_kind filters by AlgorithmKind and returns sorted names."""
        self.registry.register(
            "sac", _dummy_factory, AlgorithmMetadata("sac", AlgorithmKind.RL_POLICY)
        )
        self.registry.register(
            "ppo", _dummy_factory, AlgorithmMetadata("ppo", AlgorithmKind.RL_POLICY)
        )
        self.registry.register(
            "rrt_star", _dummy_factory, AlgorithmMetadata("rrt_star", AlgorithmKind.PLANNER)
        )
        self.registry.register(
            "astar", _dummy_factory, AlgorithmMetadata("astar", AlgorithmKind.PLANNER)
        )

        rl_algos = self.registry.list_by_kind(AlgorithmKind.RL_POLICY)
        planners = self.registry.list_by_kind(AlgorithmKind.PLANNER)

        assert rl_algos == ["ppo", "sac"]
        assert planners == ["astar", "rrt_star"]
        assert all(
            self.registry.get_metadata(name).kind == AlgorithmKind.RL_POLICY for name in rl_algos
        )
        assert all(
            self.registry.get_metadata(name).kind == AlgorithmKind.PLANNER for name in planners
        )

    def test_list_by_kind_empty_or_unmatched(self) -> None:
        """Empty registry or non-matching/invalid kind returns an empty list."""
        assert self.registry.list_by_kind(AlgorithmKind.RL_POLICY) == []
        self.registry.register(
            "ppo", _dummy_factory, AlgorithmMetadata("ppo", AlgorithmKind.RL_POLICY)
        )
        assert self.registry.list_by_kind(AlgorithmKind.PLANNER) == []
        assert self.registry.list_by_kind("invalid_kind") == []  # type: ignore


# ---------------------------------------------------------------------------
# Metadata protection and immutability tests
# ---------------------------------------------------------------------------


class TestMetadataProtection:
    """Tests proving callers cannot mutate the registry internal metadata state."""

    def setup_method(self) -> None:
        self.registry = AlgorithmRegistry()

    def test_metadata_defensive_copy(self) -> None:
        """Mutating metadata before registration, after get, or after list_all does not mutate registry state."""
        meta = AlgorithmMetadata(
            name="test_algo",
            kind=AlgorithmKind.RL_POLICY,
            trainable=True,
            hyperparameters={
                "lr": 0.001,
                "network": {
                    "layers": [64, 64],
                    "activations": ["relu", "tanh"],
                },
            },
            tags=["fast", "actor-critic"],
        )
        # Register and mutate original object
        self.registry.register("test_algo", _dummy_factory, meta)
        meta.tags.append("mutated_after_register")
        meta.hyperparameters["network"]["layers"].append(128)  # type: ignore

        # Mutate retrieved copy
        retrieved = self.registry.get_metadata("test_algo")
        retrieved.trainable = False
        retrieved.tags.append("mutated_after_get")
        retrieved.hyperparameters["lr"] = 999.0
        retrieved.hyperparameters["new_key"] = "hacked"
        retrieved.hyperparameters["network"]["layers"].append(256)  # type: ignore

        # Mutate list_all copy
        all_meta = self.registry.list_all_metadata()
        all_meta["test_algo"].tags.append("mutated_after_list")
        all_meta["test_algo"].hyperparameters["network"]["activations"].append("sigmoid")  # type: ignore
        all_meta.clear()

        # Fresh retrieval must remain intact
        fresh = self.registry.get_metadata("test_algo")
        assert fresh.trainable is True
        assert fresh.tags == ["fast", "actor-critic"]
        assert fresh.hyperparameters["lr"] == 0.001
        assert fresh.hyperparameters["network"]["layers"] == [64, 64]
        assert fresh.hyperparameters["network"]["activations"] == ["relu", "tanh"]
        assert "new_key" not in fresh.hyperparameters
        assert "test_algo" in self.registry.list_all_metadata()


# ---------------------------------------------------------------------------
# Optional algorithm behavior & error handling
# ---------------------------------------------------------------------------


class TestOptionalAlgorithmHandling:
    """Tests verifying uninstalled or unavailable algorithms raise clean errors."""

    def test_unregistered_algorithm_raises_registry_error(self) -> None:
        """Querying an algorithm that is not registered raises AlgorithmRegistryError."""
        registry = AlgorithmRegistry()
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm 'uninstalled_planner'"):
            registry.get_factory("uninstalled_planner")
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm 'uninstalled_planner'"):
            registry.get_metadata("uninstalled_planner")

    def test_safe_import_algorithm_missing_module(self) -> None:
        """_safe_import_algorithm returns None when module is genuinely not found."""
        result = _safe_import_algorithm("nonexistent_package_xyz", "NonexistentClass")
        assert result is None

    def test_safe_import_algorithm_propagates_unexpected_error(self) -> None:
        """_safe_import_algorithm propagates genuine runtime/syntax errors within installed modules."""
        with patch("importlib.util.find_spec", return_value=True):
            with patch(
                "importlib.import_module", side_effect=RuntimeError("Syntax defect inside module")
            ):
                with pytest.raises(RuntimeError, match="Syntax defect inside module"):
                    _safe_import_algorithm("adaptive_rl.planning.astar", "AStarPlanner")

    def test_default_registration_omits_missing_planner(self) -> None:
        """When planner modules are unavailable, default registration registers RL but omits planners."""
        isolated_reg = AlgorithmRegistry()
        with patch("adaptive_rl.algorithms.registry._safe_import_algorithm", return_value=None):
            _register_defaults(isolated_reg)

        algos = isolated_reg.list_algorithms()
        assert "ppo" in algos
        assert "sac" in algos
        assert "astar" not in algos
        assert "rrt_star" not in algos

        # Querying the missing planner strictly raises AlgorithmRegistryError
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm 'astar'"):
            isolated_reg.get_factory("astar")
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm 'astar'"):
            isolated_reg.get_metadata("astar")

    def test_default_registration_propagates_broken_planner_import(self) -> None:
        """When an installed planner module raises an import error, default registration propagates it."""
        isolated_reg = AlgorithmRegistry()
        with patch("importlib.util.find_spec", return_value=True):
            with patch(
                "importlib.import_module", side_effect=ImportError("broken dependency in planner")
            ):
                with pytest.raises(ImportError, match="broken dependency in planner"):
                    _register_defaults(isolated_reg)

    def test_safe_import_algorithm_fallback_resolution(self) -> None:
        """_safe_import_algorithm falls back to fallback_module when primary is missing."""
        cls = _safe_import_algorithm(
            "nonexistent.primary.module",
            "AStarPlanner",
            fallback_module="adaptive_rl.planning.astar",
        )
        assert cls is AStarPlanner


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

    def test_reset_defaults(self) -> None:
        """reset_defaults() and reset_algorithm_defaults() restore built-in algorithms cleanly."""
        # Test method on global registry
        algorithm_registry.clear()
        assert algorithm_registry.list_algorithms() == []

        algorithm_registry.reset_defaults()
        algos = list_algorithms()
        assert "ppo" in algos
        assert "sac" in algos
        assert "astar" in algos
        assert "rrt_star" in algos

        # Test convenience function
        algorithm_registry.clear()
        assert algorithm_registry.list_algorithms() == []

        reset_algorithm_defaults()
        algos2 = list_algorithms()
        assert "ppo" in algos2
        assert "sac" in algos2

    def test_convenience_functions_exported(self) -> None:
        """Verify all convenience functions are defined and exported in both modules."""
        import adaptive_rl.algorithms as algo_pkg
        import adaptive_rl.algorithms.registry as reg_mod

        for name in [
            "register_algorithm",
            "get_algorithm_factory",
            "get_algorithm_metadata",
            "list_algorithms",
            "list_algorithms_by_kind",
            "list_all_algorithm_metadata",
            "reset_algorithm_defaults",
        ]:
            assert hasattr(algo_pkg, name), f"{name} not in adaptive_rl.algorithms"
            assert hasattr(reg_mod, name), f"{name} not in adaptive_rl.algorithms.registry"
            assert callable(getattr(algo_pkg, name))
            assert callable(getattr(reg_mod, name))
            assert name in algo_pkg.__all__
            assert name in reg_mod.__all__
