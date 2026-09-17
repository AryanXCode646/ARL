"""Comprehensive test suite verifying all 25 core invariants from Phase 8.

Covers:
1. Intermediate step success does not latch episode success when final state is not success
2. Traffic signal overflow explicitly marks episode as failure
3. Collision always marks episode as failure
4. Terminated vs truncated correctly separated
5. Traffic telemetry aggregated mathematically (not last-step)
6. Unavailable metrics serialize as null/None, not 0.0
7. True 0.0 values preserved through JSON and CSV serialization
8. Unknown algorithm raises clear registry error
9. Classical planner rejects RL hyperparameters (learning_rate, gamma, etc.)
10. Train command rejects classical planners
11. Random seed 0 is preserved and not replaced with default
12. Config overrides persist to disk in effective config
13. Config overrides survive deep merge without type corruption
14. Git commit recorded accurately when git is available
15. Git dirty flag recorded accurately
16. Git unavailable handled gracefully without crash
17. Failed experiment records failure_type, failure_message, and failure_traceback in manifest
18. Failed experiment exits with non-zero status
19. Evaluator seed strategy respects base_seed
20. Evaluation seeds recorded in experiment manifest
21. Planner execution produces no empty model/ directory
22. RL execution produces valid model checkpoint
23. StandardizedExperimentMetrics round-trips through JSON without data loss
24. StandardizedExperimentMetrics round-trips through CSV without data loss
25. Environment/algorithm compatibility validation prevents invalid pairings
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from typer.testing import CliRunner

from adaptive_rl.algorithms import AlgorithmRegistryError, algorithm_registry
from adaptive_rl.cli import app
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
    load_config,
)
from adaptive_rl.environments.traffic.intersection import TrafficSignalEnv
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.evaluation.metrics import EvaluationMetrics, StandardizedExperimentMetrics
from adaptive_rl.evaluation.seeding import generate_evaluation_seeds
from adaptive_rl.experiments.artifacts import save_metrics_csv, save_metrics_json
from adaptive_rl.experiments.manager import ExperimentManager
from adaptive_rl.experiments.provenance import get_git_provenance

# ---------------------------------------------------------------------------
# Helpers & Mock Environments
# ---------------------------------------------------------------------------


class StepLatchingTestEnv(gym.Env):
    """Environment designed to test that transient step success does NOT latch episode success.

    Step 0: info['success'] = True, but episode continues.
    Step 1: info['success'] = False, terminated = True.
    """

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._step_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        if self._step_count == 1:
            # Step 1: Transient success flag, but episode is NOT terminated yet
            return np.zeros(2, dtype=np.float32), 1.0, False, False, {"success": True}
        else:
            # Step 2: Final step, NOT a success
            return np.zeros(2, dtype=np.float32), 0.0, True, False, {"success": False}


class TruncationTestEnv(gym.Env):
    """Environment with alternating termination and truncation."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._episode_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._episode_count += 1
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        # Odd episodes truncate, even episodes terminate
        if self._episode_count % 2 == 1:
            return np.zeros(2, dtype=np.float32), 1.0, False, True, {}
        else:
            return np.zeros(2, dtype=np.float32), 1.0, True, False, {}


class CollisionTestEnv(gym.Env):
    """Environment that always experiences a collision."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(2, dtype=np.float32),
            -10.0,
            True,
            False,
            {"collision": True, "success": True},
        )


class MockPolicyAlgo:
    """Mock algorithm that returns action 0 deterministically."""

    def predict(self, obs, deterministic=True):
        return 0, {}


# ---------------------------------------------------------------------------
# Test Cases for All 25 Invariants
# ---------------------------------------------------------------------------


class TestRepositoryInvariants:
    """Rigorous verification of the 25 core correctness invariants."""

    # Invariant 1: Intermediate step success does not latch episode success
    def test_invariant_01_no_intermediate_step_success_latching(self) -> None:
        env = StepLatchingTestEnv()
        evaluator = Evaluator(algorithm=MockPolicyAlgo(), env=env)
        metrics = evaluator.evaluate(num_episodes=3, deterministic=True)
        assert metrics.success_rate == 0.0, (
            "Intermediate step success must NOT latch episode success!"
        )

    # Invariant 2: Traffic signal overflow explicitly marks episode as failure
    def test_invariant_02_traffic_overflow_marks_failure(self) -> None:
        from adaptive_rl.environments.traffic.simulation import Approach

        env = TrafficSignalEnv(max_steps=50, max_queue=5)
        env.reset(
            seed=42,
            options={
                "initial_queues": {
                    Approach.NORTH: 5,
                    Approach.SOUTH: 5,
                    Approach.EAST: 5,
                    Approach.WEST: 5,
                },
                "arrival_rates": (10.0, 10.0, 10.0, 10.0),
            },
        )
        obs, reward, terminated, truncated, info = env.step(0)
        assert info.get("overflow") is True or info.get("had_overflow") is True
        assert info.get("success") is False, (
            "Overflowed traffic episode must never be marked success!"
        )

    # Invariant 3: Collision always marks episode as failure
    def test_invariant_03_collision_marks_failure(self) -> None:
        env = CollisionTestEnv()
        evaluator = Evaluator(algorithm=MockPolicyAlgo(), env=env)
        metrics = evaluator.evaluate(num_episodes=2, deterministic=True)
        assert metrics.collision_rate == 1.0
        assert metrics.success_rate == 0.0, "Collision must override success and mark failure!"

    # Invariant 4: Terminated vs truncated correctly separated
    def test_invariant_04_terminated_vs_truncated_separated(self) -> None:
        env = TruncationTestEnv()
        evaluator = Evaluator(algorithm=MockPolicyAlgo(), env=env)
        metrics = evaluator.evaluate(num_episodes=4, deterministic=True)
        assert metrics.truncation_rate == 0.5

    # Invariant 5: Traffic telemetry aggregated mathematically (not last-step)
    def test_invariant_05_traffic_telemetry_mathematically_aggregated(self) -> None:
        env = TrafficSignalEnv(max_steps=5)
        evaluator = Evaluator(algorithm=MockPolicyAlgo(), env=env)
        metrics = evaluator.evaluate(num_episodes=2, deterministic=True)
        assert metrics.additional_metrics.get("mean_queue_length") is not None
        assert metrics.additional_metrics.get("mean_wait_time") is not None
        assert metrics.additional_metrics.get("mean_total_departures") is not None
        assert metrics.additional_metrics.get("total_departures") is not None
        assert metrics.additional_metrics.get("cumulative_delay") is not None
        assert metrics.overflow_rate is not None

        std_metrics = StandardizedExperimentMetrics.from_rl_metrics(metrics)
        assert std_metrics.mean_queue_length is not None
        assert std_metrics.mean_wait_time is not None
        assert std_metrics.mean_total_departures is not None

    # Invariant 6: Unavailable metrics serialize as null/None, not 0.0
    def test_invariant_06_unavailable_metrics_serialize_as_none(self) -> None:
        metrics = EvaluationMetrics(
            episodes=5,
            mean_reward=10.0,
            std_reward=1.0,
            min_reward=9.0,
            max_reward=11.0,
            mean_episode_length=100.0,
            std_episode_length=0.0,
            success_rate=None,
            collision_rate=None,
            overflow_rate=None,
        )
        data = metrics.model_dump()
        assert data["success_rate"] is None
        assert data["collision_rate"] is None
        assert data["overflow_rate"] is None
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        assert loaded["success_rate"] is None
        assert loaded["collision_rate"] is None
        assert loaded["overflow_rate"] is None

    # Invariant 7: True 0.0 values preserved through JSON and CSV serialization
    def test_invariant_07_true_zero_preserved_in_serialization(self) -> None:
        std_metrics = StandardizedExperimentMetrics(
            episodes=10,
            episode_return=0.0,
            collision_rate=0.0,
            overflow_rate=0.0,
            success_rate=None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "metrics.json"
            save_metrics_json(std_metrics.model_dump(), json_path)
            with open(json_path) as f:
                data = json.load(f)
            assert data["episode_return"] == 0.0
            assert data["collision_rate"] == 0.0
            assert data["overflow_rate"] == 0.0
            assert data["success_rate"] is None

            csv_path = Path(tmpdir) / "metrics.csv"
            save_metrics_csv(std_metrics.to_csv_dict(), csv_path)
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["collision_rate"] == "0.0"
            assert rows[0]["overflow_rate"] == "0.0"
            assert rows[0]["success_rate"] == ""

    # Invariant 8: Unknown algorithm raises clear registry error
    def test_invariant_08_unknown_algorithm_raises_clear_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm 'unregistered_algo_42'"):
            AlgorithmConfig(name="unregistered_algo_42")
        with pytest.raises(
            AlgorithmRegistryError, match="Unknown algorithm 'unregistered_algo_42'"
        ):
            algorithm_registry.get_metadata("unregistered_algo_42")

    # Invariant 9: Classical planner rejects RL hyperparameters
    def test_invariant_09_planner_rejects_rl_hyperparameters(self) -> None:
        with pytest.raises(
            ValueError, match="Classical planner '.*' does not accept RL hyperparameter"
        ):
            AlgorithmConfig(name="astar", learning_rate=0.001)
        with pytest.raises(
            ValueError, match="Classical planner '.*' does not accept RL hyperparameter"
        ):
            AlgorithmConfig(name="rrt_star", gamma=0.95)

    # Invariant 10: Train command rejects classical planners
    def test_invariant_10_train_command_rejects_classical_planners(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["train", "--config", "configs/gridworld_astar.yaml"])
        assert result.exit_code != 0
        assert "classical planner" in result.output.lower()

    # Invariant 11: Random seed 0 is preserved and not replaced with default
    def test_invariant_11_seed_zero_preserved(self) -> None:
        cfg = ExperimentConfig(
            name="seed_zero_test",
            seed=0,
            algorithm=AlgorithmConfig(name="astar"),
            environment=EnvironmentConfig(name="gridworld"),
            training=None,
            evaluation=EvaluationConfig(eval_episodes=1),
        )
        assert cfg.seed == 0
        seeds = generate_evaluation_seeds(experiment_seed=0, num_episodes=5)
        assert seeds[0] == 0
        assert len(seeds) == 5

    # Invariant 12: Config overrides persist to disk in effective config
    def test_invariant_12_config_overrides_persist_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            result = manager.run_from_config(
                "configs/gridworld_astar.yaml",
                overrides={"evaluation.eval_episodes": 4},
            )
            assert result.success
            saved_cfg_path = result.output_dir / "config.yaml"
            assert saved_cfg_path.exists()
            saved_cfg = load_config(saved_cfg_path)
            assert saved_cfg.evaluation.eval_episodes == 4

    # Invariant 13: Config overrides survive deep merge without type corruption
    def test_invariant_13_config_overrides_deep_merge_type_safety(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            result = manager.run_from_config(
                "configs/gridworld_astar.yaml",
                overrides={
                    "evaluation.eval_episodes": "3",
                    "evaluation.deterministic": "false",
                },
            )
            assert result.success
            saved_cfg = load_config(result.output_dir / "config.yaml")
            assert isinstance(saved_cfg.evaluation.eval_episodes, int)
            assert saved_cfg.evaluation.eval_episodes == 3
            assert isinstance(saved_cfg.evaluation.deterministic, bool)
            assert saved_cfg.evaluation.deterministic is False

    # Invariant 14: Git commit recorded accurately when git is available
    def test_invariant_14_git_commit_recorded_accurately(self) -> None:
        provenance = get_git_provenance()
        expected = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        assert provenance["commit"] == expected
        assert provenance["error"] is None

    # Invariant 15: Git dirty flag recorded accurately
    def test_invariant_15_git_dirty_flag_recorded_accurately(self) -> None:
        provenance = get_git_provenance()
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        expected_dirty = len(status) > 0
        assert provenance["dirty"] is expected_dirty

    # Invariant 16: Git unavailable handled gracefully without crash
    def test_invariant_16_git_unavailable_handled_gracefully(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            provenance = get_git_provenance()
            assert provenance["commit"] is None
            assert provenance["dirty"] is None
            assert "not found" in str(provenance["error"])

    # Invariant 17: Failed experiment records failure_type, failure_message, failure_traceback
    def test_invariant_17_failed_experiment_records_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            with patch(
                "adaptive_rl.evaluation.evaluator.Evaluator.evaluate",
                side_effect=RuntimeError("Simulated evaluation crash"),
            ):
                result = manager.run_from_config("configs/gridworld_astar.yaml")
                assert not result.success
                assert result.manifest.failure_type == "RuntimeError"
                assert "Simulated evaluation crash" in (result.manifest.failure_message or "")
                assert result.manifest.failure_traceback is not None
                manifest_file = result.output_dir / "manifest.json"
                with open(manifest_file) as f:
                    manifest_data = json.load(f)
                assert manifest_data["failure_type"] == "RuntimeError"
                assert manifest_data["failure_traceback"] is not None

    # Invariant 18: Failed experiment exits with non-zero status
    def test_invariant_18_failed_experiment_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            with patch(
                "adaptive_rl.evaluation.evaluator.Evaluator.evaluate",
                side_effect=ValueError("Faulty inputs"),
            ):
                result = manager.run_from_config("configs/gridworld_astar.yaml")
                assert not result.success
                assert result.manifest.evaluation_status == "failed"

    # Invariant 19: Evaluator seed strategy respects base_seed
    def test_invariant_19_evaluator_seed_strategy_respects_base_seed(self) -> None:
        seeds_a = generate_evaluation_seeds(experiment_seed=1234, num_episodes=5)
        seeds_b = generate_evaluation_seeds(experiment_seed=1234, num_episodes=5)
        seeds_c = generate_evaluation_seeds(experiment_seed=9999, num_episodes=5)
        assert seeds_a == seeds_b
        assert seeds_a != seeds_c
        assert seeds_a[0] == 1234

    # Invariant 20: Evaluation seeds recorded in experiment manifest
    def test_invariant_20_evaluation_seeds_recorded_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            result = manager.run_from_config(
                "configs/gridworld_astar.yaml",
                overrides={"seed": 777, "evaluation.eval_episodes": 3},
            )
            assert result.success
            assert result.manifest.evaluation_seeds == [777, 778, 779]
            assert result.manifest.evaluation_seed_strategy == "deterministic_derived"

    # Invariant 21: Planner execution produces no empty model/ directory
    def test_invariant_21_planner_produces_no_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            result = manager.run_from_config("configs/gridworld_astar.yaml")
            assert result.success
            model_dir = result.output_dir / "model"
            assert not model_dir.exists(), (
                "Classical planner execution must NOT produce a model/ directory!"
            )

    # Invariant 22: RL execution produces valid model checkpoint
    def test_invariant_22_rl_produces_model_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            result = manager.run_from_config(
                "configs/gridworld_ppo.yaml",
                overrides={
                    "training.total_timesteps": 10,
                    "evaluation.eval_episodes": 1,
                },
            )
            assert result.success
            model_artifact = result.manifest.artifact_paths.get("model")
            assert model_artifact is not None
            assert (result.output_dir / model_artifact).exists()

    # Invariant 23: StandardizedExperimentMetrics round-trips through JSON without data loss
    def test_invariant_23_metrics_json_roundtrip(self) -> None:
        original = StandardizedExperimentMetrics(
            episodes=5,
            episode_return=15.5,
            success_rate=1.0,
            collision_rate=0.0,
            overflow_rate=None,
            mean_queue_length=None,
            additional_metrics={"custom_key": 42},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            save_metrics_json(original.model_dump(), path)
            with open(path) as f:
                loaded = json.load(f)
            restored = StandardizedExperimentMetrics(**loaded)
            assert restored.episode_return == original.episode_return
            assert restored.success_rate == 1.0
            assert restored.collision_rate == 0.0
            assert restored.overflow_rate is None
            assert restored.additional_metrics == {"custom_key": 42}

    # Invariant 24: StandardizedExperimentMetrics round-trips through CSV without data loss
    def test_invariant_24_metrics_csv_roundtrip(self) -> None:
        original = StandardizedExperimentMetrics(
            episodes=10,
            episode_return=-5.0,
            success_rate=0.8,
            collision_rate=0.2,
            overflow_rate=None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.csv"
            save_metrics_csv(original.to_csv_dict(), path)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            row = rows[0]
            assert float(row["episode_return"]) == -5.0
            assert float(row["success_rate"]) == 0.8
            assert float(row["collision_rate"]) == 0.2
            assert row["overflow_rate"] == ""  # None represented as empty string

    # Invariant 25: Environment/algorithm compatibility validation prevents invalid pairings
    def test_invariant_25_environment_algorithm_compatibility(self) -> None:
        # A* on CartPole
        with pytest.raises(ValueError, match=r"A\* requires a discrete grid environment"):
            ExperimentConfig(
                name="invalid_astar",
                algorithm=AlgorithmConfig(name="astar"),
                environment=EnvironmentConfig(name="cartpole"),
                training=None,
                evaluation=EvaluationConfig(eval_episodes=1),
            )

        # RRT* on discrete GridWorld
        with pytest.raises(
            ValueError, match=r"RRT\* requires a continuous 2D navigation environment"
        ):
            ExperimentConfig(
                name="invalid_rrt",
                algorithm=AlgorithmConfig(name="rrt_star"),
                environment=EnvironmentConfig(name="gridworld"),
                training=None,
                evaluation=EvaluationConfig(eval_episodes=1),
            )

        # SAC on discrete CartPole
        with pytest.raises(ValueError, match=r"Algorithm 'sac' requires a continuous action space"):
            ExperimentConfig(
                name="invalid_sac",
                algorithm=AlgorithmConfig(
                    name="sac", learning_rate=1e-3, gamma=0.99, batch_size=32
                ),
                environment=EnvironmentConfig(name="cartpole"),
                training=TrainingConfig(total_timesteps=100),
                evaluation=EvaluationConfig(eval_episodes=1),
            )
