"""Tests for Phase 14 — Experiment Manager and Reproducibility."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from adaptive_rl.algorithms import AlgorithmKind, AlgorithmMetadata, algorithm_registry
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.experiments.manager import (
    ExperimentManager,
    ExperimentManifest,
    ExperimentResult,
    _get_git_commit,
    _get_git_provenance,
    _get_package_version,
    _make_experiment_id,
    _make_run_id,
)

# ---------------------------------------------------------------------------
# Helper configs
# ---------------------------------------------------------------------------


def _make_minimal_config(
    algo: str = "astar", env: str = "gridworld", seed: int = 42
) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for testing."""
    return ExperimentConfig(
        name=f"{env}_{algo}_test",
        seed=seed,
        algorithm=AlgorithmConfig(name=algo, learning_rate=3e-4, gamma=0.99, batch_size=64),
        environment=EnvironmentConfig(name=env, max_steps=10),
        training=TrainingConfig(total_timesteps=1, checkpoint_freq=0, log_interval=1),
        evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
    )


# ---------------------------------------------------------------------------
# Experiment ID & Run ID generation
# ---------------------------------------------------------------------------


class TestExperimentIDGeneration:
    """Tests for deterministic experiment ID generation."""

    def test_deterministic_same_config_same_id(self) -> None:
        """Identical configurations always produce identical experiment IDs."""
        config1 = _make_minimal_config(seed=42)
        config2 = _make_minimal_config(seed=42)
        assert _make_experiment_id(config1) == _make_experiment_id(config2)

    def test_id_contains_env_and_algo_and_seed(self) -> None:
        """Experiment ID contains normalized environment, algorithm, and seed."""
        config = _make_minimal_config(algo="astar", env="gridworld", seed=99)
        exp_id = _make_experiment_id(config)
        assert "gridworld" in exp_id
        assert "astar" in exp_id
        assert "seed99" in exp_id

    def test_id_filesystem_safe(self) -> None:
        """Experiment ID contains no characters unsafe for directory names."""
        config = _make_minimal_config()
        exp_id = _make_experiment_id(config)
        invalid_chars = set('\\ / : * ? " < > |')
        assert not any(ch in exp_id for ch in invalid_chars)

    def test_different_seeds_different_ids(self) -> None:
        """Different seeds produce distinct experiment identities."""
        config1 = _make_minimal_config(seed=42)
        config2 = _make_minimal_config(seed=43)
        assert _make_experiment_id(config1) != _make_experiment_id(config2)

    def test_different_parameters_different_ids(self) -> None:
        """Modifying effective parameters changes the deterministic experiment ID."""
        config1 = _make_minimal_config(seed=42)
        config2 = _make_minimal_config(seed=42)
        config2.training.total_timesteps = 50000
        assert _make_experiment_id(config1) != _make_experiment_id(config2)


# ---------------------------------------------------------------------------
# Run uniqueness & Concurrency
# ---------------------------------------------------------------------------


class TestRunUniquenessAndConcurrency:
    """Tests verifying repeated executions cannot overwrite previous runs."""

    def test_same_experiment_executed_twice_different_runs(self) -> None:
        """Running the same experiment twice produces distinct run directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld", seed=42)

            res1 = manager.run(config=config)
            res2 = manager.run(config=config)

            # Same experiment identity, different run executions
            assert res1.experiment_id == res2.experiment_id
            assert res1.run_id != res2.run_id
            assert res1.output_dir != res2.output_dir
            assert res1.output_dir.exists()
            assert res2.output_dir.exists()
            assert (res1.output_dir / "manifest.json").exists()
            assert (res2.output_dir / "manifest.json").exists()

    def test_run_id_format(self) -> None:
        """_make_run_id produces a unique string starting with run_."""
        run_id1 = _make_run_id()
        run_id2 = _make_run_id()
        assert run_id1.startswith("run_")
        assert run_id2.startswith("run_")
        assert run_id1 != run_id2


# ---------------------------------------------------------------------------
# Configuration provenance & Overrides
# ---------------------------------------------------------------------------


class TestConfigProvenanceAndOverrides:
    """Tests verifying source_config, overrides, and effective_config tracking."""

    def test_source_and_effective_config_tracked(self) -> None:
        """Overrides modify effective_config while preserving source_config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg_file = tmp_path / "test_config.yaml"
            cfg_file.write_text(
                """name: "override_test"
seed: 42
algorithm:
  name: "astar"
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 64
environment:
  name: "gridworld"
  max_steps: 10
training:
  total_timesteps: 10
evaluation:
  eval_episodes: 2
  deterministic: true
"""
            )

            manager = ExperimentManager(base_output_dir=tmp_path / "results")
            result = manager.run_from_config(
                cfg_file,
                timesteps_override=999,
                seed_override=77,
            )

            assert result.success
            manifest = result.manifest

            # Verify source_config preserves original values
            assert manifest.source_config["training"]["total_timesteps"] == 10
            assert manifest.source_config["seed"] == 42

            # Verify overrides are explicitly recorded
            assert manifest.overrides["training.total_timesteps"] == 999
            assert manifest.overrides["seed"] == 77

            # Verify effective_config reflects overrides
            assert manifest.effective_config["training"]["total_timesteps"] == 999
            assert manifest.effective_config["seed"] == 77

            # Verify both config files exist in output
            assert (result.output_dir / "config.yaml").exists()
            assert (result.output_dir / "source_config.yaml").exists()

    def test_source_config_object_not_mutated(self) -> None:
        """Calling manager.run does not mutate the passed config in place."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", seed=42)
            original_seed = config.seed

            manager.run(config=config, overrides={"seed": 100})
            assert config.seed == original_seed


# ---------------------------------------------------------------------------
# Algorithm Registry Integration (PR #82)
# ---------------------------------------------------------------------------


class TestAlgorithmRegistryIntegration:
    """Tests proving ExperimentManager uses AlgorithmRegistry without PPO fallbacks."""

    def test_unknown_algorithm_fails_explicitly(self) -> None:
        """An unknown algorithm fails explicitly without falling back to PPO."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="definitely_not_real_algorithm_xyz")

            with patch("adaptive_rl.algorithms.ppo.PPOAlgorithm") as mock_ppo:
                result = manager.run(config=config)

                # Must NOT fall back to PPO
                assert mock_ppo.call_count == 0
                assert not result.success
                assert "definitely_not_real_algorithm_xyz" in result.error_message
                assert result.manifest.evaluation_status == "failed"

    def test_planner_does_not_create_rl_artifacts(self) -> None:
        """Planner execution does not produce empty model/ or logs/ directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld")

            result = manager.run(config=config)
            assert result.success
            # Classical planners do not produce model or log dirs
            assert not (result.output_dir / "model").exists()
            assert not (result.output_dir / "logs").exists()
            assert (result.output_dir / "metrics.json").exists()


# ---------------------------------------------------------------------------
# Git Provenance Hardening
# ---------------------------------------------------------------------------


class TestGitProvenance:
    """Tests for explicit git provenance and failure handling."""

    def test_git_provenance_live(self) -> None:
        """_get_git_provenance returns structured dictionary."""
        prov = _get_git_provenance()
        assert "commit" in prov
        assert "branch" in prov
        assert "dirty" in prov
        assert "error" in prov

    def test_git_failure_records_error_explicitly(self) -> None:
        """When git command fails, error is recorded explicitly and commit is None."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            prov = _get_git_provenance()
            assert prov["commit"] is None
            assert prov["error"] == "git binary not found"

    def test_legacy_git_commit_helper(self) -> None:
        """_get_git_commit returns a string without crashing."""
        commit = _get_git_commit()
        assert isinstance(commit, str)
        assert len(commit) > 0


# ---------------------------------------------------------------------------
# End-to-End Planner & RL Execution
# ---------------------------------------------------------------------------


class TestExperimentExecution:
    """End-to-end execution tests for planners and RL algorithms."""

    def test_run_astar_planner_experiment(self) -> None:
        """ExperimentManager runs A* planner experiment and produces valid metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld", seed=42)

            result = manager.run(config=config)

            assert isinstance(result, ExperimentResult)
            assert result.success, f"Experiment failed: {result.error_message}"
            assert result.experiment_id != ""
            assert result.run_id != ""
            assert result.output_dir.exists()
            assert (result.output_dir / "manifest.json").exists()
            assert (result.output_dir / "config.yaml").exists()
            assert (result.output_dir / "metrics.json").exists()
            assert (result.output_dir / "metrics.csv").exists()

            with open(result.output_dir / "metrics.json") as f:
                metrics = json.load(f)
            assert "success_rate" in metrics
            assert "episodes" in metrics

    def test_run_rl_ppo_experiment(self) -> None:
        """ExperimentManager runs real PPO training and evaluation end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="ppo", env="gridworld", seed=42)

            result = manager.run(config=config)

            assert result.success, f"RL experiment failed: {result.error_message}"
            # RL runs produce model/ and logs/
            assert (result.output_dir / "model").exists()
            assert (result.output_dir / "logs").exists()
            assert (result.output_dir / "metrics.json").exists()
            assert (result.output_dir / "manifest.json").exists()
            assert result.manifest.training_timesteps is not None


# ---------------------------------------------------------------------------
# Experiment Inspection & Utilities
# ---------------------------------------------------------------------------


class TestExperimentInspection:
    """Tests for querying completed experiments."""

    def test_list_experiments_empty(self) -> None:
        """list_experiments returns empty list when no experiments exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            assert manager.list_experiments() == []

    def test_list_and_get_experiments_after_run(self) -> None:
        """list_experiments and get_experiment locate runs across directory hierarchy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld", seed=42)

            res = manager.run(config=config)
            experiments = manager.list_experiments()
            assert len(experiments) == 1
            assert experiments[0]["run_id"] == res.run_id

            # Lookup by run_id
            by_run = manager.get_experiment(res.run_id)
            assert by_run is not None
            assert by_run["run_id"] == res.run_id

            # Lookup by experiment_id
            by_exp = manager.get_experiment(res.experiment_id)
            assert by_exp is not None
            assert by_exp["experiment_id"] == res.experiment_id

            # Metrics retrieval
            metrics = manager.get_metrics(res.run_id)
            assert metrics is not None
            assert "success_rate" in metrics

    def test_invalid_config_path_returns_failure(self) -> None:
        """run_from_config returns failure result for nonexistent config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            result = manager.run_from_config(Path("/nonexistent/path/config.yaml"))
            assert not result.success
            assert "Config loading failed" in result.error_message
            assert result.manifest.evaluation_status == "failed"

    def test_helper_package_version(self) -> None:
        """_get_package_version returns a version string or 'not_installed'."""
        version = _get_package_version("pydantic")
        assert isinstance(version, str)
        assert len(version) > 0
        assert version == "not_installed" or "." in version
