"""Tests for Phase 14 — Experiment Manager and Reproducibility."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.experiments.manager import (
    ExperimentManager,
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


PLANNER_ALGOS = {"astar", "rrt_star", "rrt*"}


def _make_minimal_config(
    algo: str = "astar", env: str = "gridworld", seed: int = 42
) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for testing."""
    is_planner = algo in PLANNER_ALGOS
    try:
        algo_cfg = (
            AlgorithmConfig(name=algo)
            if is_planner
            else AlgorithmConfig(name=algo, learning_rate=3e-4, gamma=0.99, batch_size=64)
        )
        return ExperimentConfig(
            name=f"{env}_{algo}_test",
            seed=seed,
            algorithm=algo_cfg,
            environment=EnvironmentConfig(name=env, max_steps=10),
            training=None
            if is_planner
            else TrainingConfig(total_timesteps=1, checkpoint_freq=0, log_interval=1),
            evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        )
    except Exception:
        algo_cfg = (
            AlgorithmConfig.model_construct(name=algo)
            if is_planner
            else AlgorithmConfig.model_construct(
                name=algo, learning_rate=3e-4, gamma=0.99, batch_size=64
            )
        )
        return ExperimentConfig.model_construct(
            name=f"{env}_{algo}_test",
            seed=seed,
            algorithm=algo_cfg,
            environment=EnvironmentConfig(name=env, max_steps=10),
            training=None
            if is_planner
            else TrainingConfig(total_timesteps=1, checkpoint_freq=0, log_interval=1),
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
        config2.environment.max_steps = 50000
        assert _make_experiment_id(config1) != _make_experiment_id(config2)

    def test_path_manipulation_defense(self) -> None:
        """Malicious environment or algorithm names with directory traversal components are sanitized."""
        config = _make_minimal_config()
        config.environment.name = "../../etc/passwd"
        config.algorithm.name = "../malicious/algo"
        exp_id = _make_experiment_id(config)
        assert ".." not in exp_id
        assert "/" not in exp_id
        assert "\\" not in exp_id

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            # Environment doesn't exist, will fail during algorithm/env setup,
            # but the output directory created must be strictly inside tmpdir
            res = manager.run(config=config)
            assert res.output_dir.is_relative_to(Path(tmpdir).resolve())


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
environment:
  name: "gridworld"
  max_steps: 10
evaluation:
  eval_episodes: 2
  deterministic: true
"""
            )

            manager = ExperimentManager(base_output_dir=tmp_path / "results")
            result = manager.run_from_config(
                cfg_file,
                seed_override=77,
            )

            assert result.success
            manifest = result.manifest

            # Verify source_config preserves original values
            assert manifest.source_config["seed"] == 42

            # Verify overrides are explicitly recorded
            assert manifest.overrides["seed"] == 77

            # Verify effective_config reflects overrides
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

    def test_all_advertised_overrides_modify_effective_config(self) -> None:
        """Every advertised override actually modifies effective_config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="ppo", env="gridworld", seed=42)
            overrides = {
                "training.total_timesteps": 500,
                "seed": 99,
                "algorithm.learning_rate": 0.001,
                "environment.max_steps": 25,
                "evaluation.eval_episodes": 3,
            }
            res = manager.run(config=config, overrides=overrides)
            assert res.success
            assert res.manifest.overrides == overrides
            assert res.manifest.effective_config["training"]["total_timesteps"] == 500
            assert res.manifest.effective_config["seed"] == 99
            assert res.manifest.effective_config["algorithm"]["learning_rate"] == 0.001
            assert res.manifest.effective_config["environment"]["max_steps"] == 25
            assert res.manifest.effective_config["evaluation"]["eval_episodes"] == 3
            # Source config preserved
            assert res.manifest.source_config["seed"] == 42
            assert res.manifest.source_config["training"]["total_timesteps"] == 1

    def test_invalid_override_fails_cleanly(self) -> None:
        """Invalid or unknown override keys produce clean failures rather than false manifests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config()
            res = manager.run(config=config, overrides={"unknown.nonexistent.key": "val"})
            assert not res.success
            assert res.manifest.evaluation_status == "failed"
            assert "Configuration override failed" in res.manifest.notes


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

    def test_run_rl_sac_experiment(self) -> None:
        """ExperimentManager runs real SAC training and evaluation end-to-end on continuous navigation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = ExperimentConfig(
                name="sac_test",
                seed=42,
                algorithm=AlgorithmConfig(
                    name="sac",
                    learning_rate=3e-4,
                    gamma=0.99,
                    batch_size=32,
                    parameters={"buffer_size": 1000, "learning_starts": 10},
                ),
                environment=EnvironmentConfig(name="navigation", max_steps=15),
                training=TrainingConfig(total_timesteps=20, checkpoint_freq=0, log_interval=10),
                evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
            )

            result = manager.run(config=config)
            assert result.success, f"SAC experiment failed: {result.error_message}"
            assert "model" in result.manifest.artifact_paths
            model_rel = result.manifest.artifact_paths["model"]
            assert (result.output_dir / model_rel).exists()
            assert (result.output_dir / "metrics.json").exists()
            assert result.manifest.artifact_paths["metrics"] == "metrics.json"

    def test_artifact_paths_are_portable_relative_paths(self) -> None:
        """All manifest artifact paths are relative strings, ensuring cross-machine portability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld")
            res = manager.run(config=config)
            assert res.success

            for name, path_str in res.manifest.artifact_paths.items():
                p = Path(path_str)
                assert not p.is_absolute(), f"Artifact '{name}' has absolute path: {path_str}"
                # Must resolve relative to output_dir
                assert (res.output_dir / p).exists(), (
                    f"Artifact '{name}' does not exist at {res.output_dir / p}"
                )


# ---------------------------------------------------------------------------
# Failure & Interruption Modes
# ---------------------------------------------------------------------------


class TestFailureAndInterruptionModes:
    """Tests proving failure, crash, and interruption behaviors are faithfully recorded."""

    def test_rl_training_failure_records_failed_manifest(self) -> None:
        """RL training crash saves failure manifest and returns unsuccessful ExperimentResult."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="ppo", env="gridworld")

            with patch("adaptive_rl.training.get_trainer") as mock_trainer_getter:
                mock_trainer = mock_trainer_getter.return_value
                mock_trainer.fit.side_effect = RuntimeError(
                    "Simulated training crash: CUDA out of memory"
                )

                res = manager.run(config=config)
                assert not res.success
                assert "Simulated training crash" in res.error_message
                assert (res.output_dir / "manifest.json").exists()
                assert res.manifest.evaluation_status == "failed"
                assert "Simulated training crash" in res.manifest.notes

    def test_planner_execution_failure_records_failed_manifest(self) -> None:
        """Planner execution crash saves failure manifest and returns unsuccessful ExperimentResult."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld")

            with patch("adaptive_rl.experiments.manager._resolve_planner_policy") as mock_resolve:
                mock_resolve.side_effect = ValueError("Corrupt planner parameters")

                res = manager.run(config=config)
                assert not res.success
                assert "Corrupt planner parameters" in res.error_message
                assert (res.output_dir / "manifest.json").exists()
                assert res.manifest.evaluation_status == "failed"

    def test_keyboard_interrupt_preserves_interrupted_manifest(self) -> None:
        """KeyboardInterrupt marks manifest as interrupted on disk before re-raising."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="ppo", env="gridworld")

            with patch("adaptive_rl.training.get_trainer") as mock_trainer_getter:
                mock_trainer = mock_trainer_getter.return_value
                mock_trainer.fit.side_effect = KeyboardInterrupt()

                with pytest.raises(KeyboardInterrupt):
                    manager.run(config=config)

            # Check that manifest was saved with 'interrupted' status
            experiments = manager.list_experiments()
            assert len(experiments) == 1
            assert experiments[0]["evaluation_status"] == "interrupted"


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


class TestModularOwnershipAndCompatibility:
    """Verifies that modularization maintains backward compatibility and clean ownership."""

    def test_reexported_classes_identity(self) -> None:
        from adaptive_rl.experiments.manager import (
            ExperimentManifest as ManagerManifestClass,
        )
        from adaptive_rl.experiments.manager import (
            ExperimentResult as ManagerResultClass,
        )
        from adaptive_rl.experiments.manifest import (
            ExperimentManifest as ManifestClass,
        )
        from adaptive_rl.experiments.manifest import (
            ExperimentResult as ResultClass,
        )

        assert ManagerManifestClass is ManifestClass
        assert ManagerResultClass is ResultClass

    def test_reexported_provenance_functions_identity(self) -> None:
        from adaptive_rl.experiments.manager import (
            collect_environment_provenance as MgrEnvProv,
        )
        from adaptive_rl.experiments.manager import (
            get_git_commit as MgrGitCommit,
        )
        from adaptive_rl.experiments.manager import (
            get_git_provenance as MgrGitProv,
        )
        from adaptive_rl.experiments.provenance import (
            collect_environment_provenance as ProvEnvProv,
        )
        from adaptive_rl.experiments.provenance import (
            get_git_commit as ProvGitCommit,
        )
        from adaptive_rl.experiments.provenance import (
            get_git_provenance as ProvGitProv,
        )

        assert MgrGitCommit is ProvGitCommit
        assert MgrGitProv is ProvGitProv
        assert MgrEnvProv is ProvEnvProv

    def test_reexported_artifact_functions_identity(self) -> None:
        from adaptive_rl.experiments.artifacts import (
            _make_experiment_id as ArtMakeExpId,
        )
        from adaptive_rl.experiments.artifacts import (
            _make_run_id as ArtMakeRunId,
        )
        from adaptive_rl.experiments.artifacts import (
            _sanitize_path_component as ArtSanitize,
        )
        from adaptive_rl.experiments.manager import (
            _make_experiment_id as MgrMakeExpId,
        )
        from adaptive_rl.experiments.manager import (
            _make_run_id as MgrMakeRunId,
        )
        from adaptive_rl.experiments.manager import (
            _sanitize_path_component as MgrSanitize,
        )

        assert MgrMakeExpId is ArtMakeExpId
        assert MgrMakeRunId is ArtMakeRunId
        assert MgrSanitize is ArtSanitize

    def test_manager_complete_artifact_and_boundary_validation(self) -> None:
        """Explicitly validates contents of config.yaml, source_config.yaml, metrics.json,

        metrics.csv, manifest.json, provenance fields, IDs, and failed experiment artifacts.
        """
        import csv
        import json

        import yaml

        from adaptive_rl.experiments.manifest import ExperimentManifest

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld", seed=42)

            # Successful run with overrides
            overrides = {"seed": 99}
            res = manager.run(config=config, overrides=overrides)
            assert res.success

            out = res.output_dir

            # 1. config.yaml contents validation
            config_file = out / "config.yaml"
            assert config_file.exists()
            with open(config_file, encoding="utf-8") as f:
                saved_config = yaml.safe_load(f)
            assert saved_config["seed"] == 99
            assert saved_config["algorithm"]["name"] == "astar"

            # 2. source_config.yaml contents validation
            source_file = out / "source_config.yaml"
            assert source_file.exists()
            with open(source_file, encoding="utf-8") as f:
                saved_source = yaml.safe_load(f)
            assert saved_source["seed"] == 42
            assert saved_source["algorithm"]["name"] == "astar"

            # 3. metrics.json contents validation
            metrics_json_file = out / "metrics.json"
            assert metrics_json_file.exists()
            with open(metrics_json_file, encoding="utf-8") as f:
                saved_metrics = json.load(f)
            assert isinstance(saved_metrics, dict)
            assert (
                "success_rate" in saved_metrics
                or "eval_success_rate" in saved_metrics
                or "mean_success_rate" in saved_metrics
                or "steps" in saved_metrics
            )

            # 4. metrics.csv contents validation
            metrics_csv_file = out / "metrics.csv"
            assert metrics_csv_file.exists()
            with open(metrics_csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert reader.fieldnames is not None
            assert len(reader.fieldnames) > 0

            # 5. manifest.json contents validation via ExperimentManifest.load
            manifest_file = out / "manifest.json"
            assert manifest_file.exists()
            manifest = ExperimentManifest.load(manifest_file)
            assert manifest.experiment_id == res.experiment_id
            assert manifest.run_id == res.run_id
            assert manifest.seed == 99
            assert manifest.evaluation_status == "completed"

            # 6. Provenance fields validation
            assert isinstance(manifest.python_version, str) and len(manifest.python_version) > 0
            assert isinstance(manifest.platform_info, str) and len(manifest.platform_info) > 0
            assert isinstance(manifest.package_versions, dict)
            assert "adaptive-rl" in manifest.package_versions

            # 7. Failed experiment artifacts validation
            fail_res = manager.run(config=config, overrides={"nonexistent_field": 123})
            assert not fail_res.success
            assert fail_res.output_dir.exists()
            fail_manifest_file = fail_res.output_dir / "manifest.json"
            assert fail_manifest_file.exists()
            fail_manifest = ExperimentManifest.load(fail_manifest_file)
            assert fail_manifest.evaluation_status == "failed"
            assert "Configuration override failed" in fail_manifest.notes
