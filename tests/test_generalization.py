"""Comprehensive unit and integration tests for Generalization to Unseen Environments.

Covers:
- TrainingDistributionWrapper seed clamping and history tracking
- Strict data integrity validation ensuring zero overlap between train and test distributions
- GeneralizationReport construction, serialization, and gap calculations
- GeneralizationEvaluator execution across disjoint seed distributions
- End-to-end GeneralizationExperimentRunner training and unseen testing pipeline
- CLI 'generalization' benchmark command execution and table rendering
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.cli import app
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.environments import make_env
from adaptive_rl.environments.seeded_wrapper import TrainingDistributionWrapper
from adaptive_rl.evaluation.generalization import (
    GeneralizationDistribution,
    GeneralizationEvaluator,
    GeneralizationReport,
)
from adaptive_rl.evaluation.metrics import EvaluationMetrics
from adaptive_rl.experiments.generalization_runner import GeneralizationExperimentRunner

runner = CliRunner()


def test_training_distribution_wrapper_lifecycle() -> None:
    """Verify TrainingDistributionWrapper strictly draws seeds from the designated training set."""
    raw_env = make_env("gridworld", width=5, height=5, num_obstacles=2)
    train_seeds = [101, 102, 103, 104, 105]
    wrapped = TrainingDistributionWrapper(
        env=raw_env,
        seeds=train_seeds,
        shuffle=True,
        rng_seed=42,
    )

    # Perform multiple resets without arguments
    for _ in range(25):
        obs, info = wrapped.reset()
        seed_used = info["training_distribution_seed"]
        assert seed_used in train_seeds
        assert wrapped.last_seed == seed_used

    assert len(wrapped.sampled_seeds_history) == 25
    assert set(wrapped.sampled_seeds_history).issubset(set(train_seeds))

    # Sequential sampling mode
    seq_wrapped = TrainingDistributionWrapper(
        env=raw_env,
        seeds=[201, 202],
        shuffle=False,
    )
    _, i1 = seq_wrapped.reset()
    _, i2 = seq_wrapped.reset()
    _, i3 = seq_wrapped.reset()
    assert i1["training_distribution_seed"] == 201
    assert i2["training_distribution_seed"] == 202
    assert i3["training_distribution_seed"] == 201

    # Explicit valid seed
    _, i_valid = wrapped.reset(seed=103)
    assert i_valid["training_distribution_seed"] == 103

    # Explicit invalid seed is shielded and sampled from training seeds
    _, i_invalid = wrapped.reset(seed=999)
    assert i_invalid["training_distribution_seed"] in train_seeds

    wrapped.close()
    seq_wrapped.close()


def test_generalization_distribution_disjoint_validation() -> None:
    """Verify GeneralizationDistribution enforces strict data integrity and zero seed overlap."""
    # Valid disjoint partition
    dist = GeneralizationDistribution(
        train_seeds=[1, 2, 3, 4],
        test_seeds=[5, 6, 7, 8],
    )
    assert len(dist.train_seeds) == 4
    assert len(dist.test_seeds) == 4

    # From ranges constructor
    dist_ranges = GeneralizationDistribution.from_ranges(
        train_range=(100, 150),
        test_range=(150, 200),
    )
    assert len(dist_ranges.train_seeds) == 50
    assert len(dist_ranges.test_seeds) == 50
    assert set(dist_ranges.train_seeds).isdisjoint(set(dist_ranges.test_seeds))

    # Overlapping partition raises ValueError
    with pytest.raises(ValueError, match="Data integrity violation: Found 2 overlapping seeds"):
        GeneralizationDistribution(
            train_seeds=[1, 2, 3, 4],
            test_seeds=[3, 4, 5, 6],
        )

    # Empty distributions raise ValueError
    with pytest.raises(ValueError, match="Training seed distribution cannot be empty"):
        GeneralizationDistribution(train_seeds=[], test_seeds=[1, 2])

    with pytest.raises(ValueError, match="Test seed distribution cannot be empty"):
        GeneralizationDistribution(train_seeds=[1, 2], test_seeds=[])


def test_generalization_report_serialization(tmp_path: Path) -> None:
    """Verify GeneralizationReport metrics calculation and JSON persistence."""
    train_m = EvaluationMetrics(
        episodes=10,
        mean_reward=80.0,
        std_reward=5.0,
        min_reward=70.0,
        max_reward=90.0,
        success_rate=0.90,
        collision_rate=0.05,
        mean_episode_length=15.0,
        std_episode_length=2.0,
    )
    test_m = EvaluationMetrics(
        episodes=10,
        mean_reward=60.0,
        std_reward=8.0,
        min_reward=40.0,
        max_reward=80.0,
        success_rate=0.70,
        collision_rate=0.15,
        mean_episode_length=18.0,
        std_episode_length=3.0,
    )

    report = GeneralizationReport(
        experiment_name="test_report",
        environment_name="gridworld",
        algorithm_name="ppo",
        train_seeds=[1, 2, 3],
        test_seeds=[4, 5, 6],
        train_metrics=train_m,
        test_metrics=test_m,
        generalization_gap_success=0.20,  # 0.90 - 0.70
        generalization_gap_reward=20.0,  # 80.0 - 60.0
        relative_success_retention=0.70 / 0.90,
    )

    assert report.generalization_gap_success == pytest.approx(0.20)
    assert report.relative_success_retention == pytest.approx(0.777777, rel=1e-3)

    out_file = tmp_path / "report.json"
    saved = report.save_json(out_file)
    assert saved.exists()

    with open(saved, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_name"] == "test_report"
    assert data["train_metrics"]["success_rate"] == 0.90
    assert data["test_metrics"]["success_rate"] == 0.70
    assert data["generalization_gap_success"] == 0.20


def test_generalization_evaluator_execution() -> None:
    """Verify GeneralizationEvaluator benchmarks across designated seed lists."""
    env = make_env("gridworld", width=5, height=5, num_obstacles=1, max_steps=20)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)

    evaluator = GeneralizationEvaluator(algorithm=algo, env=env)
    distribution = GeneralizationDistribution(
        train_seeds=[10, 11, 12],
        test_seeds=[20, 21, 22],
    )

    report = evaluator.evaluate_generalization(
        distribution=distribution,
        experiment_name="eval_test",
        deterministic=True,
    )

    assert report.train_metrics.episodes == 3
    assert report.test_metrics.episodes == 3
    assert isinstance(report.generalization_gap_success, float)
    assert isinstance(report.generalization_gap_reward, float)
    assert 0.0 <= report.train_metrics.success_rate <= 1.0
    assert 0.0 <= report.test_metrics.success_rate <= 1.0

    evaluator.close()


def test_generalization_experiment_runner_end_to_end(tmp_path: Path) -> None:
    """Verify GeneralizationExperimentRunner trains on train seeds and evaluates on unseen test seeds."""
    config = ExperimentConfig(
        name="verify_generalization_runner",
        seed=42,
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=0.0003,
            gamma=0.99,
            batch_size=32,
            parameters={"n_steps": 64},
        ),
        environment=EnvironmentConfig(
            name="gridworld",
            max_steps=20,
            parameters={"width": 5, "height": 5, "num_obstacles": 1},
        ),
        training=TrainingConfig(
            total_timesteps=128,
            checkpoint_freq=64,
            log_interval=1,
        ),
        evaluation=EvaluationConfig(
            eval_episodes=2,
            deterministic=True,
        ),
    )

    distribution = GeneralizationDistribution(
        train_seeds=[100, 101, 102],
        test_seeds=[200, 201, 202],
    )
    runner = GeneralizationExperimentRunner(distribution=distribution)
    report = runner.run(config=config)

    assert report.experiment_name == "verify_generalization_runner"
    assert report.environment_name == "gridworld"
    assert report.train_metrics.episodes == 3
    assert report.test_metrics.episodes == 3

    saved_report = (
        tmp_path / "results" / "generalization" / "verify_generalization_runner_report.json"
    )
    assert saved_report.exists()


def test_generalization_cli_command(tmp_path: Path) -> None:
    """Verify adaptive-rl generalization CLI command execution."""
    cfg_file = tmp_path / "test_gen.yaml"
    cfg_file.write_text(
        """
name: "cli_gen_test"
seed: 42
algorithm:
  name: "ppo"
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 32
  parameters:
    n_steps: 64
environment:
  name: "gridworld"
  max_steps: 20
  parameters:
    width: 5
    height: 5
    num_obstacles: 1
training:
  total_timesteps: 64
  checkpoint_freq: 32
  log_interval: 1
evaluation:
  eval_episodes: 2
  deterministic: true
output_dir: "{out}"
log_dir: "{logs}"
""".format(out=str(tmp_path / "out"), logs=str(tmp_path / "logs")),
        encoding="utf-8",
    )

    res = runner.invoke(
        app,
        [
            "generalization",
            "--config",
            str(cfg_file),
            "--train-count",
            "3",
            "--test-count",
            "3",
            "--train-start",
            "100",
            "--test-start",
            "200",
            "--output-report",
            str(tmp_path / "gen_cli_report.json"),
        ],
    )

    assert res.exit_code == 0
    assert "Generalization Benchmark" in res.output
    assert "Training Seed Distribution" in res.output
    assert "Unseen Test Seed Distribution" in res.output
    assert "Success Rate" in res.output
    assert "Success Retention" in res.output
    assert (tmp_path / "gen_cli_report.json").exists()
