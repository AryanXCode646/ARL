"""Comprehensive tests for EvaluationMetrics, Evaluator, scenario benchmarks, and CLI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.cli import app
from adaptive_rl.environments.gridworld.grid import GridWorldEnv
from adaptive_rl.environments.testing import DummyTestEnv
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.evaluation.metrics import EvaluationMetrics
from adaptive_rl.evaluation.scenarios import EvaluationScenario

runner = CliRunner()


def test_evaluation_metrics_model() -> None:
    """Verify EvaluationMetrics schema fields and bounds."""
    metrics = EvaluationMetrics(
        episodes=10,
        mean_reward=50.5,
        std_reward=5.2,
        min_reward=40.0,
        max_reward=60.0,
        success_rate=0.8,
        collision_rate=0.2,
        mean_episode_length=15.3,
        std_episode_length=2.1,
    )
    assert metrics.episodes == 10
    assert metrics.mean_reward == 50.5
    assert metrics.success_rate == 0.8
    assert metrics.collision_rate == 0.2
    assert metrics.min_reward <= metrics.max_reward


def test_evaluator_dummy_env() -> None:
    """Verify Evaluator executes benchmark over DummyTestEnv."""
    env = DummyTestEnv(step_limit=10, reward_step=2.0)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32)
    evaluator = Evaluator(algorithm=algo, env=env)

    metrics = evaluator.evaluate(num_episodes=5, deterministic=True, base_seed=42)
    assert metrics.episodes == 5
    assert metrics.mean_reward == 20.0  # 10 steps * 2.0 reward
    assert metrics.std_reward == 0.0
    assert metrics.mean_episode_length == 10.0
    env.close()


def test_evaluator_gridworld_metrics() -> None:
    """Verify Evaluator tracks returns, collisions, and successes in GridWorld."""
    env = GridWorldEnv(width=5, height=5, num_obstacles=2, max_steps=20)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = Evaluator(algorithm=algo, env=env)

    metrics = evaluator.evaluate(num_episodes=10, deterministic=True, base_seed=100)
    assert metrics.episodes == 10
    assert 0.0 <= metrics.success_rate <= 1.0
    assert 0.0 <= metrics.collision_rate <= 1.0
    assert metrics.min_reward <= metrics.mean_reward <= metrics.max_reward
    assert 1.0 <= metrics.mean_episode_length <= 20.0
    env.close()


def test_evaluator_deterministic_reproducibility() -> None:
    """Verify identical base_seed produces identical episodic trajectories in GridWorld."""
    env = GridWorldEnv(width=5, height=5, num_obstacles=2, max_steps=20)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = Evaluator(algorithm=algo, env=env)

    metrics1 = evaluator.evaluate(num_episodes=5, deterministic=True, base_seed=777)
    metrics2 = evaluator.evaluate(num_episodes=5, deterministic=True, base_seed=777)

    assert metrics1.mean_reward == metrics2.mean_reward
    assert metrics1.mean_episode_length == metrics2.mean_episode_length
    assert metrics1.additional_metrics["all_rewards"] == metrics2.additional_metrics["all_rewards"]
    env.close()


def test_evaluator_scenario_benchmarks() -> None:
    """Verify evaluating an agent across multiple EvaluationScenario configurations."""
    env = GridWorldEnv(width=5, height=5, num_obstacles=1, max_steps=20)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = Evaluator(
        algorithm=algo, env_name="gridworld", env_kwargs={"width": 5, "height": 5}
    )

    scenarios = [
        EvaluationScenario(name="low_density", seed=10, environment_overrides={"num_obstacles": 1}),
        EvaluationScenario(
            name="high_density", seed=20, environment_overrides={"num_obstacles": 4}
        ),
    ]

    results = evaluator.evaluate_scenarios(scenarios=scenarios, deterministic=True)
    assert "low_density" in results
    assert "high_density" in results
    assert results["low_density"].episodes == 1
    assert results["high_density"].episodes == 1
    env.close()


def test_evaluator_save_report(tmp_path: Path) -> None:
    """Verify saving metrics report to JSON file and validating structure."""
    env = DummyTestEnv(step_limit=5)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32)
    evaluator = Evaluator(algorithm=algo, env=env)

    metrics = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=42)
    report_file = tmp_path / "eval_report.json"
    saved_path = evaluator.save_report(metrics, report_file)

    assert saved_path.exists()
    content = json.loads(saved_path.read_text(encoding="utf-8"))
    assert content["episodes"] == 3
    assert "mean_reward" in content
    assert "success_rate" in content
    assert "collision_rate" in content
    env.close()


def test_cli_evaluate_execution(tmp_path: Path) -> None:
    """Verify adaptive-rl evaluate executes evaluation benchmark and displays results table."""
    # 1. Train a quick model artifact on GridWorld
    env = GridWorldEnv(width=4, height=4, num_obstacles=1, max_steps=10)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32)
    algo.train(64)
    model_path = tmp_path / "model.zip"
    algo.save(model_path)
    env.close()

    config_path = tmp_path / "eval_cfg.yaml"
    config_path.write_text(
        """
name: "cli_eval_test"
seed: 42
algorithm:
  name: "ppo"
environment:
  name: "gridworld"
  max_steps: 10
  parameters:
    width: 4
    height: 4
    num_obstacles: 1
training:
  total_timesteps: 64
evaluation:
  eval_episodes: 3
  deterministic: true
""",
        encoding="utf-8",
    )

    report_path = tmp_path / "metrics_out.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(config_path),
            "--model",
            str(model_path),
            "--episodes",
            "3",
            "--output-report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert "Starting Evaluation: cli_eval_test" in result.output
    assert "Benchmark Results (3 episodes)" in result.output
    assert "Mean Reward" in result.output
    assert "Success Rate" in result.output
    assert report_path.exists()
