"""Tests for agent evaluation and JSON report generation."""

import json
from pathlib import Path

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.evaluation.evaluator import Evaluator


def test_evaluator_deterministic_evaluation(tmp_path: Path) -> None:
    """Verify evaluation generates deterministic results with fixed seed."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)

    evaluator = Evaluator(algorithm=algo, env=env)
    metrics1 = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=42)
    metrics2 = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=42)

    assert metrics1.episodes == 3
    assert metrics1.mean_reward == metrics2.mean_reward
    assert metrics1.success_rate == metrics2.success_rate
    assert metrics1.collision_rate == metrics2.collision_rate
    assert metrics1.mean_episode_length == metrics2.mean_episode_length

    # Verify JSON report creation and structure
    report_file = tmp_path / "evaluation.json"
    saved = evaluator.save_report(metrics1, report_file)
    assert saved.exists()

    with open(saved, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["episodes"] == 3
    assert "success_rate" in data
    assert "collision_rate" in data
    assert "mean_reward" in data
    assert "mean_episode_length" in data
    env.close()


def test_evaluator_episode_records() -> None:
    """Verify individual episode records are tracked correctly."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = Evaluator(algorithm=algo, env=env)

    evaluator.evaluate(num_episodes=2, deterministic=True, base_seed=10)
    assert len(evaluator.last_episode_records) == 2
    rec = evaluator.last_episode_records[0]
    assert rec.episode_index == 0
    assert rec.seed == 10
    assert isinstance(rec.return_value, float)
    assert isinstance(rec.success, bool)
    assert isinstance(rec.collision, bool)
    env.close()
