"""Tests for agent evaluation and JSON report generation."""

import csv
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.evaluation.evaluator import (
    Evaluator,
    compare_policies,
    evaluate_random_policy,
    run_obstacle_density_experiment,
)
from adaptive_rl.evaluation.statistics import (
    student_t_critical_value,
    summarize_seed_values,
)


class _SeedOutcomeEnv(gym.Env):
    observation_space = gym.spaces.Box(-1000.0, 1000.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self, *, include_outcomes: bool = True) -> None:
        super().__init__()
        self.include_outcomes = include_outcomes
        self.current_seed = 0
        self.position = np.zeros(2, dtype=np.float64)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_seed = 0 if seed is None else seed
        self.position = np.array([float(self.current_seed), 0.0])
        return np.zeros(1, dtype=np.float32), {"position": self.position.copy()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.position = self.position + np.array([1.0, 0.0])
        info = {"position": self.position.copy()}
        if self.include_outcomes:
            outcome = self.current_seed % 3
            info.update(
                {
                    "success": outcome == 0,
                    "collision": outcome == 1,
                }
            )
            return (
                np.zeros(1, dtype=np.float32),
                float(self.current_seed),
                outcome != 2,
                outcome == 2,
                info,
            )
        return np.zeros(1, dtype=np.float32), float(self.current_seed), True, False, info


class _ZeroPolicy:
    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        return np.zeros(1, dtype=np.float32), None


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


def test_student_t_statistics_match_analytical_values() -> None:
    assert student_t_critical_value(0.95, 1) == pytest.approx(12.7062047364, rel=1e-9)
    assert student_t_critical_value(0.95, 2) == pytest.approx(4.3026527297, rel=1e-9)
    assert student_t_critical_value(0.95, 5) == pytest.approx(2.5705818356, rel=1e-9)
    assert student_t_critical_value(0.95, 9) == pytest.approx(2.2621571627, rel=1e-9)
    assert student_t_critical_value(0.95, 10) == pytest.approx(2.2281388520, rel=1e-9)
    assert student_t_critical_value(0.95, 30) == pytest.approx(2.0422724563, rel=1e-9)

    stats = summarize_seed_values([1.0, 2.0, 3.0])
    margin = 4.3026527297 / math.sqrt(3.0)
    assert stats.mean == pytest.approx(2.0)
    assert stats.std == pytest.approx(1.0)
    assert stats.ci95_lower == pytest.approx(2.0 - margin)
    assert stats.ci95_upper == pytest.approx(2.0 + margin)
    assert stats.sample_count == 3


def test_student_t_statistics_handle_small_and_constant_samples() -> None:
    one = summarize_seed_values([7.0])
    assert one.mean == 7.0
    assert one.std is None
    assert one.ci95_lower is None
    assert one.ci95_upper is None

    two = summarize_seed_values([0.0, 2.0])
    assert two.mean == 1.0
    assert two.std == pytest.approx(math.sqrt(2.0))
    assert two.ci95_lower is not None and math.isfinite(two.ci95_lower)
    assert two.ci95_upper is not None and math.isfinite(two.ci95_upper)

    constant = summarize_seed_values([3.0, 3.0, 3.0])
    assert constant.mean == 3.0
    assert constant.std == 0.0
    assert constant.ci95_lower == 3.0
    assert constant.ci95_upper == 3.0

    assert summarize_seed_values([None, None]).mean is None
    with pytest.raises(ValueError, match="finite"):
        summarize_seed_values([1.0, float("nan")])


def test_evaluate_seeds_preserves_per_seed_records_and_metrics(tmp_path: Path) -> None:
    env = _SeedOutcomeEnv()
    evaluator = Evaluator(algorithm=_ZeroPolicy(), env=env)  # type: ignore[arg-type]
    requested_seeds = [10, 20]
    result = evaluator.evaluate_seeds(requested_seeds, episodes_per_seed=2)

    assert result.seeds == [10, 20]
    assert requested_seeds == [10, 20]
    assert result.total_episodes == 4
    assert [(record.seed, record.episode_index) for record in result.episodes] == [
        (10, 0),
        (10, 1),
        (20, 0),
        (20, 1),
    ]
    assert [record.episode_seed for record in result.episodes] == [20, 21, 40, 41]
    assert all(record.path_length == 1.0 for record in result.episodes)
    assert [summary.seed for summary in result.per_seed] == requested_seeds
    assert [summary.episodes for summary in result.per_seed] == [2, 2]
    assert [summary.mean_reward for summary in result.per_seed] == [20.5, 40.5]
    assert result.per_seed[0].success_rate == pytest.approx(0.5)
    assert result.per_seed[0].collision_rate == pytest.approx(0.0)
    assert result.per_seed[0].truncation_rate == pytest.approx(0.5)
    assert result.per_seed[0].mean_episode_length == 1.0
    assert result.per_seed[0].path_length == 1.0
    assert result.aggregate["mean_reward"].mean == pytest.approx(30.5)
    assert result.aggregate["mean_reward"].std == pytest.approx(math.sqrt(200.0))

    json_path = tmp_path / "evaluation_multiseed.json"
    csv_path = tmp_path / "evaluation_multiseed.csv"
    saved_json, saved_csv = evaluator.save_multiseed_report(result, json_path, csv_path)
    assert saved_json.is_file()
    assert saved_csv.is_file()
    document = json.loads(saved_json.read_text(encoding="utf-8"))
    json.dumps(document, allow_nan=False)
    assert document["metadata"] == {
        "seeds": [10, 20],
        "seed_count": 2,
        "episodes_per_seed": 2,
        "total_episodes": 4,
        "deterministic": True,
        "environment": "drone",
        "confidence_interval": (
            "two-sided 95% Student's t interval across seed summaries; "
            "sample standard deviation; unavailable when fewer than two values exist"
        ),
        "duplicate_seed_policy": "rejected",
    }
    assert len(document["episodes"]) == 4
    assert len(document["per_seed"]) == 2
    assert document["aggregate"]["mean_reward"]["sample_count"] == 2
    assert document["episodes"][0]["episode_seed"] == 20
    assert document["episodes"][0]["path_length"] == 1.0

    expected_headers = [
        "metric",
        "mean",
        "std",
        "ci95_lower",
        "ci95_upper",
        "sample_count",
        "seed_count",
        "episodes_per_seed",
        "total_episodes",
    ]
    with saved_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == expected_headers
    assert len(rows) == len(result.aggregate)
    assert rows[0]["metric"] == "mean_reward"
    assert float(rows[0]["mean"]) == pytest.approx(30.5)
    assert rows[0]["seed_count"] == "2"
    assert rows[0]["episodes_per_seed"] == "2"
    assert rows[0]["total_episodes"] == "4"
    evaluator.close()


def test_evaluate_seeds_is_deterministic_and_rejects_invalid_inputs() -> None:
    evaluator = Evaluator(algorithm=_ZeroPolicy(), env=_SeedOutcomeEnv())  # type: ignore[arg-type]
    first = evaluator.evaluate_seeds([7, 13], episodes_per_seed=2, deterministic=True)
    first_data = first.to_dict()
    second = evaluator.evaluate_seeds([7, 13], episodes_per_seed=2, deterministic=True)
    assert second.to_dict() == first_data

    with pytest.raises(ValueError, match="must not be empty"):
        evaluator.evaluate_seeds([], episodes_per_seed=1)
    with pytest.raises(ValueError, match="unique"):
        evaluator.evaluate_seeds([7, 7], episodes_per_seed=1)
    with pytest.raises(ValueError, match="positive"):
        evaluator.evaluate_seeds([7], episodes_per_seed=0)
    with pytest.raises(ValueError, match="non-negative"):
        evaluator.evaluate_seeds([-1], episodes_per_seed=1)
    evaluator.close()


def test_evaluate_seeds_preserves_unavailable_optional_metrics() -> None:
    evaluator = Evaluator(
        algorithm=_ZeroPolicy(),
        env=_SeedOutcomeEnv(include_outcomes=False),  # type: ignore[arg-type]
    )
    result = evaluator.evaluate_seeds([2, 5], episodes_per_seed=1)
    assert all(summary.success_rate is None for summary in result.per_seed)
    assert all(summary.collision_rate is None for summary in result.per_seed)
    assert result.aggregate["success_rate"].mean is None
    evaluator.close()


def test_evaluate_random_policy() -> None:
    """Verify uniform-random policy evaluation baseline executes and returns valid metrics."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=2)
    metrics = evaluate_random_policy(env=env, num_episodes=3, base_seed=42)

    assert metrics.episodes == 3
    assert isinstance(metrics.mean_reward, float)
    assert 0.0 <= (metrics.success_rate or 0.0) <= 1.0
    assert 0.0 <= (metrics.collision_rate or 0.0) <= 1.0
    assert metrics.mean_episode_length > 0
    env.close()


def test_compare_policies() -> None:
    """Verify head-to-head comparison between PPO and Random baseline."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)

    comparison = compare_policies(ppo_algorithm=algo, env=env, num_episodes=2, base_seed=42)
    assert "PPO" in comparison
    assert "Random Policy" in comparison
    assert comparison["PPO"].episodes == 2
    assert comparison["Random Policy"].episodes == 2
    env.close()


def test_run_obstacle_density_experiment(tmp_path: Path) -> None:
    """Verify obstacle-density experiment runs across varied obstacle counts."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    output_file = tmp_path / "density_exp.json"

    results = run_obstacle_density_experiment(
        algorithm=algo,
        obstacle_counts=(2, 4),
        episodes_per_density=2,
        base_seed=42,
        bounds=(20.0, 20.0, 10.0),
        output_path=output_file,
    )

    assert len(results) == 2
    assert results[0]["obstacle_count"] == 2
    assert results[1]["obstacle_count"] == 4
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert len(saved_data) == 2
    assert "success_rate" in saved_data[0]
    assert "collision_rate" in saved_data[0]
    env.close()
