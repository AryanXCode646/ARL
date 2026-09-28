import csv
import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.environments.drone import DroneNavigation3DEnv, ObstacleSphere3D
from adaptive_rl.evaluation.evaluator import (
    Evaluator,
    compare_policies,
    evaluate_random_policy,
    run_obstacle_density_experiment,
)
from adaptive_rl.evaluation.metrics import compute_trajectory_metrics


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


def test_trajectory_metrics_stationary() -> None:
    """Verify stationary trajectory produces 0 path length and 0 efficiency."""
    positions = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    goal = [5.0, 2.0, 3.0]
    metrics = compute_trajectory_metrics(positions, goal)

    assert metrics["path_length"] == 0.0
    assert metrics["path_efficiency"] == 0.0
    assert metrics["straight_line_distance"] == pytest.approx(4.0)


def test_trajectory_metrics_straight_line() -> None:
    """Verify perfect straight-line trajectory achieves ~1.0 path efficiency."""
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    goal = [5.0, 0.0, 0.0]
    metrics = compute_trajectory_metrics(positions, goal)

    assert metrics["path_length"] == pytest.approx(5.0)
    assert metrics["straight_line_distance"] == pytest.approx(5.0)
    assert metrics["path_efficiency"] == pytest.approx(1.0)


def test_trajectory_metrics_zigzag() -> None:
    """Verify non-straight trajectory yields efficiency strictly less than 1.0."""
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 2.0, 0.0],
        [4.0, 0.0, 0.0],
    ]
    goal = [4.0, 0.0, 0.0]
    metrics = compute_trajectory_metrics(positions, goal)

    assert metrics["straight_line_distance"] == pytest.approx(4.0)
    assert metrics["path_length"] > 4.0
    assert 0.0 < metrics["path_efficiency"] < 1.0


def test_trajectory_metrics_zero_length_and_empty() -> None:
    """Verify zero-length and single-waypoint trajectories do not raise division errors."""
    # Empty positions
    metrics_empty = compute_trajectory_metrics([], goal=[1.0, 1.0, 1.0])
    assert metrics_empty["path_length"] == 0.0
    assert metrics_empty["path_efficiency"] == 0.0
    assert metrics_empty["straight_line_distance"] == 0.0

    # Single position
    metrics_single = compute_trajectory_metrics([[1.0, 1.0, 1.0]], goal=[4.0, 1.0, 1.0])
    assert metrics_single["path_length"] == 0.0
    assert metrics_single["path_efficiency"] == 0.0
    assert metrics_single["straight_line_distance"] == pytest.approx(3.0)


def test_trajectory_metrics_obstacle_clearance_surface() -> None:
    """Verify closest distance calculation measures to spherical obstacle surface, not center."""
    obs = ObstacleSphere3D(center=np.array([5.0, 5.0, 5.0]), radius=1.5)
    # p1: dist to center = 5.0, surface clearance = 5.0 - 1.5 = 3.5
    # p2: dist to center = 2.0, surface clearance = 2.0 - 1.5 = 0.5
    # p3: dist to center = 6.0, surface clearance = 6.0 - 1.5 = 4.5
    positions = [[0.0, 5.0, 5.0], [3.0, 5.0, 5.0], [11.0, 5.0, 5.0]]
    goal = [11.0, 5.0, 5.0]

    metrics = compute_trajectory_metrics(positions, goal, obstacles=[obs])
    assert metrics["min_obstacle_clearance"] == pytest.approx(0.5)


def test_trajectory_metrics_max_velocity() -> None:
    """Verify maximum velocity computation against known synthetic velocity vectors."""
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    goal = [1.0, 0.0, 0.0]
    velocities = [
        [1.0, 2.0, 2.0],  # norm = 3.0
        [0.0, 4.0, 3.0],  # norm = 5.0
        [2.0, 0.0, 0.0],  # norm = 2.0
    ]
    metrics = compute_trajectory_metrics(positions, goal, velocities=velocities)
    assert metrics["max_velocity"] == pytest.approx(5.0)


def test_trajectory_metrics_max_acceleration() -> None:
    """Verify maximum acceleration computation against known synthetic acceleration vectors."""
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    goal = [1.0, 0.0, 0.0]
    accelerations = [
        [0.0, 0.0, 0.0],  # norm = 0.0
        [1.0, 2.0, 2.0],  # norm = 3.0
        [-2.0, 1.0, 2.0],  # norm = 3.0
    ]
    metrics = compute_trajectory_metrics(positions, goal, accelerations=accelerations)
    assert metrics["max_acceleration"] == pytest.approx(3.0)


def test_obstacle_vs_boundary_collision_separation() -> None:
    """Verify obstacle collisions and boundary collisions are tracked separately."""

    class DummyEnv:
        def __init__(self) -> None:
            self.episode = 0
            self.action_space = None
            self.observation_space = None
            self.obstacles = []

        def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
            self.episode += 1
            return np.zeros(29, dtype=np.float32), {
                "drone_position": np.array([0.0, 0.0, 0.0]),
                "target_position": np.array([5.0, 5.0, 5.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "acceleration": np.array([0.0, 0.0, 0.0]),
                "obstacles": [],
            }

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
            if self.episode == 1:
                # Obstacle collision
                info = {
                    "drone_position": np.array([1.0, 0.0, 0.0]),
                    "target_position": np.array([5.0, 5.0, 5.0]),
                    "velocity": np.array([1.0, 0.0, 0.0]),
                    "acceleration": np.array([0.5, 0.0, 0.0]),
                    "collision": True,
                    "collision_type": "obstacle",
                    "success": False,
                    "obstacles": [],
                }
                return np.zeros(29, dtype=np.float32), -50.0, True, False, info
            elif self.episode == 2:
                # Boundary collision
                info = {
                    "drone_position": np.array([0.0, 10.0, 0.0]),
                    "target_position": np.array([5.0, 5.0, 5.0]),
                    "velocity": np.array([0.0, 2.0, 0.0]),
                    "acceleration": np.array([0.0, 1.0, 0.0]),
                    "collision": True,
                    "collision_type": "boundary_y",
                    "success": False,
                    "obstacles": [],
                }
                return np.zeros(29, dtype=np.float32), -50.0, True, False, info
            else:
                # Success
                info = {
                    "drone_position": np.array([5.0, 5.0, 5.0]),
                    "target_position": np.array([5.0, 5.0, 5.0]),
                    "velocity": np.array([0.1, 0.0, 0.0]),
                    "acceleration": np.array([0.0, 0.0, 0.0]),
                    "collision": False,
                    "collision_type": None,
                    "success": True,
                    "obstacles": [],
                }
                return np.zeros(29, dtype=np.float32), 100.0, True, False, info

        def close(self) -> None:
            pass

    dummy_env = DummyEnv()
    evaluator = Evaluator(algorithm=None, env=dummy_env)  # type: ignore[arg-type]
    metrics = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=1)

    assert metrics.episodes == 3
    assert metrics.collision_rate == pytest.approx(2 / 3)
    assert metrics.obstacle_collision_count == 1
    assert metrics.obstacle_collision_rate == pytest.approx(1 / 3)
    assert metrics.boundary_collision_count == 1
    assert metrics.boundary_collision_rate == pytest.approx(1 / 3)
    assert metrics.success_rate == pytest.approx(1 / 3)


def test_evaluator_trajectory_metrics_and_csv_export(tmp_path: Path) -> None:
    """Verify evaluator produces trajectory metrics and exports to JSON and CSV."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = Evaluator(algorithm=algo, env=env)

    metrics = evaluator.evaluate(num_episodes=2, deterministic=True, base_seed=42)

    # Trajectory metrics populated
    assert metrics.mean_path_length is not None
    assert metrics.mean_straight_line_distance is not None
    assert metrics.mean_path_efficiency is not None
    assert metrics.mean_max_velocity is not None
    assert metrics.mean_max_acceleration is not None
    assert metrics.obstacle_collision_count is not None
    assert metrics.boundary_collision_count is not None
    assert metrics.obstacle_collision_rate is not None
    assert metrics.boundary_collision_rate is not None

    # JSON export
    json_path = tmp_path / "eval.json"
    evaluator.save_report(metrics, json_path)
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "mean_path_length" in data
    assert "mean_path_efficiency" in data
    assert "obstacle_collision_count" in data
    assert "boundary_collision_count" in data

    # CSV export
    csv_path = tmp_path / "eval.csv"
    evaluator.save_csv_report(metrics, csv_path)
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        assert "episode_return" in row
        assert "path_length" in row
        assert "path_efficiency" in row
        assert "obstacle_collision_count" in row
        assert "boundary_collision_count" in row

    env.close()


def test_train_test_seeds_disjoint() -> None:
    """Verify that the train and test seed sets are strictly disjoint (zero intersection)."""
    from adaptive_rl.evaluation.generalization import get_split_seeds

    train_seeds = get_split_seeds("train")
    test_seeds = get_split_seeds("test")

    assert len(train_seeds) > 0
    assert len(test_seeds) > 0
    assert set(train_seeds).isdisjoint(set(test_seeds))
    assert len(set(train_seeds).intersection(set(test_seeds))) == 0


def test_train_test_seed_generation_deterministic() -> None:
    """Verify that train and test seed generation is 100% deterministic and repeatable."""
    from adaptive_rl.evaluation.generalization import get_split_seeds

    train_seeds_1 = get_split_seeds("train", num_episodes=20)
    train_seeds_2 = get_split_seeds("train", num_episodes=20)
    assert train_seeds_1 == train_seeds_2
    assert len(train_seeds_1) == 20

    test_seeds_1 = get_split_seeds("test", num_episodes=20)
    test_seeds_2 = get_split_seeds("test", num_episodes=20)
    assert test_seeds_1 == test_seeds_2
    assert len(test_seeds_1) == 20

    # Prefix consistency
    train_prefix = get_split_seeds("train", num_episodes=10)
    assert train_prefix == train_seeds_1[:10]


def test_same_test_seeds_produce_reproducible_layouts() -> None:
    """Verify that identical test seeds produce identical obstacle layouts."""
    env = DroneNavigation3DEnv(bounds=(30.0, 30.0, 15.0), num_obstacles=4)

    test_seed = 1042
    _, _ = env.reset(seed=test_seed)
    obstacles_run1 = [(obs.center.copy(), obs.radius) for obs in env._obstacles]

    _, _ = env.reset(seed=test_seed)
    obstacles_run2 = [(obs.center.copy(), obs.radius) for obs in env._obstacles]

    assert len(obstacles_run1) == len(obstacles_run2) == 4
    for (c1, r1), (c2, r2) in zip(obstacles_run1, obstacles_run2):
        np.testing.assert_allclose(c1, c2, rtol=1e-6)
        assert r1 == pytest.approx(r2)

    env.close()


def test_train_and_test_representative_layouts_are_distinct() -> None:
    """Verify that representative train layouts and test layouts produce distinct geometry."""
    env = DroneNavigation3DEnv(bounds=(30.0, 30.0, 15.0), num_obstacles=4)

    # Train layout (seed 42)
    env.reset(seed=42)
    train_obstacles = [(obs.center.copy(), obs.radius) for obs in env._obstacles]

    # Test layout (seed 1042)
    env.reset(seed=1042)
    test_obstacles = [(obs.center.copy(), obs.radius) for obs in env._obstacles]

    centers_train = np.array([c for c, _ in train_obstacles])
    centers_test = np.array([c for c, _ in test_obstacles])

    # Obstacle coordinate layouts must not be identical
    assert not np.allclose(centers_train, centers_test)
    env.close()


def test_environment_split_enforcement_and_leakage_prevention() -> None:
    """Verify environment enforces split boundaries and strictly blocks cross-split seeds."""
    # 1. Train environment draws only train seeds
    train_env = DroneNavigation3DEnv(split="train")
    for _ in range(5):
        _, info = train_env.reset()
        assert info["split"] == "train"
        assert 0 <= info["split_seed"] < 1000

    # Train environment rejects test seed
    with pytest.raises(ValueError, match="out of bounds for 'train' split"):
        train_env.reset(seed=1005)

    # 2. Test environment draws only test seeds
    test_env = DroneNavigation3DEnv(split="test")
    for _ in range(5):
        _, info = test_env.reset()
        assert info["split"] == "test"
        assert 1000 <= info["split_seed"] < 1200

    # Test environment rejects train seed
    with pytest.raises(ValueError, match="out of bounds for 'test' split"):
        test_env.reset(seed=5)

    # Invalid split name
    with pytest.raises(ValueError, match="Invalid split"):
        DroneNavigation3DEnv(split="validation")

    train_env.close()
    test_env.close()


def test_generalization_gap_calculation_normal() -> None:
    """Verify standard generalization gap computation: Delta = Train - Test."""
    from adaptive_rl.evaluation.generalization import compute_generalization_gap
    from adaptive_rl.evaluation.metrics import EvaluationMetrics

    train_m = EvaluationMetrics(
        episodes=20,
        mean_reward=80.0,
        std_reward=5.0,
        min_reward=70.0,
        max_reward=90.0,
        success_rate=0.85,
        collision_rate=0.10,
        mean_episode_length=120.0,
    )
    test_m = EvaluationMetrics(
        episodes=20,
        mean_reward=30.0,
        std_reward=10.0,
        min_reward=10.0,
        max_reward=50.0,
        success_rate=0.55,
        collision_rate=0.40,
        mean_episode_length=80.0,
    )

    gap = compute_generalization_gap(train_m, test_m)

    assert gap.success_gap == pytest.approx(0.85 - 0.55)  # +0.30
    assert gap.reward_gap == pytest.approx(80.0 - 30.0)  # +50.0
    assert gap.collision_gap == pytest.approx(0.10 - 0.40)  # -0.30


def test_generalization_gap_edge_cases() -> None:
    """Verify generalization gap calculation under boundary conditions."""
    from adaptive_rl.evaluation.generalization import compute_generalization_gap
    from adaptive_rl.evaluation.metrics import EvaluationMetrics

    # Case 1: 100% train success, 0% test success
    m_100 = EvaluationMetrics(
        episodes=10,
        mean_reward=100.0,
        success_rate=1.0,
        collision_rate=0.0,
        mean_episode_length=50.0,
    )
    m_0 = EvaluationMetrics(
        episodes=10,
        mean_reward=-50.0,
        success_rate=0.0,
        collision_rate=1.0,
        mean_episode_length=20.0,
    )
    gap1 = compute_generalization_gap(m_100, m_0)
    assert gap1.success_gap == pytest.approx(1.0)
    assert gap1.reward_gap == pytest.approx(150.0)

    # Case 2: 0% train success, 100% test success
    gap2 = compute_generalization_gap(m_0, m_100)
    assert gap2.success_gap == pytest.approx(-1.0)
    assert gap2.reward_gap == pytest.approx(-150.0)

    # Case 3: 0% train success, 0% test success
    gap3 = compute_generalization_gap(m_0, m_0)
    assert gap3.success_gap == pytest.approx(0.0)
    assert gap3.reward_gap == pytest.approx(0.0)

    # Case 4: 100% train success, 100% test success
    gap4 = compute_generalization_gap(m_100, m_100)
    assert gap4.success_gap == pytest.approx(0.0)
    assert gap4.reward_gap == pytest.approx(0.0)

    # Case 5: Empty episodes error
    m_empty = EvaluationMetrics(
        episodes=1,  # Pydantic requires >0
        mean_reward=0.0,
        mean_episode_length=0.0,
    )
    object.__setattr__(m_empty, "episodes", 0)
    with pytest.raises(ValueError, match="empty evaluation episodes"):
        compute_generalization_gap(m_empty, m_100)

    # Case 6: Non-finite reward error
    m_nan = EvaluationMetrics(
        episodes=10,
        mean_reward=float("nan"),
        mean_episode_length=10.0,
    )
    with pytest.raises(ValueError, match="Non-finite mean reward"):
        compute_generalization_gap(m_nan, m_100)


def test_rerunning_test_evaluation_is_deterministic() -> None:
    """Verify that re-running test split evaluation produces deterministic identical results."""
    from adaptive_rl.algorithms.random_policy import RandomPolicy

    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    policy = RandomPolicy(action_space=env.action_space, seed=123)
    evaluator = Evaluator(algorithm=policy, env=env)

    # First test split evaluation
    metrics1 = evaluator.evaluate(num_episodes=3, deterministic=True, split="test")
    records1 = list(evaluator.last_episode_records)

    # Second test split evaluation under identical conditions
    # Re-instantiate policy with same seed
    policy2 = RandomPolicy(action_space=env.action_space, seed=123)
    evaluator2 = Evaluator(algorithm=policy2, env=env)
    metrics2 = evaluator2.evaluate(num_episodes=3, deterministic=True, split="test")
    records2 = list(evaluator2.last_episode_records)

    assert metrics1.episodes == metrics2.episodes == 3
    assert metrics1.mean_reward == pytest.approx(metrics2.mean_reward)
    assert metrics1.success_rate == metrics2.success_rate
    assert metrics1.collision_rate == metrics2.collision_rate

    # Seeds evaluated are identical
    seeds1 = [r.seed for r in records1]
    seeds2 = [r.seed for r in records2]
    assert seeds1 == seeds2 == [1000, 1001, 1002]

    env.close()


def test_evaluate_generalization_workflow_and_json_export(tmp_path: Path) -> None:
    """Verify evaluate_generalization runs both splits, computes gaps, and exports JSON."""
    from adaptive_rl.algorithms.random_policy import RandomPolicy
    from adaptive_rl.evaluation.generalization import evaluate_generalization

    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=15, num_obstacles=1)
    policy = RandomPolicy(action_space=env.action_space, seed=42)

    json_target = tmp_path / "generalization_benchmark.json"
    result = evaluate_generalization(
        algorithm=policy,
        env=env,
        num_episodes=2,
        deterministic=True,
        output_path=json_target,
    )

    assert json_target.exists()

    # Verify result structure
    assert result.train["episodes"] == 2
    assert result.test["episodes"] == 2
    assert len(result.train["seeds"]) == 2
    assert len(result.test["seeds"]) == 2

    # Verify seed disjointness
    assert set(result.train["seeds"]).isdisjoint(set(result.test["seeds"]))

    # Verify gap fields
    assert "success" in result.generalization_gap
    assert "reward" in result.generalization_gap
    assert isinstance(result.generalization_gap["reward"], float)

    # Verify serialized JSON content
    with open(json_target, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "train" in data
    assert "test" in data
    assert "generalization_gap" in data
    assert data["train"]["seeds"] == [0, 1]
    assert data["test"]["seeds"] == [1000, 1001]
    assert "success" in data["generalization_gap"]
    assert "reward" in data["generalization_gap"]

    env.close()
