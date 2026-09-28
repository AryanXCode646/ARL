"""Comprehensive tests for reward-function ablation framework (Issue #249)."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from adaptive_rl.benchmarking.ablation import (
    REWARD_ABLATION_VARIANTS,
    ConvergenceEvaluationCallback,
    get_ablation_variant,
    run_reward_ablation_experiment,
)
from adaptive_rl.environments.drone import DroneNavigation3DEnv


def test_variant_a_configuration() -> None:
    """Requirement 1: Verify exact Variant A configuration (Progress Only)."""
    var = get_ablation_variant("A")
    assert var.id == "A"
    assert "Progress Only" in var.name
    assert var.progress_weight == 2.0
    assert var.goal_reward == 100.0
    assert var.step_penalty == 0.0
    assert var.action_penalty_weight == 0.0
    assert var.collision_reward == 0.0

    params = var.get_reward_parameters()
    assert params["progress_weight"] == 2.0
    assert params["goal_reward"] == 100.0
    assert params["step_penalty"] == 0.0
    assert params["action_penalty_weight"] == 0.0
    assert params["collision_reward"] == 0.0


def test_variant_b_configuration() -> None:
    """Requirement 2: Verify exact Variant B configuration (Progress + Collision)."""
    var = get_ablation_variant("B")
    assert var.id == "B"
    assert "Collision" in var.name
    assert var.progress_weight == 2.0
    assert var.goal_reward == 100.0
    assert var.step_penalty == 0.0
    assert var.action_penalty_weight == 0.0
    assert var.collision_reward == -100.0

    params = var.get_reward_parameters()
    assert params["progress_weight"] == 2.0
    assert params["goal_reward"] == 100.0
    assert params["step_penalty"] == 0.0
    assert params["action_penalty_weight"] == 0.0
    assert params["collision_reward"] == -100.0


def test_variant_c_configuration() -> None:
    """Requirement 3: Verify exact Variant C configuration (Progress + Collision + Step)."""
    var = get_ablation_variant("C")
    assert var.id == "C"
    assert "Step" in var.name
    assert var.progress_weight == 2.0
    assert var.goal_reward == 100.0
    assert var.step_penalty == -0.05
    assert var.action_penalty_weight == 0.0
    assert var.collision_reward == -100.0

    params = var.get_reward_parameters()
    assert params["progress_weight"] == 2.0
    assert params["goal_reward"] == 100.0
    assert params["step_penalty"] == -0.05
    assert params["action_penalty_weight"] == 0.0
    assert params["collision_reward"] == -100.0


def test_variant_d_configuration() -> None:
    """Requirement 4: Verify exact Variant D configuration (Full Baseline)."""
    var = get_ablation_variant("D")
    assert var.id == "D"
    assert "Full Baseline" in var.name
    assert var.progress_weight == 2.0
    assert var.goal_reward == 100.0
    assert var.step_penalty == -0.05
    assert var.action_penalty_weight == 0.01
    assert var.collision_reward == -100.0

    params = var.get_reward_parameters()
    assert params["progress_weight"] == 2.0
    assert params["goal_reward"] == 100.0
    assert params["step_penalty"] == -0.05
    assert params["action_penalty_weight"] == 0.01
    assert params["collision_reward"] == -100.0


def test_variant_a_reward_pure_progress() -> None:
    """Requirement 5: Variant A non-terminal reward contains pure progress delta without step or effort penalties."""
    var = get_ablation_variant("A")
    env = DroneNavigation3DEnv(
        bounds=(30.0, 30.0, 15.0),
        max_steps=20,
        **var.get_reward_parameters(),
    )
    env.reset(seed=42)

    # Command a non-zero action with control effort
    action = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    prev_dist = env._prev_distance_to_goal

    obs, reward, terminated, truncated, info = env.step(action)
    curr_dist = env._prev_distance_to_goal
    dist_delta = prev_dist - curr_dist

    # Pure progress: 2.0 * dist_delta, no step penalty (0.0), no effort penalty (0.0)
    expected_reward = 2.0 * dist_delta
    assert reward == pytest.approx(expected_reward, abs=1e-5)
    assert not terminated

    # Collision reward in Variant A is 0.0
    env.close()

    env_coll = DroneNavigation3DEnv(
        bounds=(20.0, 20.0, 10.0),
        start_pos=np.array([0.5, 10.0, 5.0]),
        collision_radius=1.0,
        **var.get_reward_parameters(),
    )
    env_coll.reset(seed=42)
    _, coll_reward, terminated, _, info = env_coll.step(np.array([0.0, 0.0, 0.0]))
    assert info["collision"] is True
    assert coll_reward == 0.0
    env_coll.close()


def test_variant_b_reward_collision_and_no_step_effort() -> None:
    """Requirement 6: Variant B applies collision penalty (-100.0) while non-terminal steps have zero step/effort penalties."""
    var = get_ablation_variant("B")
    env = DroneNavigation3DEnv(
        bounds=(30.0, 30.0, 15.0),
        max_steps=20,
        **var.get_reward_parameters(),
    )
    env.reset(seed=42)

    # Non-terminal stepping with effort
    action = np.array([0.6, 0.6, 0.0], dtype=np.float32)
    prev_dist = env._prev_distance_to_goal
    obs, reward, terminated, truncated, info = env.step(action)
    curr_dist = env._prev_distance_to_goal
    dist_delta = prev_dist - curr_dist

    # Still zero step and effort penalty
    expected_reward = 2.0 * dist_delta
    assert reward == pytest.approx(expected_reward, abs=1e-5)
    env.close()

    # Collision reward is -100.0
    env_coll = DroneNavigation3DEnv(
        bounds=(20.0, 20.0, 10.0),
        start_pos=np.array([0.5, 10.0, 5.0]),
        collision_radius=1.0,
        **var.get_reward_parameters(),
    )
    env_coll.reset(seed=42)
    _, coll_reward, terminated, _, info = env_coll.step(np.array([0.0, 0.0, 0.0]))
    assert info["collision"] is True
    assert coll_reward == -100.0
    env_coll.close()


def test_variant_c_step_penalty_applied() -> None:
    """Requirement 7: Variant C applies step penalty (-0.05) and collision penalty (-100.0), without effort penalty."""
    var = get_ablation_variant("C")
    env = DroneNavigation3DEnv(
        bounds=(30.0, 30.0, 15.0),
        max_steps=20,
        **var.get_reward_parameters(),
    )
    env.reset(seed=42)

    action = np.array([0.4, 0.4, 0.0], dtype=np.float32)
    prev_dist = env._prev_distance_to_goal
    obs, reward, terminated, truncated, info = env.step(action)
    curr_dist = env._prev_distance_to_goal
    dist_delta = prev_dist - curr_dist

    # Progress + step penalty (-0.05), no effort penalty
    expected_reward = 2.0 * dist_delta - 0.05
    assert reward == pytest.approx(expected_reward, abs=1e-5)

    env.close()


def test_variant_d_matches_default_reward_behavior() -> None:
    """Requirement 8: Variant D exactly matches the current default reward behavior."""
    var_d = get_ablation_variant("D")
    env_default = DroneNavigation3DEnv(bounds=(30.0, 30.0, 15.0), max_steps=20)
    env_variant_d = DroneNavigation3DEnv(
        bounds=(30.0, 30.0, 15.0),
        max_steps=20,
        **var_d.get_reward_parameters(),
    )

    env_default.reset(seed=123)
    env_variant_d.reset(seed=123)

    actions = [
        np.array([0.2, -0.3, 0.1], dtype=np.float32),
        np.array([0.5, 0.2, -0.1], dtype=np.float32),
        np.array([-0.1, 0.4, 0.3], dtype=np.float32),
    ]

    for act in actions:
        _, rew_def, term_def, trunc_def, _ = env_default.step(act)
        _, rew_d, term_d, trunc_d, _ = env_variant_d.step(act)

        assert rew_def == pytest.approx(rew_d, abs=1e-6)
        assert term_def == term_d
        assert trunc_def == trunc_d

    env_default.close()
    env_variant_d.close()


def test_default_environment_regression() -> None:
    """Requirement 9: Default environment regression - DroneNavigation3DEnv() defaults unchanged."""
    env = DroneNavigation3DEnv()
    assert env.progress_weight == 2.0
    assert env.goal_reward == 100.0
    assert env.step_penalty == -0.05
    assert env.action_penalty_weight == 0.01
    assert env.collision_reward == -100.0

    env.reset(seed=42)
    action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    prev_dist = env._prev_distance_to_goal
    _, rew, _, _, _ = env.step(action)
    curr_dist = env._prev_distance_to_goal
    dist_delta = prev_dist - curr_dist

    expected = 2.0 * dist_delta - 0.05 - 0.01 * float(np.sum(np.square(action)))
    assert rew == pytest.approx(expected, abs=1e-5)
    env.close()


def test_minimal_runner_and_exports(tmp_path: Path) -> None:
    """Requirements 10-14: Execute all 4 variants on minimal budget, verify JSON, CSV, and deterministic seeds."""
    out_dir = tmp_path / "benchmarks"
    json_file = out_dir / "reward_ablation.json"
    csv_file = out_dir / "reward_ablation.csv"

    budget = 128
    episodes = 2
    base_seed = 42

    data = run_reward_ablation_experiment(
        timesteps=budget,
        eval_episodes=episodes,
        seed=base_seed,
        eval_freq=64,
        output_dir=out_dir,
        output_json=json_file,
        output_csv=csv_file,
        variants=["A", "B", "C", "D"],
    )

    # 11. All four variants produce results
    results = data["results"]
    assert len(results) == 4
    var_ids = [r["variant_id"] for r in results]
    assert var_ids == ["A", "B", "C", "D"]

    # 12. Verify JSON export
    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["experiment"] == "reward_function_ablation"
    assert loaded_json["base_seed"] == base_seed
    assert loaded_json["eval_seed"] == base_seed + 1000
    assert len(loaded_json["results"]) == 4

    # 13. Verify CSV export
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 4
    for row in reader:
        assert row["variant"] in [v.name for v in REWARD_ABLATION_VARIANTS.values()]
        assert row["seed"] == str(base_seed)
        assert row["training_timesteps"] == str(budget)
        assert "success_rate" in row
        assert "collision_rate" in row
        assert "timeout_rate" in row
        assert "mean_reward" in row
        assert "mean_path_efficiency" in row
        assert "convergence_speed" in row

    # 14. Verify deterministic configuration/seeds
    for r in results:
        assert r["seed"] == base_seed
        assert r["training_timesteps"] == budget
        assert (
            r["reward_weights"] == REWARD_ABLATION_VARIANTS[r["variant_id"]].get_reward_parameters()
        )
        assert r["mean_path_efficiency"] is not None
        assert 0.0 <= r["mean_path_efficiency"] <= 1.0


def test_convergence_speed_semantics() -> None:
    """Requirement 15: Verify convergence_speed is None when threshold (>0.70) is not reached, and integer when crossed."""

    class DummyAlgorithm:
        def __init__(self, success_pattern: list[bool]) -> None:
            self.pattern = success_pattern
            self.call_count = 0

        def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
            return np.zeros(3, dtype=np.float32), None

    class DummyEvalEnv:
        def __init__(self, dummy_algo: DummyAlgorithm) -> None:
            self.algo = dummy_algo
            self.step_in_ep = 0

        def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
            self.step_in_ep = 0
            return np.zeros(29, dtype=np.float32), {"position": np.zeros(3), "goal": np.ones(3)}

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            self.step_in_ep += 1
            idx = self.algo.call_count % len(self.algo.pattern)
            succ = self.algo.pattern[idx]
            self.algo.call_count += 1
            info = {
                "success": succ,
                "collision": not succ,
                "position": np.ones(3) if succ else np.zeros(3),
                "goal": np.ones(3),
            }
            return np.zeros(29, dtype=np.float32), 100.0 if succ else -50.0, True, False, info

        def close(self) -> None:
            pass

    # Case 1: Threshold never exceeded (e.g. 50% success <= 70%)
    algo_low = DummyAlgorithm([True, False])
    env_low = DummyEvalEnv(algo_low)
    cb_low = ConvergenceEvaluationCallback(
        eval_env=env_low,  # type: ignore[arg-type]
        eval_freq=100,
        eval_episodes=2,
        success_threshold=0.70,
        model=algo_low,  # type: ignore[arg-type]
    )

    cb_low.on_step(100)
    cb_low.on_step(200)
    # Never crossed > 70%: must be None, NOT 0
    assert cb_low.convergence_step is None

    # Case 2: Exactly 70% does NOT trigger (> 0.70 strictly required)
    algo_exact = DummyAlgorithm(
        [True, True, True, True, True, True, True, False, False, False]
    )  # 70%
    env_exact = DummyEvalEnv(algo_exact)
    cb_exact = ConvergenceEvaluationCallback(
        eval_env=env_exact,  # type: ignore[arg-type]
        eval_freq=100,
        eval_episodes=10,
        success_threshold=0.70,
        model=algo_exact,  # type: ignore[arg-type]
    )
    cb_exact.on_step(100)
    assert cb_exact.convergence_step is None

    # Case 3: Threshold exceeded at step 200 (80% > 70%)
    algo_high = DummyAlgorithm([True, True, True, True, False])  # 80%
    env_high = DummyEvalEnv(algo_high)
    cb_high = ConvergenceEvaluationCallback(
        eval_env=env_high,  # type: ignore[arg-type]
        eval_freq=100,
        eval_episodes=5,
        success_threshold=0.70,
        model=algo_high,  # type: ignore[arg-type]
    )
    cb_high.on_step(100)
    assert cb_high.convergence_step == 100

    # Ensure subsequent steps do not overwrite first convergence step
    cb_high.on_step(200)
    assert cb_high.convergence_step == 100


def test_runner_validation_errors() -> None:
    """Requirement 12: Verify experiment runner rejects invalid inputs with ValueError."""
    with pytest.raises(ValueError, match="timesteps must be positive"):
        run_reward_ablation_experiment(timesteps=0)

    with pytest.raises(ValueError, match="eval_episodes must be positive"):
        run_reward_ablation_experiment(eval_episodes=-5)

    with pytest.raises(ValueError, match="seed must be non-negative"):
        run_reward_ablation_experiment(seed=-1)

    with pytest.raises(ValueError, match="Unknown ablation variant"):
        run_reward_ablation_experiment(variants=["NonExistentVariant"])
