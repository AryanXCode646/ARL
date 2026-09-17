"""Unit and integration tests for TrafficSignalEnv, simulation models, and PPO."""

import json
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.config import (
    AlgorithmConfig,
    CurriculumConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.curriculum.presets import get_curriculum_preset
from adaptive_rl.curriculum.trainer import CurriculumTrainer
from adaptive_rl.environments import make_env
from adaptive_rl.environments.traffic.intersection import TrafficSignalEnv
from adaptive_rl.environments.traffic.simulation import (
    Approach,
    ApproachState,
    Phase,
    TrafficIntersection,
)
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.training.trainer import PPOTrainer, TrainingResult


def test_traffic_gymnasium_checker() -> None:
    """Verify TrafficSignalEnv complies 100% with Farama Gymnasium check_env."""
    env = TrafficSignalEnv(max_steps=20)
    check_env(env)
    env.close()


def test_traffic_registry_instantiation() -> None:
    """Verify environment can be created via global make_env registry."""
    env1 = make_env("traffic", max_steps=30)
    env2 = make_env("traffic_signal", max_steps=30)

    assert isinstance(env1, TrafficSignalEnv)
    assert isinstance(env2, TrafficSignalEnv)
    assert env1.max_steps == 30
    assert env2.max_steps == 30

    env1.close()
    env2.close()


def test_traffic_spaces() -> None:
    """Verify observation space is Box(10,) in [0, 1] and action space is Discrete(2)."""
    env = TrafficSignalEnv()
    assert env.action_space.n == 2
    assert env.observation_space.shape == (10,)
    assert env.observation_space.dtype == np.float32
    np.testing.assert_allclose(env.observation_space.low, np.zeros(10, dtype=np.float32))
    np.testing.assert_allclose(env.observation_space.high, np.ones(10, dtype=np.float32))
    env.close()


def test_traffic_deterministic_seeding() -> None:
    """Verify identical seeds produce identical observations and step trajectories."""
    env1 = TrafficSignalEnv(max_steps=20)
    env2 = TrafficSignalEnv(max_steps=20)

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)

    np.testing.assert_array_equal(obs1, obs2)
    assert info1["queues"] == info2["queues"]

    # Execute identical actions
    actions = [0, 0, 1, 1, 0, 1]
    for act in actions:
        o1, r1, t1, tr1, i1 = env1.step(act)
        o2, r2, t2, tr2, i2 = env2.step(act)
        np.testing.assert_array_almost_equal(o1, o2)
        assert r1 == r2
        assert t1 == t2
        assert tr1 == tr2
        assert i1["queues"] == i2["queues"]

    env1.close()
    env2.close()


def test_traffic_approach_state_queuing() -> None:
    """Verify ApproachState queuing, waiting increments, and FIFO discharges."""
    from collections import deque

    state = ApproachState(
        direction=Approach.NORTH,
        arrival_rate=0.5,
        max_capacity=10,
        queue=deque(),
    )
    assert state.queue_length == 0
    assert state.max_wait_time == 0
    assert state.mean_wait_time == 0.0

    # Add 3 vehicles
    added = state.add_arrivals(3)
    assert added == 3
    assert state.queue_length == 3
    assert list(state.queue) == [0, 0, 0]

    # Advance time by 2 steps
    state.step_wait()
    state.step_wait()
    assert list(state.queue) == [2, 2, 2]
    assert state.max_wait_time == 2
    assert state.mean_wait_time == 2.0

    # Add 1 more vehicle (arrives with wait time 0)
    state.add_arrivals(1)
    assert list(state.queue) == [2, 2, 2, 0]

    # Discharge 2 vehicles (FIFO)
    discharged_count, discharged_waits = state.discharge(2)
    assert discharged_count == 2
    assert discharged_waits == [2, 2]
    assert list(state.queue) == [2, 0]
    assert state.queue_length == 2


def test_traffic_intersection_phase_departures() -> None:
    """Verify vehicles only discharge in the direction with the active green signal."""
    intersection = TrafficIntersection(
        arrival_rates=(0.0, 0.0, 0.0, 0.0),  # Zero arrivals to isolate discharge physics
        departure_rate=2,
        max_queue=20,
    )
    # Set initial queues: North=3, South=2, East=4, West=1
    intersection.reset(
        initial_queues={
            Approach.NORTH: 3,
            Approach.SOUTH: 2,
            Approach.EAST: 4,
            Approach.WEST: 1,
        }
    )

    rng = np.random.default_rng(42)

    # Step with action 0 (NORTH_SOUTH)
    telemetry = intersection.step(0, rng)
    assert telemetry.current_phase == Phase.NORTH_SOUTH
    assert telemetry.departures[Approach.NORTH] == 2  # Discharged 2 of 3
    assert telemetry.departures[Approach.SOUTH] == 2  # Discharged 2 of 2
    assert telemetry.departures[Approach.EAST] == 0  # Red
    assert telemetry.departures[Approach.WEST] == 0  # Red

    assert telemetry.queue_lengths[Approach.NORTH] == 1
    assert telemetry.queue_lengths[Approach.SOUTH] == 0
    assert telemetry.queue_lengths[Approach.EAST] == 4
    assert telemetry.queue_lengths[Approach.WEST] == 1

    # Switch to EAST_WEST
    telemetry2 = intersection.step(1, rng)
    assert telemetry2.current_phase == Phase.EAST_WEST
    assert telemetry2.phase_switched is True
    assert telemetry2.departures[Approach.NORTH] == 0  # Red
    assert telemetry2.departures[Approach.SOUTH] == 0  # Red
    assert telemetry2.departures[Approach.EAST] == 2  # Discharged 2 of 4
    assert telemetry2.departures[Approach.WEST] == 1  # Discharged 1 of 1


def test_traffic_signal_switch_penalties() -> None:
    """Verify phase switching and premature switching penalties."""
    env = TrafficSignalEnv(
        min_green_steps=3,
        switch_penalty=1.5,
        premature_switch_penalty=3.0,
        arrival_rates=(0.0, 0.0, 0.0, 0.0),
    )
    env.reset(seed=42)

    # Step 1: Action 0 keeps Phase 0 (no switch penalty)
    obs, r1, term, trunc, info1 = env.step(0)
    assert info1["phase_switched"] is False
    assert info1["premature_switch"] is False

    # Step 2: Premature switch from Phase 0 to Phase 1 at duration=1 (< min_green_steps=3)
    obs, r2, term, trunc, info2 = env.step(1)
    assert info2["phase_switched"] is True
    assert info2["premature_switch"] is True
    # Reward should include both switch_penalty (1.5) and premature_switch_penalty (3.0)
    assert r2 <= -(1.5 + 3.0)

    # Maintain Phase 1 for 3 steps
    env.step(1)
    env.step(1)
    env.step(1)

    # Now switch back to Phase 0 (duration is now >= 3 steps -> regular switch, not premature)
    obs, r3, term, trunc, info3 = env.step(0)
    assert info3["phase_switched"] is True
    assert info3["premature_switch"] is False

    env.close()


def test_traffic_overflow_termination() -> None:
    """Verify terminate_on_overflow terminates when capacity is exceeded."""
    env = TrafficSignalEnv(
        max_queue=5,
        arrival_rates=(10.0, 10.0, 10.0, 10.0),  # Massive arrival rates guaranteed to overflow
        terminate_on_overflow=True,
    )
    env.reset(seed=42)

    terminated = False
    for _ in range(5):
        _, _, terminated, _, info = env.step(0)
        if terminated:
            assert info["overflow"] is True
            break

    assert terminated is True
    env.close()


def test_traffic_truncation_at_max_steps() -> None:
    """Verify episode truncates at max_steps."""
    max_steps = 15
    env = TrafficSignalEnv(max_steps=max_steps, terminate_on_overflow=False)
    env.reset(seed=42)

    for step in range(1, max_steps):
        _, _, term, trunc, _ = env.step(0)
        assert term is False
        assert trunc is False

    # 15th step should truncate
    _, _, term, trunc, info = env.step(0)
    assert trunc is True
    assert term is False
    assert info["step"] == max_steps
    env.close()


def test_traffic_render_modes() -> None:
    """Verify ansi string output and human print rendering."""
    env = TrafficSignalEnv(render_mode="ansi")
    env.reset(seed=42)

    rendered = env.render()
    assert isinstance(rendered, str)
    assert "4-WAY SIGNALIZED INTERSECTION" in rendered
    assert "[G]" in rendered
    assert "[R]" in rendered
    assert "Queues:" in rendered

    # Test human render
    env_human = TrafficSignalEnv(render_mode="human")
    env_human.reset(seed=42)
    env_human.step(0)
    env_human.close()
    env.close()


def test_traffic_curriculum_preset() -> None:
    """Verify traffic curriculum preset creation and progression stages."""
    curr = get_curriculum_preset("traffic")
    assert len(curr.stages) == 4
    assert curr.stages[0].name == "Light Balanced"
    assert curr.stages[1].name == "Moderate Balanced"
    assert curr.stages[2].name == "Arterial Rush Hour"
    assert curr.stages[3].name == "Peak Gridlock Challenge"

    # Also resolve via alias
    curr_alias = get_curriculum_preset("traffic_signal")
    assert len(curr_alias.stages) == 4


def test_traffic_ppo_training(tmp_path: Path) -> None:
    """Integration test: Train PPO on traffic environment for a short duration."""
    exp_cfg = ExperimentConfig(
        name="test_traffic_ppo",
        seed=42,
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=0.0003,
            gamma=0.99,
            batch_size=32,
            parameters={"n_steps": 64, "n_epochs": 2},
        ),
        environment=EnvironmentConfig(
            name="traffic",
            max_steps=20,
            parameters={"arrival_rates": (0.2, 0.2, 0.2, 0.2)},
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

    trainer = PPOTrainer(config=exp_cfg)
    result = trainer.fit()

    assert isinstance(result, TrainingResult)
    assert result.total_timesteps == 128
    assert result.final_model_path.exists()
    assert len(result.checkpoints) >= 1

    # Evaluate trained model
    mean_reward, std_reward = trainer.evaluate(episodes=3)
    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)


def test_traffic_evaluator_benchmarking(tmp_path: Path) -> None:
    """Integration test: Benchmarking traffic environment using Evaluator."""
    env = make_env("traffic", max_steps=25)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = Evaluator(
        algorithm=algo,
        env=env,
    )
    metrics = evaluator.evaluate(num_episodes=5, deterministic=True, base_seed=100)

    assert metrics.episodes == 5
    assert isinstance(metrics.mean_reward, float)
    assert metrics.mean_episode_length == 25.0

    # Save report
    json_path = tmp_path / "eval_report.json"
    evaluator.save_report(metrics, json_path)
    assert json_path.exists()
    report_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert report_data["episodes"] == 5
    env.close()


def test_traffic_curriculum_trainer_workflow(tmp_path: Path) -> None:
    """Integration test: Train with curriculum on traffic environment."""
    exp_cfg = ExperimentConfig(
        name="test_traffic_curriculum",
        seed=42,
        output_dir=tmp_path / "curr_results",
        log_dir=tmp_path / "curr_logs",
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=0.0003,
            gamma=0.99,
            batch_size=32,
            parameters={"n_steps": 64, "n_epochs": 2},
        ),
        environment=EnvironmentConfig(
            name="traffic",
            max_steps=20,
        ),
        curriculum=CurriculumConfig(
            enabled=True,
            preset="traffic",
            eval_window=5,
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

    trainer = CurriculumTrainer(config=exp_cfg)
    result = trainer.fit()

    assert result.total_timesteps == 128
    assert result.final_model_path.exists()
    assert (
        tmp_path / "curr_results" / "curriculum" / "test_traffic_curriculum_curriculum.json"
    ).exists()


def test_traffic_scenario_a_successful_nonzero_traffic() -> None:
    """TEST A: Deterministic non-zero traffic scenario resulting in verified success.

    Proves that with real vehicle arrivals and adequate signal service,
    an episode completes without overflow, discharges vehicles, and reports
    explicit episode success with controlled queues.
    """
    env = TrafficSignalEnv(
        arrival_rates=(0.3, 0.3, 0.2, 0.2),
        departure_rate=3,
        max_queue=25,
        max_steps=10,
        min_green_steps=2,
        terminate_on_overflow=True,
        success_queue_threshold=10,
    )
    obs, info = env.reset(seed=42)
    assert info["had_overflow"] is False
    assert info["step"] == 0

    had_arrivals = False
    for step_idx in range(10):
        # Service NS for first 5 steps, then EW for next 5 steps
        action = 0 if step_idx < 5 else 1
        obs, reward, terminated, truncated, step_info = env.step(action)
        if step_info["step_arrivals"] > 0:
            had_arrivals = True

    assert had_arrivals, "Expected non-zero vehicle arrivals during test run"
    assert terminated is False
    assert truncated is True
    assert step_info["had_overflow"] is False
    assert step_info["total_queue"] <= 10
    assert step_info["success"] is True
    assert step_info["cumulative_departures"] > 0
    assert step_info["mean_wait"] >= 0.0
    env.close()


def test_traffic_scenario_b_overflow_failure() -> None:
    """TEST B: Deterministic scenario with guaranteed queue overflow.

    Proves that queue overflow permanently sets had_overflow=True,
    forces success=False, and immediately triggers termination when
    terminate_on_overflow=True.
    """
    env = TrafficSignalEnv(
        arrival_rates=(15.0, 15.0, 15.0, 15.0),
        departure_rate=1,
        max_queue=5,
        max_steps=10,
        terminate_on_overflow=True,
        overflow_penalty=50.0,
    )
    obs, info = env.reset(seed=123)
    assert info["had_overflow"] is False

    obs, reward, terminated, truncated, step_info = env.step(0)
    assert step_info["overflow"] is True
    assert step_info["had_overflow"] is True
    assert step_info["success"] is False
    assert terminated is True
    assert truncated is False
    assert reward < -40.0  # Includes overflow penalty
    env.close()


def test_traffic_scenario_c_terminal_vs_truncation_semantics() -> None:
    """TEST C: Explicit differentiation between terminated and truncated.

    Case 1: Overflow causes terminated=True, truncated=False.
    Case 2: Completing all steps without overflow causes terminated=False, truncated=True.
    """
    # Case 1: Early termination on overflow
    env_term = TrafficSignalEnv(
        arrival_rates=(20.0, 0.0, 0.0, 0.0),
        departure_rate=1,
        max_queue=4,
        max_steps=50,
        terminate_on_overflow=True,
    )
    env_term.reset(seed=99)
    _, _, terminated, truncated, info_term = env_term.step(0)
    assert terminated is True
    assert truncated is False
    assert info_term["had_overflow"] is True
    assert info_term["success"] is False
    env_term.close()

    # Case 2: Normal truncation at max_steps
    env_trunc = TrafficSignalEnv(
        arrival_rates=(0.0, 0.0, 0.0, 0.0),
        departure_rate=2,
        max_queue=30,
        max_steps=5,
        terminate_on_overflow=True,
    )
    env_trunc.reset(seed=99)
    for _ in range(4):
        _, _, term, trunc, _ = env_trunc.step(0)
        assert not term and not trunc
    _, _, term, trunc, info_trunc = env_trunc.step(0)
    assert term is False
    assert trunc is True
    assert info_trunc["had_overflow"] is False
    assert info_trunc["success"] is True
    env_trunc.close()


def test_traffic_scenario_d_known_wait_values_and_aggregation() -> None:
    """TEST D: Deterministic queue with known wait values and exact aggregation.

    Verifies vehicle delay accumulation per step and approach max_wait calculation.
    """
    env = TrafficSignalEnv(
        arrival_rates=(0.0, 0.0, 0.0, 0.0),
        departure_rate=2,
        max_queue=20,
        max_steps=10,
        min_green_steps=1,
    )
    # Seed 3 vehicles in North and 1 vehicle in East
    env.reset(
        seed=42,
        options={
            "initial_queues": {
                Approach.NORTH: 3,
                Approach.SOUTH: 0,
                Approach.EAST: 1,
                Approach.WEST: 0,
            }
        },
    )

    # Step 1: Green for EW (action=1).
    # East has 1 vehicle, discharges it.
    # North has 3 vehicles, wait increments by 1.
    _, _, _, _, info1 = env.step(1)
    assert info1["step_departures"] == 1
    assert info1["total_queue"] == 3
    # North vehicles waited 1 step
    assert info1["max_wait"] == 1
    assert info1["mean_wait"] == 0.25  # (1 + 0 + 0 + 0) / 4 approaches

    # Step 2: Green for EW (action=1) again.
    # No departures (EW queues empty).
    # North vehicles wait another step (wait = 2).
    _, _, _, _, info2 = env.step(1)
    assert info2["step_departures"] == 0
    assert info2["total_queue"] == 3
    assert info2["max_wait"] == 2
    assert info2["mean_wait"] == 0.5  # (2 + 0 + 0 + 0) / 4 approaches
    env.close()
