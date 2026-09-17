"""Comprehensive unit and integration tests for Drone Disturbances & Constraints.

Covers:
- Gymnasium interface compliance (check_env)
- 33-dimensional observation and continuous 3D acceleration action spaces
- 3D Wind field dynamics, altitude shear, and Ornstein-Uhlenbeck stochastic turbulence
- Quadrotor battery depletion model and power consumption
- Moving dynamic 3D obstacle kinematics and boundary bouncing
- Wind drift physical integration
- Battery exhaustion termination and penalties
- Dynamic and static collision detection via 16-ray 3D LiDAR
- ASCII flight deck dashboard rendering
- Short-horizon PPO and SAC training
- Multi-episode evaluation benchmarking
- Progressive curriculum scheduling and dynamic stage parameter injection
"""

import json
from pathlib import Path

import numpy as np
import pytest
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
from adaptive_rl.curriculum.presets import create_disturbed_drone_curriculum, get_curriculum_preset
from adaptive_rl.curriculum.trainer import CurriculumTrainer
from adaptive_rl.environments import make_env
from adaptive_rl.environments.drone.battery import BatteryModel
from adaptive_rl.environments.drone.disturbed_drone import DroneDisturbance3DEnv
from adaptive_rl.environments.drone.dynamic_obstacles import (
    DynamicObstacleSphere3D,
    generate_dynamic_drone_obstacles,
)
from adaptive_rl.environments.drone.wind import WindField3D
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.training.trainer import PPOTrainer, SACTrainer


def test_disturbed_drone_gymnasium_checker() -> None:
    """Verify DroneDisturbance3DEnv complies 100% with Gymnasium API standards."""
    env = DroneDisturbance3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20)
    check_env(env)
    env.close()


def test_disturbed_drone_registry_instantiation() -> None:
    """Verify environment can be instantiated via make_env under aliases."""
    env1 = make_env("drone_disturbed", max_steps=30)
    env2 = make_env("drone_constrained", max_steps=30)

    assert isinstance(env1, DroneDisturbance3DEnv)
    assert isinstance(env2, DroneDisturbance3DEnv)

    env1.close()
    env2.close()


def test_disturbed_drone_spaces() -> None:
    """Verify 3D action space and 33-dimensional observation space."""
    env = DroneDisturbance3DEnv(bounds=(50.0, 50.0, 25.0), num_lidar_rays=16)

    # Action space: 3D acceleration commands in [-1, 1]
    assert env.action_space.shape == (3,)
    assert env.action_space.dtype == np.float32
    np.testing.assert_allclose(env.action_space.low, np.array([-1.0, -1.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(env.action_space.high, np.array([1.0, 1.0, 1.0], dtype=np.float32))

    # Observation space: 13 (base) + 16 (lidar) + 1 (battery) + 3 (wind) = 33
    assert env.observation_space.shape == (33,)
    assert env.observation_space.dtype == np.float32
    np.testing.assert_allclose(env.observation_space.low, -np.ones(33, dtype=np.float32))
    np.testing.assert_allclose(env.observation_space.high, np.ones(33, dtype=np.float32))
    env.close()


def test_wind_model_and_gust_turbulence() -> None:
    """Verify WindField3D steady vectors, altitude shear, and Ornstein-Uhlenbeck gusts."""
    wind = WindField3D(
        steady_wind=(2.0, 1.0, 0.0),
        gust_theta=0.2,
        gust_sigma=0.5,
        altitude_shear=0.05,
        dt=0.1,
    )

    rng = np.random.default_rng(42)
    # At ground altitude z=0, steady wind is unaffected by shear
    ground_wind = wind.get_wind(np.array([10.0, 10.0, 0.0]))
    np.testing.assert_allclose(ground_wind.steady, np.array([2.0, 1.0, 0.0]))
    assert ground_wind.speed == pytest.approx(np.sqrt(5.0))

    # At altitude z=20, shear factor is 1 + 0.05 * 20 = 2.0
    high_wind = wind.get_wind(np.array([10.0, 10.0, 20.0]))
    np.testing.assert_allclose(high_wind.steady, np.array([4.0, 2.0, 0.0]))

    # Step gust turbulence
    prev_gust = wind._current_gust.copy()
    for _ in range(10):
        new_gust = wind.step_gust(rng)
        assert new_gust.shape == (3,)
        assert np.linalg.norm(new_gust) <= wind.max_gust

    assert not np.array_equal(prev_gust, wind._current_gust)

    # Test parameter validation
    with pytest.raises(ValueError, match="gust_theta cannot be negative"):
        WindField3D(gust_theta=-0.1)
    with pytest.raises(ValueError, match="gust_sigma cannot be negative"):
        WindField3D(gust_sigma=-0.1)
    with pytest.raises(ValueError, match="dt must be positive"):
        WindField3D(dt=-0.1)


def test_battery_model_and_depletion() -> None:
    """Verify BatteryModel power draw calculation and depletion state."""
    battery = BatteryModel(
        capacity=10.0,
        base_power=0.1,
        thrust_coefficient=0.2,
        speed_coefficient=0.1,
    )

    assert battery.remaining_energy == 10.0
    assert battery.state_of_charge == 1.0
    assert not battery.is_depleted

    acc = np.array([1.0, 0.0, 0.0])  # norm_sq = 1.0
    vel = np.array([0.0, 2.0, 0.0])  # norm_sq = 4.0
    # Expected power: 0.1 + 0.2*(1.0) + 0.1*(4.0) = 0.1 + 0.2 + 0.4 = 0.7 units/s
    # Energy step (dt=0.5): 0.7 * 0.5 = 0.35
    telem = battery.step(acc, vel, dt=0.5)

    assert telem.power_consumed == pytest.approx(0.7)
    assert telem.energy_step == pytest.approx(0.35)
    assert battery.remaining_energy == pytest.approx(10.0 - 0.35)
    assert battery.state_of_charge == pytest.approx((10.0 - 0.35) / 10.0)

    # Force complete depletion
    for _ in range(50):
        battery.step(np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0]), dt=1.0)
    assert battery.remaining_energy == 0.0
    assert battery.is_depleted
    assert battery.state_of_charge == 0.0

    # Test reset
    battery.reset(0.5)
    assert battery.remaining_energy == 5.0
    assert battery.state_of_charge == 0.5

    # Parameter validation
    with pytest.raises(ValueError, match="capacity must be positive"):
        BatteryModel(capacity=-5.0)
    with pytest.raises(ValueError, match="base_power cannot be negative"):
        BatteryModel(base_power=-1.0)


def test_dynamic_obstacles_movement_and_reflection() -> None:
    """Verify DynamicObstacleSphere3D trajectory stepping and bounding reflection."""
    pos = np.array([19.0, 10.0, 10.0])
    vel = np.array([2.0, 0.0, 0.0])
    bounds = (20.0, 20.0, 20.0)
    obs = DynamicObstacleSphere3D(
        center=pos,
        velocity=vel,
        radius=1.5,
        bounds=bounds,
    )

    assert obs.radius == 1.5
    static_sphere = obs.to_static_sphere()
    np.testing.assert_allclose(static_sphere.center, pos)

    # Step: position will reach 19.0 + 2.0 * 0.5 = 20.0, hitting or exceeding boundary 20.0 - 1.5 = 18.5
    obs.step(dt=0.5)
    # Velocity x must have reflected (negated)
    assert obs.velocity[0] < 0.0
    assert obs.center[0] <= bounds[0] - obs.radius

    # Generator test
    rng = np.random.default_rng(123)
    dyn_list = generate_dynamic_drone_obstacles(
        bounds=(40.0, 40.0, 20.0),
        start_pos=np.array([5.0, 5.0, 5.0]),
        goal_pos=np.array([35.0, 35.0, 15.0]),
        num_obstacles=4,
        obstacle_radius=1.5,
        speed=2.0,
        rng=rng,
    )
    assert len(dyn_list) == 4
    for o in dyn_list:
        assert float(np.linalg.norm(o.center - np.array([5.0, 5.0, 5.0]))) >= 1.5 + 3.0


def test_disturbed_drone_wind_drift() -> None:
    """Verify wind vector forces drift when zero control action is applied."""
    env = DroneDisturbance3DEnv(
        bounds=(50.0, 50.0, 30.0),
        steady_wind=(3.0, 0.0, 0.0),
        gust_sigma=0.0,  # disable gust noise for deterministic drift check
        linear_damping=0.2,
        dt=0.1,
    )
    obs, info = env.reset(seed=42)
    start_x = info["position"][0]

    # Command neutral zero acceleration for 5 steps
    zero_act = np.zeros(3, dtype=np.float32)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(zero_act)

    # Effective acceleration has pushed the drone along +X due to positive crosswind
    end_x = info["position"][0]
    assert end_x > start_x
    assert info["velocity"][0] > 0.0
    env.close()


def test_disturbed_drone_battery_depletion_termination() -> None:
    """Verify battery exhaustion triggers episode termination and penalty."""
    env = DroneDisturbance3DEnv(
        bounds=(50.0, 50.0, 30.0),
        battery_capacity=0.5,  # Very small battery
        battery_base_power=1.0,
        battery_exhaustion_penalty=-50.0,
        terminate_on_exhaustion=True,
    )
    obs, info = env.reset(seed=42)

    terminated = False
    step_count = 0
    full_throttle = np.ones(3, dtype=np.float32)

    while not terminated and step_count < 20:
        obs, reward, terminated, truncated, info = env.step(full_throttle)
        step_count += 1

    assert terminated
    assert info["battery_exhausted"]
    assert reward == -50.0
    env.close()


def test_disturbed_drone_collision_detection() -> None:
    """Verify collision detection handles dynamic obstacles properly."""
    env = DroneDisturbance3DEnv(
        bounds=(30.0, 30.0, 30.0),
        start_pos=(5.0, 5.0, 5.0),
        goal_pos=(25.0, 25.0, 25.0),
        num_obstacles=0,
        num_dynamic_obstacles=1,
        collision_radius=1.0,
        terminate_on_collision=True,
    )
    env.reset(seed=42)
    # Manually place dynamic obstacle directly at drone position
    env._dynamic_obstacles[0].center = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    env._dynamic_obstacles[0].radius = 1.0

    # Step in place
    obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert info["collision"]
    assert info["collision_type"] == "dynamic_obstacle"
    env.close()


def test_disturbed_drone_render_ascii() -> None:
    """Verify ASCII flight deck renders wind vector and battery gauge."""
    env = DroneDisturbance3DEnv(
        bounds=(30.0, 30.0, 15.0),
        steady_wind=(2.5, -1.0, 0.5),
        battery_capacity=100.0,
    )
    env.reset(seed=42)
    rendered = env.render()
    assert rendered is not None
    assert "AUTONOMOUS 3D DRONE FLIGHT DECK (DISTURBED / CONSTRAINED)" in rendered
    assert "Ambient Wind Field:" in rendered
    assert "Battery Reserve:" in rendered
    assert "%" in rendered
    env.close()


def test_disturbed_drone_curriculum_preset() -> None:
    """Verify disturbed drone curriculum stages and progressive parameter injection."""
    direct_curr = create_disturbed_drone_curriculum(eval_window=10)
    curriculum = get_curriculum_preset("drone_disturbed", eval_window=10)
    assert len(direct_curr.stages) == len(curriculum.stages)
    assert len(curriculum.stages) == 4
    assert curriculum.stages[0].name == "Calm Skies"
    assert curriculum.stages[1].name == "Crosswind Drift"
    assert curriculum.stages[2].name == "Gusting Turbulence"
    assert curriculum.stages[3].name == "Storm Hazard Challenge"

    env = make_env("drone_disturbed")
    from adaptive_rl.curriculum.wrapper import CurriculumEnvWrapper

    wrapper = CurriculumEnvWrapper(env=env, curriculum=curriculum)
    obs, info = wrapper.reset(seed=42)

    assert info["curriculum_stage_name"] == "Calm Skies"
    assert wrapper.unwrapped.battery_capacity == 200.0
    assert not wrapper.unwrapped.wind_enabled
    wrapper.close()


def test_disturbed_drone_ppo_training_pipeline(tmp_path: Path) -> None:
    """Verify end-to-end PPO training on disturbed drone environment."""
    config = ExperimentConfig(
        name="test_disturbed_drone_ppo",
        seed=42,
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=0.0003,
            gamma=0.99,
            batch_size=64,
            parameters={"n_steps": 128},
        ),
        environment=EnvironmentConfig(
            name="drone_disturbed",
            max_steps=50,
            parameters={
                "bounds": [30.0, 30.0, 15.0],
                "num_obstacles": 2,
                "num_dynamic_obstacles": 1,
            },
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

    trainer = PPOTrainer(config=config)
    report = trainer.fit()

    assert report.total_timesteps == 128
    assert report.final_model_path.exists()
    assert (tmp_path / "results" / "checkpoints" / "test_disturbed_drone_ppo").exists()


def test_disturbed_drone_sac_training_pipeline(tmp_path: Path) -> None:
    """Verify end-to-end SAC training on disturbed drone environment."""
    config = ExperimentConfig(
        name="test_disturbed_drone_sac",
        seed=42,
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
        algorithm=AlgorithmConfig(
            name="sac",
            learning_rate=0.0003,
            gamma=0.99,
            batch_size=32,
            parameters={"learning_starts": 10, "buffer_size": 1000},
        ),
        environment=EnvironmentConfig(
            name="drone_disturbed",
            max_steps=20,
            parameters={
                "bounds": [30.0, 30.0, 15.0],
                "num_obstacles": 2,
                "num_dynamic_obstacles": 1,
            },
        ),
        training=TrainingConfig(
            total_timesteps=64,
            checkpoint_freq=32,
            log_interval=1,
        ),
        evaluation=EvaluationConfig(
            eval_episodes=2,
            deterministic=True,
        ),
    )

    trainer = SACTrainer(config=config)
    report = trainer.fit()

    assert report.total_timesteps == 64
    assert report.final_model_path.exists()


def test_disturbed_drone_evaluation(tmp_path: Path) -> None:
    """Verify Evaluator generates multi-episode benchmarks and telemetry for disturbed drone."""
    env = make_env("drone_disturbed", max_steps=25, num_obstacles=2, num_dynamic_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)

    evaluator = Evaluator(algorithm=algo, env=env)
    metrics = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=100)

    assert metrics.episodes == 3
    assert isinstance(metrics.mean_reward, float)

    report_path = tmp_path / "eval_out" / "evaluation_report.json"
    evaluator.save_report(metrics, report_path)
    assert report_path.exists()

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["episodes"] == 3
    env.close()


def test_disturbed_drone_curriculum_trainer_pipeline(tmp_path: Path) -> None:
    """Verify CurriculumTrainer executes stages on disturbed drone environment."""
    config = ExperimentConfig(
        name="test_curriculum_drone_disturbed",
        seed=42,
        output_dir=tmp_path / "curriculum_results",
        log_dir=tmp_path / "curriculum_logs",
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=0.0003,
            gamma=0.99,
            batch_size=64,
            parameters={"n_steps": 128},
        ),
        environment=EnvironmentConfig(
            name="drone_disturbed",
            max_steps=20,
        ),
        curriculum=CurriculumConfig(
            enabled=True,
            preset="drone_disturbed",
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

    trainer = CurriculumTrainer(config=config)
    report = trainer.fit()

    assert report.total_timesteps == 128
    assert report.final_model_path.exists()
    assert trainer.curriculum.current_stage.stage_id >= 0
