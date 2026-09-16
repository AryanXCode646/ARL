"""Unit and integration tests for Continuous 2D Navigation, LiDAR sensors, and SAC."""

from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.environments import make_env
from adaptive_rl.environments.navigation.generator import (
    compute_lidar_readings,
    generate_navigation_obstacles,
    ray_cast_arena_boundaries,
    ray_cast_circle,
)
from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.training.trainer import SACTrainer, get_trainer


def test_navigation_gymnasium_compliance() -> None:
    """Verify ContinuousNavigation2DEnv passes Farama Gymnasium check_env."""
    env = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, max_steps=20)
    check_env(env)
    env.close()


def test_navigation_spaces() -> None:
    """Verify observation and action spaces."""
    env = ContinuousNavigation2DEnv(
        arena_width=20.0,
        arena_height=20.0,
        num_lidar_rays=8,
    )
    from gymnasium import spaces

    assert isinstance(env.action_space, spaces.Box)
    assert env.action_space.shape == (2,)
    np.testing.assert_allclose(env.action_space.low, np.array([-1.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(env.action_space.high, np.array([1.0, 1.0], dtype=np.float32))

    # Obs shape = 2 (norm pos) + 2 (norm goal) + 2 (rel goal) + 8 (lidar) = 14
    assert env.observation_space.shape == (14,)
    assert env.observation_space.dtype == np.float32
    env.close()


def test_navigation_deterministic_seeding() -> None:
    """Verify identical seeds produce identical obstacle layouts and observations."""
    env1 = ContinuousNavigation2DEnv(arena_width=20.0, arena_height=20.0, num_obstacles=4)
    obs1, info1 = env1.reset(seed=42)

    env2 = ContinuousNavigation2DEnv(arena_width=20.0, arena_height=20.0, num_obstacles=4)
    obs2, info2 = env2.reset(seed=42)

    np.testing.assert_allclose(obs1, obs2)
    assert info1["obstacles"] == info2["obstacles"]

    # Different seed yields different obstacle placement
    _, info3 = env2.reset(seed=999)
    assert info1["obstacles"] != info3["obstacles"]

    env1.close()
    env2.close()


def test_ray_cast_circle() -> None:
    """Test ray-circle analytical intersection mathematics."""
    origin = np.array([0.0, 0.0], dtype=np.float32)
    direction_x = np.array([1.0, 0.0], dtype=np.float32)
    center = np.array([5.0, 0.0], dtype=np.float32)
    radius = 1.0

    # Ray points directly at circle center, closest surface at x = 4.0
    dist = ray_cast_circle(origin, direction_x, center, radius, max_range=10.0)
    assert pytest.approx(dist, rel=1e-4) == 4.0

    # Ray pointing opposite direction should return max_range
    direction_neg_x = np.array([-1.0, 0.0], dtype=np.float32)
    dist_miss = ray_cast_circle(origin, direction_neg_x, center, radius, max_range=10.0)
    assert dist_miss == 10.0

    # Ray origin inside circle returns 0.0
    inside_origin = np.array([5.0, 0.5], dtype=np.float32)
    dist_inside = ray_cast_circle(inside_origin, direction_x, center, radius, max_range=10.0)
    assert dist_inside == 0.0


def test_ray_cast_arena_boundaries() -> None:
    """Test ray casting against rectangular arena boundaries."""
    origin = np.array([5.0, 5.0], dtype=np.float32)
    width, height = 20.0, 20.0

    # Ray pointing east (+x): dist to x=20 is 15.0
    dist_east = ray_cast_arena_boundaries(origin, np.array([1.0, 0.0]), width, height, 30.0)
    assert pytest.approx(dist_east, rel=1e-4) == 15.0

    # Ray pointing west (-x): dist to x=0 is 5.0
    dist_west = ray_cast_arena_boundaries(origin, np.array([-1.0, 0.0]), width, height, 30.0)
    assert pytest.approx(dist_west, rel=1e-4) == 5.0

    # Ray pointing north (+y): dist to y=20 is 15.0
    dist_north = ray_cast_arena_boundaries(origin, np.array([0.0, 1.0]), width, height, 30.0)
    assert pytest.approx(dist_north, rel=1e-4) == 15.0

    # Ray pointing south (-y): dist to y=0 is 5.0
    dist_south = ray_cast_arena_boundaries(origin, np.array([0.0, -1.0]), width, height, 30.0)
    assert pytest.approx(dist_south, rel=1e-4) == 5.0


def test_compute_lidar_readings() -> None:
    """Test multi-directional LiDAR distance readings."""
    agent_pos = np.array([10.0, 10.0], dtype=np.float32)
    obstacles = [(14.0, 10.0, 1.0)]  # Obstacle along 0 rad ray (east)
    readings = compute_lidar_readings(
        agent_pos=agent_pos,
        obstacles=obstacles,
        arena_width=20.0,
        arena_height=20.0,
        num_rays=8,
        max_range=10.0,
    )
    assert len(readings) == 8
    assert all(0.0 <= r <= 1.0 for r in readings)
    # 0 rad ray reaches obstacle surface at dist = 3.0 -> normalized = 0.3
    assert pytest.approx(readings[0], abs=0.01) == 0.3


def test_generate_navigation_obstacles_clearance() -> None:
    """Verify generated obstacles maintain clearance around start and goal."""
    rng = np.random.default_rng(123)
    start = np.array([2.0, 2.0], dtype=np.float32)
    goal = np.array([18.0, 18.0], dtype=np.float32)
    radius = 1.0
    clearance = 2.0

    obstacles = generate_navigation_obstacles(
        width=20.0,
        height=20.0,
        start_pos=start,
        goal_pos=goal,
        num_obstacles=6,
        obstacle_radius=radius,
        rng=rng,
        clearance_margin=clearance,
    )

    assert len(obstacles) == 6
    for ox, oy, r in obstacles:
        pos = np.array([ox, oy], dtype=np.float32)
        dist_start = float(np.linalg.norm(pos - start))
        dist_goal = float(np.linalg.norm(pos - goal))
        assert dist_start >= (radius + clearance)
        assert dist_goal >= (radius + clearance)


def test_navigation_continuous_movement() -> None:
    """Verify kinematic translation updates position proportionally to action."""
    env = ContinuousNavigation2DEnv(
        arena_width=20.0,
        arena_height=20.0,
        start_pos=(5.0, 5.0),
        goal_pos=(15.0, 15.0),
        max_speed=2.0,
        dt=0.5,
        num_obstacles=0,
    )
    env.reset()

    # Move right (+x): action [1.0, 0.0] -> delta = [1.0 * 2.0 * 0.5, 0.0] = [1.0, 0.0]
    obs, reward, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(info["agent_pos"], [6.0, 5.0], atol=1e-5)
    assert not term
    assert not trunc
    env.close()


def test_navigation_boundary_collision() -> None:
    """Verify boundary collisions trigger collision reward and termination."""
    env = ContinuousNavigation2DEnv(
        arena_width=10.0,
        arena_height=10.0,
        start_pos=(0.5, 5.0),
        goal_pos=(8.0, 5.0),
        agent_radius=0.4,
        max_speed=2.0,
        dt=1.0,
        terminate_on_collision=True,
        collision_reward=-100.0,
        num_obstacles=0,
    )
    env.reset()

    # Move left into wall x=0
    obs, reward, term, trunc, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert info["collision"] is True
    assert reward == -100.0
    assert term is True
    env.close()


def test_navigation_obstacle_collision() -> None:
    """Verify obstacle collisions trigger collision reward and termination."""
    env = ContinuousNavigation2DEnv(
        arena_width=20.0,
        arena_height=20.0,
        start_pos=(5.0, 5.0),
        goal_pos=(18.0, 18.0),
        agent_radius=0.3,
        fixed_obstacles=[(6.0, 5.0, 0.8)],
        max_speed=2.0,
        dt=0.5,
        terminate_on_collision=True,
    )
    env.reset()

    # Move right into obstacle at (6.0, 5.0)
    obs, reward, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert info["collision"] is True
    assert reward == -100.0
    assert term is True
    env.close()


def test_navigation_goal_reached() -> None:
    """Verify reaching goal gives +100 reward and terminates."""
    env = ContinuousNavigation2DEnv(
        arena_width=20.0,
        arena_height=20.0,
        start_pos=(9.5, 10.0),
        goal_pos=(10.0, 10.0),
        goal_radius=1.0,
        goal_reward=100.0,
        num_obstacles=0,
    )
    env.reset()

    obs, reward, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert info["success"] is True
    assert reward == 100.0
    assert term is True
    env.close()


def test_navigation_truncation_max_steps() -> None:
    """Verify episode truncates when reaching max_steps without goal or collision."""
    env = ContinuousNavigation2DEnv(
        arena_width=20.0,
        arena_height=20.0,
        start_pos=(10.0, 10.0),
        goal_pos=(18.0, 18.0),
        max_steps=5,
        num_obstacles=0,
    )
    env.reset()

    for s in range(4):
        obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert not term
        assert not trunc

    obs, reward, term, trunc, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert not term
    assert trunc is True
    env.close()


def test_navigation_ascii_render() -> None:
    """Verify render produces textual map containing agent, start, goal, and obstacles."""
    env = ContinuousNavigation2DEnv(
        arena_width=20.0,
        arena_height=20.0,
        start_pos=(2.0, 2.0),
        goal_pos=(18.0, 18.0),
        fixed_obstacles=[(10.0, 10.0, 2.0)],
        render_mode="ansi",
    )
    env.reset()
    rendered = env.render()
    assert isinstance(rendered, str)
    assert "+" in rendered
    assert "A" in rendered
    assert "G" in rendered
    assert "#" in rendered
    assert "Dist to Goal" in rendered
    env.close()


def test_navigation_registry_make() -> None:
    """Verify environment can be created through make_env factory."""
    env1 = make_env("navigation")
    assert isinstance(env1, ContinuousNavigation2DEnv)
    env1.close()

    env2 = make_env("navigation_2d")
    assert isinstance(env2, ContinuousNavigation2DEnv)
    env2.close()


def test_sac_algorithm_train_and_predict(tmp_path: Path) -> None:
    """Verify SACAlgorithm wrapper initializes, trains, predicts, and serializes."""
    env = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, max_steps=20)
    algo = SACAlgorithm(
        env=env,
        learning_rate=1e-3,
        buffer_size=1000,
        learning_starts=10,
        batch_size=32,
        seed=42,
    )

    # Train short duration
    algo.train(total_timesteps=64)
    assert algo.num_timesteps >= 64

    # Predict
    obs, _ = env.reset(seed=42)
    action, _ = algo.predict(obs, deterministic=True)
    assert action.shape == (2,)
    assert -1.0 <= action[0] <= 1.0
    assert -1.0 <= action[1] <= 1.0

    # Save and load
    model_path = tmp_path / "sac_nav.zip"
    algo.save(model_path)
    assert model_path.exists()

    loaded_algo = SACAlgorithm.from_pretrained(model_path, env=env)
    pred_action, _ = loaded_algo.predict(obs, deterministic=True)
    np.testing.assert_allclose(action, pred_action, atol=1e-4)

    env.close()


def test_ppo_algorithm_continuous_navigation() -> None:
    """Verify PPO algorithm trains on continuous navigation environment."""
    env = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, max_steps=20)
    algo = PPOAlgorithm(
        env=env,
        n_steps=32,
        batch_size=16,
        seed=42,
    )
    algo.train(total_timesteps=64)
    obs, _ = env.reset(seed=42)
    action, _ = algo.predict(obs, deterministic=True)
    assert action.shape == (2,)
    env.close()


def test_sac_trainer_pipeline(tmp_path: Path) -> None:
    """Verify SACTrainer executes end-to-end training and checkpointing."""
    config = ExperimentConfig(
        name="test_sac_pipeline",
        seed=42,
        algorithm=AlgorithmConfig(
            name="sac",
            learning_rate=0.001,
            gamma=0.99,
            batch_size=32,
            parameters={
                "buffer_size": 1000,
                "learning_starts": 20,
            },
        ),
        environment=EnvironmentConfig(
            name="navigation",
            max_steps=25,
            parameters={"arena_width": 10.0, "arena_height": 10.0, "num_obstacles": 2},
        ),
        training=TrainingConfig(
            total_timesteps=50,
            checkpoint_freq=25,
            log_interval=5,
        ),
        evaluation=EvaluationConfig(
            eval_episodes=2,
            deterministic=True,
        ),
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
    )

    trainer = get_trainer(config=config)
    assert isinstance(trainer, SACTrainer)

    result = trainer.fit()
    assert result.total_timesteps == 50
    assert result.final_model_path.exists()
    assert len(result.checkpoints) >= 1

    # Evaluate
    mean_rew, std_rew = trainer.evaluate(episodes=2)
    assert isinstance(mean_rew, float)


def test_evaluator_with_navigation_policy() -> None:
    """Verify Evaluator benchmark engine with ContinuousNavigation2DEnv and trained policy."""
    env = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, max_steps=15)
    algo = PPOAlgorithm(env=env, n_steps=32, batch_size=16, seed=42)
    algo.train(total_timesteps=32)

    evaluator = Evaluator(algorithm=algo, env=env)
    metrics = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=100)

    assert metrics.episodes == 3
    assert len(metrics.additional_metrics["all_rewards"]) == 3
    assert 0.0 <= metrics.success_rate <= 1.0
    assert 0.0 <= metrics.collision_rate <= 1.0
    env.close()
