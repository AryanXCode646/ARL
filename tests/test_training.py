"""Tests for PPO training pipeline and model persistence."""

from pathlib import Path

import numpy as np

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.training.trainer import PPOTrainer


def test_ppo_algorithm_init_and_train() -> None:
    """Verify PPO algorithm initializes and completes a short rollout."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)

    assert algo.model is not None
    algo.train(total_timesteps=64)
    assert algo.num_timesteps >= 64

    obs, _ = env.reset(seed=42)
    action, _ = algo.predict(obs, deterministic=True)
    assert action.shape == (3,)
    assert not np.isnan(action).any()
    assert not np.isinf(action).any()
    env.close()


def test_ppo_model_save_and_load(tmp_path: Path) -> None:
    """Verify trained PPO weights can be saved to disk and loaded back."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    algo.train(total_timesteps=64)

    save_path = tmp_path / "saved_ppo.zip"
    algo.save(save_path)
    assert save_path.exists()

    loaded_algo = PPOAlgorithm.from_pretrained(save_path, env=env)
    obs, _ = env.reset(seed=10)
    action1, _ = algo.predict(obs, deterministic=True)
    action2, _ = loaded_algo.predict(obs, deterministic=True)

    np.testing.assert_allclose(action1, action2, rtol=1e-5)
    env.close()


def test_ppo_trainer_full_lifecycle(tmp_path: Path) -> None:
    """Verify PPOTrainer runs fit, creates models and metadata, and exits successfully."""
    config = ExperimentConfig(
        name="test_lifecycle",
        seed=100,
        output_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=3e-4,
            parameters={"n_steps": 64, "batch_size": 32, "n_epochs": 1},
        ),
        environment=EnvironmentConfig(
            name="drone",
            max_steps=25,
            parameters={"bounds": [20.0, 20.0, 10.0], "num_obstacles": 1},
        ),
        training=TrainingConfig(
            total_timesteps=128,
            checkpoint_freq=64,
            log_interval=1,
        ),
        evaluation=EvaluationConfig(eval_episodes=2),
    )

    trainer = PPOTrainer(config=config)
    result = trainer.fit()

    assert result.total_timesteps == 128
    assert result.final_model_path.exists()
    assert result.metadata_path is not None and result.metadata_path.exists()
    assert isinstance(result.mean_reward, float)
    trainer.close()
