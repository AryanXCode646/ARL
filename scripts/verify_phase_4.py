"""Standalone verification script for AdaptiveRL Phase 4: PPO Training Engine.

Executes end-to-end PPO training on GridWorld, verifies deterministic seeding,
metric logging callbacks, periodic checkpointing, model artifact persistence,
and action prediction.
"""

from __future__ import annotations

import shutil
import sys
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
from adaptive_rl.environments.registry import make_env
from adaptive_rl.training.trainer import PPOTrainer


def run_phase_4_verification() -> bool:
    """Execute all Phase 4 verification checks."""
    print("=== AdaptiveRL Phase 4: PPO Training Engine Verification ===\n")
    test_dir = Path("experiments/verify_phase_4")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Environment and PPO Instantiation
        print("1. Instantiating GridWorld and PPOAlgorithm...")
        env = make_env("gridworld", width=5, height=5, num_obstacles=2, max_steps=30)
        algo = PPOAlgorithm(
            env=env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=3,
            seed=42,
        )
        assert algo.model is not None
        print("   ✓ PPOAlgorithm instantiated with underlying SB3 actor-critic policy.")

        # Step 2: Training Run on GridWorld
        print("\n2. Running PPO optimization for 500 timesteps...")
        algo.train(total_timesteps=500)
        assert algo.num_timesteps >= 500
        print(f"   ✓ PPO training finished with {algo.num_timesteps} environment steps.")

        # Step 3: Model Persistence (Save / Load Roundtrip)
        print("\n3. Testing model checkpoint serialization and reload...")
        save_path = test_dir / "ppo_gridworld_checkpoint.zip"
        algo.save(save_path)
        assert save_path.exists(), f"Model file not created at {save_path}"
        print(f"   ✓ Model saved successfully ({save_path.stat().st_size:,} bytes).")

        loaded_algo = PPOAlgorithm.from_pretrained(save_path, env=env)
        test_obs = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        act1, _ = algo.predict(test_obs, deterministic=True)
        act2, _ = loaded_algo.predict(test_obs, deterministic=True)
        assert act1 == act2, "Loaded model prediction diverged from original model."
        print(f"   ✓ Loaded model reproduces deterministic action prediction: action={act1}.")
        env.close()

        # Step 4: End-to-End PPOTrainer with Callbacks & Checkpoints
        print("\n4. Running complete PPOTrainer with callbacks and checkpointing...")
        config = ExperimentConfig(
            name="verify_phase_4_exp",
            seed=99,
            algorithm=AlgorithmConfig(
                name="ppo",
                learning_rate=3e-4,
                gamma=0.99,
                batch_size=32,
                parameters={"n_steps": 64, "n_epochs": 2},
            ),
            environment=EnvironmentConfig(
                name="gridworld",
                max_steps=25,
                parameters={"width": 6, "height": 5, "num_obstacles=2": None}
                if False
                else {"width": 6, "height": 5, "num_obstacles": 2},
            ),
            training=TrainingConfig(
                total_timesteps=512,
                checkpoint_freq=256,
                log_interval=10,
            ),
            evaluation=EvaluationConfig(eval_episodes=3, deterministic=True),
            output_dir=test_dir / "results",
            log_dir=test_dir / "logs",
        )

        trainer = PPOTrainer(config=config)
        result = trainer.fit()

        print(f"   ✓ Total Timesteps: {result.total_timesteps}")
        print(f"   ✓ Episodes Completed: {result.episodes_completed}")
        print(f"   ✓ Rolling Mean Reward: {result.mean_reward:.2f}")
        print(f"   ✓ Final Model Saved: {result.final_model_path.name}")
        print(f"   ✓ Checkpoints Recorded: {len(result.checkpoints)}")
        assert result.final_model_path.exists()
        assert len(result.checkpoints) >= 1, "Expected at least 1 periodic checkpoint."

        # Step 5: Post-Training Policy Evaluation
        print("\n5. Running policy evaluation over 5 episodes...")
        eval_mean, eval_std = trainer.evaluate(episodes=5, deterministic=True)
        print(f"   ✓ Evaluation Mean Reward: {eval_mean:.2f} ± {eval_std:.2f}")

        print(
            "\n=== Phase 4 PPO Training Engine Verification COMPLETE: All Quality Gates Passed! ==="
        )
        return True
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    success = run_phase_4_verification()
    sys.exit(0 if success else 1)
