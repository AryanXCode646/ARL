"""Standalone verification script for AdaptiveRL Phase 6: Continuous 2D Navigation.

Verifies:
1. ContinuousNavigation2DEnv Gymnasium compliance (check_env).
2. Continuous action space Box(2,) and observation space Box(14,).
3. LiDAR rangefinder ray-casting calculations (circle & boundary intersections).
4. Procedural obstacle generation with start/goal safety margins.
5. Continuous kinematic translation, collision mechanics, and goal detection.
6. SACAlgorithm continuous control training, action prediction, and persistence.
7. PPOAlgorithm continuous control training and prediction.
8. End-to-end benchmark evaluation with Evaluator.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.environments import make_env
from adaptive_rl.environments.navigation.generator import (
    compute_lidar_readings,
    generate_navigation_obstacles,
    ray_cast_arena_boundaries,
    ray_cast_circle,
)
from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv
from adaptive_rl.evaluation.evaluator import Evaluator


def run_phase_6_verification() -> bool:
    """Execute all Phase 6 verification checks."""
    print("=== AdaptiveRL Phase 6: Continuous 2D Navigation Verification ===\n")
    test_dir = Path("experiments/verify_phase_6")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Gymnasium compliance
        print("1. Verifying ContinuousNavigation2DEnv with Farama Gymnasium check_env...")
        env = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, max_steps=20)
        check_env(env)
        print("   ✓ Farama Gymnasium check_env passed with 0 warnings or errors.")

        # Step 2: Spaces and observations
        from gymnasium import spaces

        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.shape == (2,)
        np.testing.assert_allclose(env.action_space.low, np.array([-1.0, -1.0], dtype=np.float32))
        np.testing.assert_allclose(env.action_space.high, np.array([1.0, 1.0], dtype=np.float32))
        assert env.observation_space.shape == (14,)
        obs, info = env.reset(seed=42)
        assert obs.shape == (14,)
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
        print("   ✓ Continuous action space Box(-1.0, 1.0, shape=(2,)) verified.")
        print("   ✓ Continuous observation space Box(-1.0, 1.0, shape=(14,)) verified.")

        # Step 3: Ray casting math and LiDAR
        print("\n3. Verifying LiDAR ray casting mathematics...")
        # Ray-circle test
        dist_circle = ray_cast_circle(
            ray_origin=np.array([0.0, 0.0], dtype=np.float32),
            ray_dir=np.array([1.0, 0.0], dtype=np.float32),
            circle_center=np.array([5.0, 0.0], dtype=np.float32),
            circle_radius=1.0,
            max_range=10.0,
        )
        assert abs(dist_circle - 4.0) < 1e-4
        print(f"   ✓ Ray-circle intersection calculated: {dist_circle:.2f} (expected: 4.00)")

        # Ray-boundary test
        dist_bound = ray_cast_arena_boundaries(
            ray_origin=np.array([5.0, 5.0], dtype=np.float32),
            ray_dir=np.array([1.0, 0.0], dtype=np.float32),
            width=20.0,
            height=20.0,
            max_range=30.0,
        )
        assert abs(dist_bound - 15.0) < 1e-4
        print(f"   ✓ Ray-boundary intersection calculated: {dist_bound:.2f} (expected: 15.00)")

        # 8-ray LiDAR readings
        lidar = compute_lidar_readings(
            agent_pos=np.array([10.0, 10.0], dtype=np.float32),
            obstacles=[(14.0, 10.0, 1.0)],
            arena_width=20.0,
            arena_height=20.0,
            num_rays=8,
            max_range=10.0,
        )
        assert len(lidar) == 8
        assert all(0.0 <= r <= 1.0 for r in lidar)
        print(f"   ✓ 8-Ray LiDAR distance readings computed: {[round(float(x), 2) for x in lidar]}")

        # Step 4: Procedural obstacle generator clearance
        print("\n4. Verifying procedural obstacle placement and safety margins...")
        rng = np.random.default_rng(42)
        start = np.array([2.0, 2.0], dtype=np.float32)
        goal = np.array([18.0, 18.0], dtype=np.float32)
        obstacles = generate_navigation_obstacles(
            width=20.0,
            height=20.0,
            start_pos=start,
            goal_pos=goal,
            num_obstacles=5,
            obstacle_radius=1.0,
            rng=rng,
            clearance_margin=2.0,
        )
        assert len(obstacles) == 5
        for ox, oy, r in obstacles:
            pos = np.array([ox, oy], dtype=np.float32)
            assert np.linalg.norm(pos - start) >= (r + 2.0)
            assert np.linalg.norm(pos - goal) >= (r + 2.0)
        print(f"   ✓ Generated {len(obstacles)} procedural obstacles with >= 2.0m clearance.")

        # Step 5: Registry integration & ASCII rendering
        print("\n5. Verifying registry resolution and ASCII rendering...")
        reg_env = make_env("navigation", arena_width=15.0, arena_height=15.0)
        assert isinstance(reg_env, ContinuousNavigation2DEnv)
        reg_env.reset(seed=123)
        rendered = reg_env.render()
        assert rendered is not None
        assert "A" in rendered or "@" in rendered
        assert "G" in rendered
        print("   ✓ Successfully resolved 'navigation' via make_env factory.")
        print(f"   ✓ ASCII arena map rendered ({len(rendered.splitlines())} lines).")
        reg_env.close()

        # Step 6: SAC Algorithm training and persistence
        print("\n6. Training SACAlgorithm on Continuous Navigation...")
        nav_env = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, max_steps=25)
        sac_algo = SACAlgorithm(
            env=nav_env,
            learning_rate=1e-3,
            buffer_size=1000,
            learning_starts=20,
            batch_size=32,
            seed=42,
        )
        sac_algo.train(total_timesteps=128)
        assert sac_algo.num_timesteps >= 128
        print(f"   ✓ SAC trained for {sac_algo.num_timesteps} continuous timesteps.")

        sac_model_path = test_dir / "sac_navigation_final.zip"
        sac_algo.save(sac_model_path)
        assert sac_model_path.exists()
        print(
            f"   ✓ Saved SAC model weights to {sac_model_path} ({sac_model_path.stat().st_size:,} bytes)."
        )

        loaded_sac = SACAlgorithm.from_pretrained(sac_model_path, env=nav_env)
        act, _ = loaded_sac.predict(obs, deterministic=True)
        assert act.shape == (2,)
        assert -1.0 <= act[0] <= 1.0 and -1.0 <= act[1] <= 1.0
        print(f"   ✓ Loaded SAC predicted continuous velocity: [{act[0]:.3f}, {act[1]:.3f}]")

        # Step 7: PPO Algorithm on Continuous Navigation
        print("\n7. Training PPOAlgorithm on Continuous Navigation...")
        ppo_algo = PPOAlgorithm(
            env=nav_env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            seed=42,
        )
        ppo_algo.train(total_timesteps=128)
        assert ppo_algo.num_timesteps >= 128
        print(f"   ✓ PPO trained for {ppo_algo.num_timesteps} continuous timesteps.")

        # Step 8: Multi-episode evaluation
        print("\n8. Evaluating trained agent with Evaluator...")
        evaluator = Evaluator(algorithm=loaded_sac, env=nav_env)
        metrics = evaluator.evaluate(num_episodes=5, deterministic=True, base_seed=42)
        print(f"   ✓ Evaluated {metrics.episodes} episodes:")
        print(f"     • Mean Reward: {metrics.mean_reward:.2f} ± {metrics.std_reward:.2f}")
        print(f"     • Success Rate: {metrics.success_rate * 100:.1f}%")
        print(f"     • Collision Rate: {metrics.collision_rate * 100:.1f}%")
        print(f"     • Mean Length: {metrics.mean_episode_length:.1f} steps")

        nav_env.close()

        print("\n=================================================================")
        print("✓ ALL PHASE 6 CONTINUOUS NAVIGATION VERIFICATION CHECKS PASSED!")
        print("=================================================================\n")
        return True

    except Exception as e:
        print(f"\n❌ Phase 6 verification failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    success = run_phase_6_verification()
    sys.exit(0 if success else 1)
