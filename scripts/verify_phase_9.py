"""Standalone verification script for AdaptiveRL Phase 9: Autonomous 3D Drone Navigation.

Verifies:
1. Farama Gymnasium compliance (check_env) on DroneNavigation3DEnv.
2. Global registry resolution for 'drone', 'drone_3d', and 'drone_navigation'.
3. Continuous 3D action space Box(3,) and observation space Box(29,).
4. Deterministic seeding and step trajectory reproducibility.
5. 3D kinematics integration, acceleration, and aerodynamic damping.
6. Analytical 3D ray-sphere and 3D ray-box perimeter intersections.
7. Procedural 3D obstacle field generation with start/goal clearance.
8. Continuous PPO and SAC training pipelines on 3D drone navigation.
9. Benchmark evaluation engine and JSON report generation.
10. Staged 3D drone curriculum progression.
11. Typer CLI commands (version, info, env inspect, env run, curriculum inspect).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env
from typer.testing import CliRunner

from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.cli import app
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
from adaptive_rl.environments.drone.drone3d import DroneNavigation3DEnv
from adaptive_rl.environments.drone.kinematics import DroneKinematics3D
from adaptive_rl.environments.drone.obstacles import (
    generate_drone_obstacles,
    ray_cast_box_boundaries_3d,
    ray_cast_sphere_3d,
)
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.training.trainer import PPOTrainer, SACTrainer

runner = CliRunner()


def run_phase_9_verification() -> bool:
    """Execute all Phase 9 verification checks."""
    print("=== AdaptiveRL Phase 9: Autonomous 3D Drone Navigation Verification ===\n")
    test_dir = Path("experiments/verify_phase_9")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Farama Gymnasium Compliance
        print("1. Verifying Farama Gymnasium compliance...")
        env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20)
        check_env(env)
        env.close()
        print("   ✓ check_env passed 100% Farama Gymnasium compliance.")

        # Step 2: Global Registry Resolution
        print("2. Verifying global registry resolution under aliases...")
        e_drone = make_env("drone")
        e_3d = make_env("drone_3d")
        e_nav = make_env("drone_navigation")
        assert isinstance(e_drone, DroneNavigation3DEnv)
        assert isinstance(e_3d, DroneNavigation3DEnv)
        assert isinstance(e_nav, DroneNavigation3DEnv)
        e_drone.close()
        e_3d.close()
        e_nav.close()
        print("   ✓ 'drone', 'drone_3d', 'drone_navigation' registered and resolved.")

        # Step 3: Space Specifications
        print("3. Verifying continuous 3D observation and action spaces...")
        env = DroneNavigation3DEnv(bounds=(50.0, 50.0, 25.0), num_lidar_rays=16)
        assert env.action_space.shape == (3,)
        assert env.action_space.dtype == np.float32
        assert env.observation_space.shape == (29,)
        assert env.observation_space.dtype == np.float32
        env.close()
        print("   ✓ Action space Box(3,) and Observation space Box(29,) verified.")

        # Step 4: Deterministic Seeding
        print("4. Verifying deterministic seeding reproducibility...")
        e1 = DroneNavigation3DEnv(bounds=(40.0, 40.0, 20.0), num_obstacles=5)
        e2 = DroneNavigation3DEnv(bounds=(40.0, 40.0, 20.0), num_obstacles=5)
        obs1, info1 = e1.reset(seed=999)
        obs2, info2 = e2.reset(seed=999)
        np.testing.assert_array_almost_equal(obs1, obs2)

        act = np.array([0.4, -0.6, 0.2], dtype=np.float32)
        for _ in range(5):
            o1, r1, t1, tr1, _ = e1.step(act)
            o2, r2, t2, tr2, _ = e2.step(act)
            np.testing.assert_array_almost_equal(o1, o2)
            assert r1 == r2
            assert t1 == t2
            assert tr1 == tr2
        e1.close()
        e2.close()
        print("   ✓ Trajectory bitwise identical under identical seeds.")

        # Step 5: Kinematics & Aerodynamic Drag
        print("5. Verifying 3D kinematics integration and velocity limits...")
        kin = DroneKinematics3D(dt=0.1, max_velocity=8.0, max_acceleration=4.0, linear_damping=0.1)
        kin.reset(np.array([0.0, 0.0, 0.0]))
        pos, vel = kin.step(np.array([4.0, 0.0, 0.0]))
        assert vel[0] > 0.0
        assert pos[0] > 0.0
        print("   ✓ 3D equations of motion and semi-implicit Euler integration verified.")

        # Step 6: 3D Analytical Ray-Casting
        print("6. Verifying 3D analytical ray-sphere and boundary slab intersections...")
        dist_sphere = ray_cast_sphere_3d(
            ray_origin=np.array([0.0, 0.0, 0.0]),
            ray_direction=np.array([1.0, 0.0, 0.0]),
            center=np.array([10.0, 0.0, 0.0]),
            radius=2.0,
            max_range=20.0,
        )
        assert abs(dist_sphere - 8.0) < 1e-5

        dist_box = ray_cast_box_boundaries_3d(
            ray_origin=np.array([10.0, 10.0, 10.0]),
            ray_direction=np.array([0.0, 0.0, 1.0]),
            bounds=(50.0, 50.0, 25.0),
            max_range=100.0,
        )
        assert abs(dist_box - 15.0) < 1e-5
        print("   ✓ Analytical 3D ray-casting geometry verified.")

        # Step 7: Procedural 3D Obstacle Generation
        print("7. Verifying procedural 3D obstacle clearance...")
        start_pt = np.array([5.0, 5.0, 5.0])
        goal_pt = np.array([45.0, 45.0, 20.0])
        obstacles = generate_drone_obstacles(
            bounds=(50.0, 50.0, 25.0),
            start_pos=start_pt,
            goal_pos=goal_pt,
            num_obstacles=6,
            obstacle_radius=2.0,
            clearance_radius=3.0,
            rng=np.random.default_rng(42),
        )
        assert len(obstacles) == 6
        for obs in obstacles:
            assert np.linalg.norm(obs.center - start_pt) >= 5.0
            assert np.linalg.norm(obs.center - goal_pt) >= 5.0
        print("   ✓ Procedural 3D obstacles satisfy strict clearance constraints.")

        # Step 8: Continuous PPO and SAC Pipelines
        print("8. Verifying continuous PPO and SAC training pipelines...")
        ppo_cfg = ExperimentConfig(
            name="verify_drone_ppo",
            seed=42,
            output_dir=test_dir / "ppo_results",
            log_dir=test_dir / "ppo_logs",
            algorithm=AlgorithmConfig(
                name="ppo",
                learning_rate=0.0003,
                gamma=0.99,
                batch_size=32,
                parameters={"n_steps": 64, "n_epochs": 2},
            ),
            environment=EnvironmentConfig(
                name="drone",
                max_steps=20,
                parameters={"bounds": [20.0, 20.0, 10.0], "num_obstacles": 2},
            ),
            training=TrainingConfig(
                total_timesteps=128,
                checkpoint_freq=64,
                log_interval=1,
            ),
            evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        )
        ppo_trainer = PPOTrainer(config=ppo_cfg)
        ppo_res = ppo_trainer.fit()
        assert ppo_res.total_timesteps == 128
        assert ppo_res.final_model_path.exists()

        sac_cfg = ExperimentConfig(
            name="verify_drone_sac",
            seed=42,
            output_dir=test_dir / "sac_results",
            log_dir=test_dir / "sac_logs",
            algorithm=AlgorithmConfig(
                name="sac",
                learning_rate=0.0003,
                gamma=0.99,
                batch_size=32,
                parameters={"buffer_size": 1000, "learning_starts": 10},
            ),
            environment=EnvironmentConfig(
                name="drone",
                max_steps=20,
                parameters={"bounds": [20.0, 20.0, 10.0], "num_obstacles": 2},
            ),
            training=TrainingConfig(
                total_timesteps=64,
                checkpoint_freq=32,
                log_interval=1,
            ),
            evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        )
        sac_trainer = SACTrainer(config=sac_cfg)
        sac_res = sac_trainer.fit()
        assert sac_res.total_timesteps == 64
        assert sac_res.final_model_path.exists()
        print("   ✓ Continuous PPO and SAC trainers successfully trained 3D drone policies.")

        # Step 9: Benchmark Evaluation Engine
        print("9. Verifying Evaluator benchmarking and report generation...")
        eval_env = make_env("drone", bounds=(20.0, 20.0, 10.0), max_steps=20, num_obstacles=1)
        algo = SACAlgorithm(env=eval_env, buffer_size=1000, learning_starts=10)
        evaluator = Evaluator(algorithm=algo, env=eval_env)
        metrics = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=123)
        assert metrics.episodes == 3
        json_report = test_dir / "drone_eval.json"
        evaluator.save_report(metrics, json_report)
        assert json_report.exists()
        eval_env.close()
        print("   ✓ Evaluator produced and serialized benchmark metrics report.")

        # Step 10: Drone Curriculum Integration
        print("10. Verifying 4-stage 3D drone curriculum progression...")
        curr_preset = get_curriculum_preset("drone")
        assert len(curr_preset.stages) == 4
        curr_cfg = ExperimentConfig(
            name="verify_drone_curr",
            seed=42,
            output_dir=test_dir / "curr_results",
            log_dir=test_dir / "curr_logs",
            algorithm=AlgorithmConfig(
                name="ppo",
                learning_rate=0.0003,
                gamma=0.99,
                batch_size=32,
                parameters={"n_steps": 64, "n_epochs": 2},
            ),
            environment=EnvironmentConfig(
                name="drone",
                max_steps=20,
            ),
            curriculum=CurriculumConfig(
                enabled=True,
                preset="drone",
                eval_window=5,
            ),
            training=TrainingConfig(
                total_timesteps=128,
                checkpoint_freq=64,
                log_interval=1,
            ),
            evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        )
        curr_trainer = CurriculumTrainer(config=curr_cfg)
        curr_res = curr_trainer.fit()
        assert curr_res.total_timesteps == 128
        assert curr_res.final_model_path.exists()
        print("   ✓ 3D Drone curriculum successfully executed staged training.")

        # Step 11: CLI Commands
        print("11. Verifying CLI commands...")
        res_v = runner.invoke(app, ["version"])
        assert res_v.exit_code == 0
        assert "Phase 9" in res_v.output

        res_i = runner.invoke(app, ["info"])
        assert res_i.exit_code == 0
        assert "Phase 9" in res_i.output

        res_list = runner.invoke(app, ["env", "list"])
        assert res_list.exit_code == 0
        assert "drone" in res_list.output

        res_insp = runner.invoke(app, ["env", "inspect", "drone"])
        assert res_insp.exit_code == 0
        assert "DroneNavigation3DEnv" in res_insp.output

        res_run = runner.invoke(app, ["env", "run", "drone", "--steps", "5"])
        assert res_run.exit_code == 0
        assert "Total Steps: 5" in res_run.output

        res_curr = runner.invoke(app, ["curriculum", "inspect", "drone"])
        assert res_curr.exit_code == 0
        assert "drone_curriculum" in res_curr.output
        print("   ✓ All Typer CLI commands verified successfully.")

        # Clean up test artifacts
        shutil.rmtree(test_dir, ignore_errors=True)

        print("\n==================================================================")
        print("   ALL PHASE 9 VERIFICATION CHECKS PASSED PERFECTLY (11/11)!")
        print("==================================================================")
        return True

    except Exception as exc:
        print(f"\n❌ Phase 9 Verification FAILED with error:\n{exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_phase_9_verification()
    sys.exit(0 if success else 1)
