"""Standalone verification script for AdaptiveRL Phase 10: Drone Disturbances and Constraints.

Verifies:
1. Farama Gymnasium compliance (check_env) on DroneDisturbance3DEnv.
2. Global registry resolution for 'drone_disturbed' and 'drone_constrained'.
3. Continuous 3D action space Box(3,) and 33-dimensional observation space Box(33,).
4. Atmospheric wind dynamics: steady vector fields, altitude shear, and Ornstein-Uhlenbeck stochastic turbulence.
5. Quadrotor battery depletion dynamics, power consumption, and telemetry.
6. Dynamic 3D obstacle motion and boundary reflection.
7. Wind drift force on drone translational kinematics.
8. Battery exhaustion episode termination and penalty enforcement.
9. Continuous PPO and SAC training pipelines on disturbed drone navigation.
10. Disturbed drone curriculum progression and parameter updates.
11. Typer CLI commands (version, info, env inspect, env run, curriculum inspect).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env
from typer.testing import CliRunner

from adaptive_rl.cli import app
from adaptive_rl.config import (
    AlgorithmConfig,
    CurriculumConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.curriculum.trainer import CurriculumTrainer
from adaptive_rl.environments import make_env
from adaptive_rl.environments.drone.battery import BatteryModel
from adaptive_rl.environments.drone.disturbed_drone import DroneDisturbance3DEnv
from adaptive_rl.environments.drone.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.environments.drone.wind import WindField3D
from adaptive_rl.training.trainer import PPOTrainer, SACTrainer

runner = CliRunner()


def run_phase_10_verification() -> bool:
    """Execute all Phase 10 verification checks."""
    print("=== AdaptiveRL Phase 10: Drone Disturbances and Constraints Verification ===\n")
    test_dir = Path("experiments/verify_phase_10")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Farama Gymnasium Compliance
        print("1. Verifying Farama Gymnasium compliance...")
        env = DroneDisturbance3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20)
        check_env(env)
        env.close()
        print("   ✓ check_env passed 100% Farama Gymnasium compliance.")

        # Step 2: Global Registry Resolution
        print("2. Verifying global registry resolution under aliases...")
        e_dist = make_env("drone_disturbed")
        e_const = make_env("drone_constrained")
        assert isinstance(e_dist, DroneDisturbance3DEnv)
        assert isinstance(e_const, DroneDisturbance3DEnv)
        e_dist.close()
        e_const.close()
        print("   ✓ 'drone_disturbed' and 'drone_constrained' registered and resolved.")

        # Step 3: Space Specifications
        print("3. Verifying continuous 3D observation and action spaces...")
        env = DroneDisturbance3DEnv(bounds=(50.0, 50.0, 25.0), num_lidar_rays=16)
        assert env.action_space.shape == (3,)
        assert env.action_space.dtype == np.float32
        assert env.observation_space.shape == (33,)
        assert env.observation_space.dtype == np.float32
        env.close()
        print("   ✓ Action space Box(3,) and Observation space Box(33,) verified.")

        # Step 4: Atmospheric Wind Dynamics
        print("4. Verifying atmospheric wind dynamics, shear, and OU turbulence...")
        wind = WindField3D(
            steady_wind=(2.0, 1.0, 0.0),
            gust_theta=0.15,
            gust_sigma=0.4,
            altitude_shear=0.04,
            dt=0.1,
        )
        rng = np.random.default_rng(123)
        w0 = wind.get_wind(np.array([0.0, 0.0, 0.0]))
        w_high = wind.get_wind(np.array([0.0, 0.0, 25.0]))
        # Altitude factor: 1 + 0.04 * 25 = 2.0
        assert w_high.speed > w0.speed
        np.testing.assert_allclose(w_high.steady, w0.steady * 2.0)
        for _ in range(5):
            wind.step_gust(rng)
        assert wind._current_gust.shape == (3,)
        print("   ✓ 3D Wind fields, altitude shear, and Ornstein-Uhlenbeck turbulence verified.")

        # Step 5: Quadrotor Battery Depletion Dynamics
        print("5. Verifying quadrotor battery depletion dynamics and telemetry...")
        battery = BatteryModel(capacity=50.0, base_power=0.1, thrust_coefficient=0.2)
        telem = battery.step(np.array([1.0, 1.0, 1.0]), np.array([2.0, 0.0, 0.0]), dt=0.5)
        assert telem.remaining_energy < 50.0
        assert 0.0 < telem.state_of_charge < 1.0
        assert not telem.is_depleted
        print("   ✓ Battery depletion dynamics and power draw verified.")

        # Step 6: Dynamic 3D Obstacle Motion & Bouncing
        print("6. Verifying dynamic obstacle motion and boundary bouncing...")
        dyn_obs = DynamicObstacleSphere3D(
            center=np.array([19.0, 10.0, 10.0]),
            velocity=np.array([3.0, 0.0, 0.0]),
            radius=1.5,
            bounds=(20.0, 20.0, 20.0),
        )
        dyn_obs.step(dt=0.5)
        # Bounded reflection: velocity negated
        assert dyn_obs.velocity[0] < 0.0
        assert dyn_obs.center[0] <= 20.0 - 1.5
        print("   ✓ Dynamic 3D obstacles and boundary reflections verified.")

        # Step 7: Physical Wind Drift on Kinematics
        print("7. Verifying physical wind drift force on translational kinematics...")
        env = DroneDisturbance3DEnv(
            bounds=(40.0, 40.0, 20.0),
            steady_wind=(4.0, 0.0, 0.0),
            gust_sigma=0.0,
            dt=0.1,
        )
        env.reset(seed=42)
        init_x = env._position[0]
        # Command neutral zero acceleration
        for _ in range(5):
            env.step(np.zeros(3, dtype=np.float32))
        assert env._position[0] > init_x
        assert env._velocity[0] > 0.0
        env.close()
        print("   ✓ Wind drift physically accelerates drone along prevailing vector.")

        # Step 8: Battery Exhaustion Episode Termination
        print("8. Verifying battery exhaustion termination and penalty...")
        env = DroneDisturbance3DEnv(
            bounds=(30.0, 30.0, 15.0),
            battery_capacity=0.2,
            battery_base_power=1.0,
            battery_exhaustion_penalty=-30.0,
            terminate_on_exhaustion=True,
        )
        env.reset(seed=42)
        terminated = False
        step_k = 0
        while not terminated and step_k < 10:
            _, rew, terminated, _, info = env.step(np.ones(3, dtype=np.float32))
            step_k += 1
        assert terminated
        assert info["battery_exhausted"]
        assert rew == -30.0
        env.close()
        print("   ✓ Battery exhaustion correctly terminates episode with penalty.")

        # Step 9: Continuous PPO and SAC Training Pipelines
        print("9. Verifying PPO and SAC training on disturbed drone environment...")
        ppo_cfg = ExperimentConfig(
            name="verify_drone_disturbed_ppo",
            seed=42,
            output_dir=test_dir / "ppo_results",
            log_dir=test_dir / "ppo_logs",
            algorithm=AlgorithmConfig(
                name="ppo",
                learning_rate=0.0003,
                gamma=0.99,
                batch_size=64,
                parameters={"n_steps": 128},
            ),
            environment=EnvironmentConfig(
                name="drone_disturbed",
                max_steps=30,
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
            evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        )
        ppo_trainer = PPOTrainer(config=ppo_cfg)
        ppo_res = ppo_trainer.fit()
        assert ppo_res.total_timesteps == 128
        assert ppo_res.final_model_path.exists()

        sac_cfg = ExperimentConfig(
            name="verify_drone_disturbed_sac",
            seed=42,
            output_dir=test_dir / "sac_results",
            log_dir=test_dir / "sac_logs",
            algorithm=AlgorithmConfig(
                name="sac",
                learning_rate=0.0003,
                gamma=0.99,
                batch_size=64,
                parameters={"learning_starts": 64, "buffer_size": 1000},
            ),
            environment=EnvironmentConfig(
                name="drone_disturbed",
                max_steps=30,
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
            evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        )
        sac_trainer = SACTrainer(config=sac_cfg)
        sac_res = sac_trainer.fit()
        assert sac_res.total_timesteps == 128
        assert sac_res.final_model_path.exists()
        print("   ✓ Continuous PPO and SAC training executed successfully.")

        # Step 10: Curriculum Progression
        print("10. Verifying progressive disturbed drone curriculum learning...")
        curr_cfg = ExperimentConfig(
            name="verify_drone_disturbed_curriculum",
            seed=42,
            output_dir=test_dir / "curriculum_results",
            log_dir=test_dir / "curriculum_logs",
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
            evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        )
        curr_trainer = CurriculumTrainer(config=curr_cfg)
        curr_res = curr_trainer.fit()
        assert curr_res.total_timesteps == 128
        assert curr_res.final_model_path.exists()
        print("   ✓ Disturbed drone curriculum successfully executed staged training.")

        # Step 11: CLI Commands
        print("11. Verifying CLI commands...")
        res_v = runner.invoke(app, ["version"])
        assert res_v.exit_code == 0
        assert "Phase 10" in res_v.output

        res_i = runner.invoke(app, ["info"])
        assert res_i.exit_code == 0
        assert "Phase 10" in res_i.output

        res_list = runner.invoke(app, ["env", "list"])
        assert res_list.exit_code == 0
        assert "drone_disturbed" in res_list.output

        res_insp = runner.invoke(app, ["env", "inspect", "drone_disturbed"])
        assert res_insp.exit_code == 0
        assert "DroneDisturbance3DEnv" in res_insp.output

        res_run = runner.invoke(app, ["env", "run", "drone_disturbed", "--steps", "5"])
        assert res_run.exit_code == 0
        assert "Total Steps: 5" in res_run.output

        res_curr = runner.invoke(app, ["curriculum", "inspect", "drone_disturbed"])
        assert res_curr.exit_code == 0
        assert "disturbed_drone_curriculum" in res_curr.output
        print("   ✓ All Typer CLI commands verified successfully.")

        # Clean up test artifacts
        shutil.rmtree(test_dir, ignore_errors=True)

        print("\n==================================================================")
        print("   ALL PHASE 10 VERIFICATION CHECKS PASSED PERFECTLY (11/11)!")
        print("==================================================================")
        return True

    except Exception as exc:
        print(f"\n❌ Phase 10 Verification FAILED with error:\n{exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_phase_10_verification()
    sys.exit(0 if success else 1)
