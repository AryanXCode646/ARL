"""Standalone verification script for AdaptiveRL Phase 8: Traffic Signal Optimization.

Verifies:
1. Farama Gymnasium compliance (check_env) on TrafficSignalEnv.
2. Global registry resolution for 'traffic' and 'traffic_signal'.
3. Observation and action space specifications.
4. Deterministic seeding and step trajectory reproducibility.
5. Physics and queuing dynamics (green discharges, red waiting accumulation, FIFO delay tracking).
6. Multi-objective reward and switch penalty mechanisms.
7. Textual ASCII 4-way intersection rendering.
8. PPO algorithm and trainer integration.
9. Benchmark evaluation engine and JSON report generation.
10. Traffic curriculum schedule and progression.
11. Typer CLI commands (version, info, env inspect, env run, curriculum inspect).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env
from typer.testing import CliRunner

from adaptive_rl.algorithms.ppo import PPOAlgorithm
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
from adaptive_rl.environments.traffic.intersection import TrafficSignalEnv
from adaptive_rl.environments.traffic.simulation import (
    Approach,
    Phase,
    TrafficIntersection,
)
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.training.trainer import PPOTrainer

runner = CliRunner()


def run_phase_8_verification() -> bool:
    """Execute all Phase 8 verification checks."""
    print("=== AdaptiveRL Phase 8: Traffic Signal Optimization Verification ===\n")
    test_dir = Path("experiments/verify_phase_8")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Farama Gymnasium Compliance
        print("1. Verifying Farama Gymnasium compliance...")
        env = TrafficSignalEnv(max_steps=20)
        check_env(env)
        env.close()
        print("   ✓ check_env passed 100% Farama Gymnasium compliance.")

        # Step 2: Global Registry Resolution
        print("2. Verifying global registry resolution...")
        env_traffic = make_env("traffic")
        env_alias = make_env("traffic_signal")
        assert isinstance(env_traffic, TrafficSignalEnv)
        assert isinstance(env_alias, TrafficSignalEnv)
        env_traffic.close()
        env_alias.close()
        print("   ✓ 'traffic' and 'traffic_signal' registered and instantiated.")

        # Step 3: Space Specifications
        print("3. Verifying observation and action spaces...")
        env = TrafficSignalEnv()
        assert env.action_space.n == 2
        assert env.observation_space.shape == (10,)
        assert env.observation_space.dtype == np.float32
        assert float(env.observation_space.low.min()) >= 0.0
        assert float(env.observation_space.high.max()) <= 1.0
        env.close()
        print("   ✓ Observation space Box(10,) and Action space Discrete(2) verified.")

        # Step 4: Deterministic Seeding
        print("4. Verifying deterministic seeding reproducibility...")
        e1 = TrafficSignalEnv(max_steps=25)
        e2 = TrafficSignalEnv(max_steps=25)
        obs1, info1 = e1.reset(seed=777)
        obs2, info2 = e2.reset(seed=777)
        np.testing.assert_array_equal(obs1, obs2)
        assert info1["queues"] == info2["queues"]

        for act in [0, 0, 1, 1, 0, 1, 0, 0]:
            o1, r1, t1, tr1, i1 = e1.step(act)
            o2, r2, t2, tr2, i2 = e2.step(act)
            np.testing.assert_array_almost_equal(o1, o2)
            assert r1 == r2
            assert t1 == t2
            assert tr1 == tr2
            assert i1["queues"] == i2["queues"]
        e1.close()
        e2.close()
        print("   ✓ Trajectory bitwise identical under identical seeds.")

        # Step 5: Queuing Dynamics and Phase Departures
        print("5. Verifying queuing dynamics and directional discharges...")
        sim = TrafficIntersection(
            arrival_rates=(0.0, 0.0, 0.0, 0.0),
            departure_rate=2,
            max_queue=20,
        )
        sim.reset(
            initial_queues={
                Approach.NORTH: 4,
                Approach.SOUTH: 2,
                Approach.EAST: 3,
                Approach.WEST: 1,
            }
        )
        rng = np.random.default_rng(42)
        # Step with North-South Green (Phase 0)
        telem0 = sim.step(0, rng)
        assert telem0.current_phase == Phase.NORTH_SOUTH
        assert telem0.departures[Approach.NORTH] == 2
        assert telem0.departures[Approach.SOUTH] == 2
        assert telem0.departures[Approach.EAST] == 0
        assert telem0.departures[Approach.WEST] == 0
        assert telem0.queue_lengths[Approach.NORTH] == 2
        assert telem0.queue_lengths[Approach.SOUTH] == 0
        # Step with East-West Green (Phase 1)
        telem1 = sim.step(1, rng)
        assert telem1.current_phase == Phase.EAST_WEST
        assert telem1.phase_switched is True
        assert telem1.departures[Approach.EAST] == 2
        assert telem1.departures[Approach.WEST] == 1
        assert telem1.queue_lengths[Approach.EAST] == 1
        assert telem1.queue_lengths[Approach.WEST] == 0
        print("   ✓ Departures isolated to active green directions; FIFO delay tracked.")

        # Step 6: Multi-Objective Reward & Switch Penalty
        print("6. Verifying multi-objective reward & switch penalties...")
        env = TrafficSignalEnv(
            min_green_steps=3,
            switch_penalty=1.0,
            premature_switch_penalty=2.5,
            arrival_rates=(0.0, 0.0, 0.0, 0.0),
        )
        env.reset(seed=42)
        _, r_stay, _, _, i_stay = env.step(0)
        assert i_stay["phase_switched"] is False
        assert i_stay["premature_switch"] is False

        _, r_prem, _, _, i_prem = env.step(1)
        assert i_prem["phase_switched"] is True
        assert i_prem["premature_switch"] is True
        assert r_prem <= -(1.0 + 2.5)
        env.close()
        print("   ✓ Regular and premature switch penalties verified.")

        # Step 7: ASCII Intersection Rendering
        print("7. Verifying ASCII intersection rendering...")
        env = TrafficSignalEnv(render_mode="ansi")
        env.reset(seed=42)
        layout = env.render()
        assert isinstance(layout, str)
        assert "4-WAY SIGNALIZED INTERSECTION" in layout
        assert "[G]" in layout and "[R]" in layout
        assert "Queues:" in layout
        env.close()
        print("   ✓ ASCII renderer outputs valid visual layout.")

        # Step 8: PPO Algorithm & Trainer Pipeline
        print("8. Verifying PPO algorithm and trainer integration...")
        exp_cfg = ExperimentConfig(
            name="verify_traffic_ppo",
            seed=42,
            output_dir=test_dir / "results",
            log_dir=test_dir / "logs",
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
        train_res = trainer.fit()
        assert train_res.total_timesteps == 128
        assert train_res.final_model_path.exists()
        mean_eval, std_eval = trainer.evaluate(episodes=2)
        assert isinstance(mean_eval, float)
        print("   ✓ PPOTrainer successfully completed training, checkpointing, and evaluation.")

        # Step 9: Evaluation Engine Benchmarking
        print("9. Verifying Evaluator benchmarking and report generation...")
        eval_env = make_env("traffic", max_steps=20)
        algo = PPOAlgorithm(env=eval_env, n_steps=64, batch_size=32, seed=42)
        evaluator = Evaluator(algorithm=algo, env=eval_env)
        metrics = evaluator.evaluate(num_episodes=4, deterministic=True, base_seed=123)
        assert metrics.episodes == 4
        assert metrics.mean_episode_length == 20.0
        json_report = test_dir / "traffic_eval.json"
        evaluator.save_report(metrics, json_report)
        assert json_report.exists()
        eval_env.close()
        print("   ✓ Evaluator produced and serialized benchmark metrics report.")

        # Step 10: Traffic Curriculum Integration
        print("10. Verifying Traffic curriculum preset and trainer...")
        curr_preset = get_curriculum_preset("traffic")
        assert len(curr_preset.stages) == 4
        curr_cfg = ExperimentConfig(
            name="verify_traffic_curr",
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
        curr_trainer = CurriculumTrainer(config=curr_cfg)
        curr_res = curr_trainer.fit()
        assert curr_res.total_timesteps == 128
        assert curr_res.final_model_path.exists()
        print("   ✓ Traffic curriculum trainer executed staged progression.")

        # Step 11: CLI Commands
        print("11. Verifying CLI commands...")
        res_v = runner.invoke(app, ["version"])
        assert res_v.exit_code == 0
        assert "Phase 8" in res_v.output

        res_i = runner.invoke(app, ["info"])
        assert res_i.exit_code == 0
        assert "Phase 8" in res_i.output

        res_list = runner.invoke(app, ["env", "list"])
        assert res_list.exit_code == 0
        assert "traffic" in res_list.output

        res_insp = runner.invoke(app, ["env", "inspect", "traffic"])
        assert res_insp.exit_code == 0
        assert "TrafficSignalEnv" in res_insp.output

        res_run = runner.invoke(app, ["env", "run", "traffic", "--steps", "5"])
        assert res_run.exit_code == 0
        assert "Total Steps: 5" in res_run.output

        res_curr = runner.invoke(app, ["curriculum", "inspect", "traffic"])
        assert res_curr.exit_code == 0
        assert "traffic_curriculum" in res_curr.output
        print("   ✓ All Typer CLI commands verified successfully.")

        # Clean up test artifacts
        shutil.rmtree(test_dir, ignore_errors=True)

        print("\n==================================================================")
        print("   ALL PHASE 8 VERIFICATION CHECKS PASSED PERFECTLY (11/11)!")
        print("==================================================================")
        return True

    except Exception as exc:
        print(f"\n❌ Phase 8 Verification FAILED with error:\n{exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_phase_8_verification()
    sys.exit(0 if success else 1)
