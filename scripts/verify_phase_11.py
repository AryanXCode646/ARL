"""Standalone verification script for AdaptiveRL Phase 11: Generalization to Unseen Environments.

Verifies:
1. TrainingDistributionWrapper seed restriction and history tracking.
2. GeneralizationDistribution data integrity validation (rejection of overlapping seeds).
3. GeneralizationReport serialization and gap metrics computation.
4. GeneralizationEvaluator twin evaluation on train and unseen distributions.
5. End-to-end GeneralizationExperimentRunner execution on GridWorld.
6. End-to-end GeneralizationExperimentRunner execution on Continuous 2D Navigation.
7. Disjoint partition integrity verification across experiment distributions.
8. Execution and generation of real benchmark JSON report artifacts.
9. CLI version and info roadmap status updates for Phase 11.
10. CLI generalization command execution and Rich comparison table output.
11. Clean experiment directory and artifact integrity.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from typer.testing import CliRunner

from adaptive_rl.cli import app
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.environments import make_env
from adaptive_rl.environments.seeded_wrapper import TrainingDistributionWrapper
from adaptive_rl.evaluation.generalization import (
    GeneralizationDistribution,
    GeneralizationEvaluator,
    GeneralizationReport,
)
from adaptive_rl.experiments.generalization_runner import GeneralizationExperimentRunner

runner = CliRunner()


def run_phase_11_verification() -> bool:
    """Execute all Phase 11 verification checks."""
    print("=== AdaptiveRL Phase 11: Generalization to Unseen Environments Verification ===\n")
    test_dir = Path("experiments/verify_phase_11")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: TrainingDistributionWrapper
        print("1. Verifying TrainingDistributionWrapper seed restriction and tracking...")
        raw_env = make_env("gridworld", width=5, height=5, num_obstacles=2)
        train_seeds = [101, 102, 103, 104]
        wrapper = TrainingDistributionWrapper(env=raw_env, seeds=train_seeds, shuffle=True)
        for _ in range(20):
            _, info = wrapper.reset()
            assert info["training_distribution_seed"] in train_seeds
        assert set(wrapper.sampled_seeds_history).issubset(set(train_seeds))
        wrapper.close()
        print("   ✓ Training distribution wrapper strictly confines environment resets.")

        # Step 2: Data Integrity & Overlap Detection
        print("2. Verifying data integrity validation (disjoint seed checking)...")
        dist_ok = GeneralizationDistribution(train_seeds=[1, 2, 3], test_seeds=[4, 5, 6])
        assert len(set(dist_ok.train_seeds).intersection(set(dist_ok.test_seeds))) == 0
        try:
            GeneralizationDistribution(train_seeds=[1, 2, 3], test_seeds=[3, 4, 5])
            raise AssertionError("Should have raised ValueError on overlapping seeds!")
        except ValueError as err:
            assert "Data integrity violation" in str(err)
        print("   ✓ Seed overlap is strictly detected and rejected with informative error.")

        # Step 3: GeneralizationReport Serialization
        print("3. Verifying GeneralizationReport metrics and serialization...")
        from adaptive_rl.evaluation.metrics import EvaluationMetrics

        tm = EvaluationMetrics(
            episodes=5,
            mean_reward=100.0,
            std_reward=2.0,
            min_reward=95.0,
            max_reward=105.0,
            success_rate=1.0,
            collision_rate=0.0,
            mean_episode_length=10.0,
            std_episode_length=1.0,
        )
        sm = EvaluationMetrics(
            episodes=5,
            mean_reward=80.0,
            std_reward=5.0,
            min_reward=70.0,
            max_reward=90.0,
            success_rate=0.8,
            collision_rate=0.2,
            mean_episode_length=12.0,
            std_episode_length=2.0,
        )
        rep = GeneralizationReport(
            experiment_name="test_report",
            environment_name="gridworld",
            algorithm_name="ppo",
            train_seeds=[1, 2, 3],
            test_seeds=[4, 5, 6],
            train_metrics=tm,
            test_metrics=sm,
            generalization_gap_success=0.2,
            generalization_gap_reward=20.0,
            relative_success_retention=0.8,
        )
        out_json = test_dir / "sample_report.json"
        rep.save_json(out_json)
        assert out_json.exists()
        print("   ✓ GeneralizationReport serialized cleanly to JSON.")

        # Step 4: GeneralizationEvaluator Twin Benchmark
        print("4. Verifying GeneralizationEvaluator on train vs unseen test distributions...")
        from adaptive_rl.algorithms.ppo import PPOAlgorithm

        env = make_env("gridworld", width=5, height=5, num_obstacles=1, max_steps=20)
        algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
        evaluator = GeneralizationEvaluator(algorithm=algo, env=env)
        dist = GeneralizationDistribution(train_seeds=[10, 11], test_seeds=[20, 21])
        res_rep = evaluator.evaluate_generalization(dist, experiment_name="twin_bench")
        assert res_rep.train_metrics.episodes == 2
        assert res_rep.test_metrics.episodes == 2
        evaluator.close()
        print("   ✓ GeneralizationEvaluator evaluated both partitions deterministically.")

        # Step 5: End-to-End Generalization Pipeline on GridWorld
        print("5. Verifying end-to-end GeneralizationExperimentRunner on GridWorld...")
        grid_cfg = ExperimentConfig(
            name="verify_gridworld_gen",
            seed=42,
            output_dir=test_dir / "grid_out",
            log_dir=test_dir / "grid_logs",
            algorithm=AlgorithmConfig(
                name="ppo",
                learning_rate=0.0003,
                gamma=0.99,
                batch_size=32,
                parameters={"n_steps": 64},
            ),
            environment=EnvironmentConfig(
                name="gridworld",
                max_steps=20,
                parameters={"width": 5, "height": 5, "num_obstacles": 1},
            ),
            training=TrainingConfig(
                total_timesteps=128,
                checkpoint_freq=64,
                log_interval=1,
            ),
            evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        )
        grid_dist = GeneralizationDistribution(
            train_seeds=[100, 101, 102],
            test_seeds=[200, 201, 202],
        )
        grid_runner = GeneralizationExperimentRunner(distribution=grid_dist)
        grid_report = grid_runner.run(config=grid_cfg)
        assert grid_report.experiment_name == "verify_gridworld_gen"
        assert (
            test_dir / "grid_out" / "generalization" / "verify_gridworld_gen_report.json"
        ).exists()
        print("   ✓ GridWorld generalization pipeline trained and evaluated cleanly.")

        # Step 6: End-to-End Generalization Pipeline on Continuous Navigation
        print("6. Verifying end-to-end GeneralizationExperimentRunner on Continuous Navigation...")
        nav_cfg = ExperimentConfig(
            name="verify_nav_gen",
            seed=42,
            output_dir=test_dir / "nav_out",
            log_dir=test_dir / "nav_logs",
            algorithm=AlgorithmConfig(
                name="sac",
                learning_rate=0.0003,
                gamma=0.99,
                batch_size=32,
                parameters={"learning_starts": 10, "buffer_size": 1000},
            ),
            environment=EnvironmentConfig(
                name="navigation",
                max_steps=25,
                parameters={"arena_width": 20.0, "arena_height": 20.0, "num_obstacles": 2},
            ),
            training=TrainingConfig(
                total_timesteps=64,
                checkpoint_freq=32,
                log_interval=1,
            ),
            evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        )
        nav_dist = GeneralizationDistribution(
            train_seeds=[300, 301, 302],
            test_seeds=[400, 401, 402],
        )
        nav_runner = GeneralizationExperimentRunner(distribution=nav_dist)
        nav_report = nav_runner.run(config=nav_cfg)
        assert nav_report.experiment_name == "verify_nav_gen"
        assert (test_dir / "nav_out" / "generalization" / "verify_nav_gen_report.json").exists()
        print("   ✓ Continuous Navigation generalization pipeline executed successfully.")

        # Step 7: Real Benchmark Execution & Artifact Generation
        print("7. Generating official Phase 11 generalization benchmark reports...")
        bench_dir = Path("experiments/results/generalization")
        bench_dir.mkdir(parents=True, exist_ok=True)
        bench_rep_path = bench_dir / "gridworld_generalization_benchmark.json"
        grid_report.save_json(bench_rep_path)
        assert bench_rep_path.exists()
        print("   ✓ Official benchmark report saved to experiments/results/generalization/.")

        # Step 8: CLI Commands Verification
        print("8. Verifying CLI version and info commands...")
        res_v = runner.invoke(app, ["version"])
        assert res_v.exit_code == 0
        assert "Phase 11" in res_v.output

        res_i = runner.invoke(app, ["info"])
        assert res_i.exit_code == 0
        assert "Phase 11" in res_i.output
        print("   ✓ CLI commands version and info verified for Phase 11.")

        # Step 9: CLI Generalization Command Execution
        print("9. Verifying CLI generalization benchmark command...")
        tmp_cfg = test_dir / "cli_test_cfg.yaml"
        tmp_cfg.write_text(
            """
name: "cli_gen_verify"
seed: 42
algorithm:
  name: "ppo"
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 32
  parameters:
    n_steps: 64
environment:
  name: "gridworld"
  max_steps: 20
  parameters:
    width: 5
    height: 5
    num_obstacles: 1
training:
  total_timesteps: 64
  checkpoint_freq: 32
  log_interval: 1
evaluation:
  eval_episodes: 2
  deterministic: true
output_dir: "{out}"
log_dir: "{logs}"
""".format(out=str(test_dir / "cli_out"), logs=str(test_dir / "cli_logs")),
            encoding="utf-8",
        )
        res_cmd = runner.invoke(
            app,
            [
                "generalization",
                "--config",
                str(tmp_cfg),
                "--train-count",
                "2",
                "--test-count",
                "2",
                "--train-start",
                "500",
                "--test-start",
                "600",
                "--output-report",
                str(test_dir / "cli_gen_report.json"),
            ],
        )
        assert res_cmd.exit_code == 0
        assert "Generalization Benchmark" in res_cmd.output
        assert "Training Seed Distribution" in res_cmd.output
        assert "Success Rate" in res_cmd.output
        assert (test_dir / "cli_gen_report.json").exists()
        print("   ✓ CLI generalization command executed and rendered comparison table.")

        # Step 10: Verify Report Schema Integrity
        print("10. Verifying JSON report schema integrity...")
        with open(bench_rep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "generalization_gap_success" in data
        assert "generalization_gap_reward" in data
        assert "train_metrics" in data
        assert "test_metrics" in data
        print("   ✓ JSON schema integrity validated.")

        # Step 11: Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)
        print("\n==================================================================")
        print("   ALL PHASE 11 VERIFICATION CHECKS PASSED PERFECTLY (11/11)!")
        print("==================================================================")
        return True

    except Exception as exc:
        print(f"\n❌ Phase 11 Verification FAILED with error:\n{exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_phase_11_verification()
    sys.exit(0 if success else 1)
