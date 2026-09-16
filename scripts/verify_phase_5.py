"""Standalone verification script for AdaptiveRL Phase 5: Evaluation Engine.

Verifies multi-episode benchmarking, standard metrics computation (mean, std,
success rate, collision rate, min/max rewards), scenario benchmarking,
reproducible seeding, and JSON report generation.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.evaluation.scenarios import EvaluationScenario


def run_phase_5_verification() -> bool:
    """Execute all Phase 5 verification checks."""
    print("=== AdaptiveRL Phase 5: Evaluation Engine Verification ===\n")
    test_dir = Path("experiments/verify_phase_5")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Train a model on GridWorld
        print("1. Preparing trained model on GridWorld...")
        env = make_env("gridworld", width=5, height=5, num_obstacles=2, max_steps=25)
        algo = PPOAlgorithm(
            env=env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            seed=42,
        )
        algo.train(total_timesteps=512)
        model_path = test_dir / "eval_model.zip"
        algo.save(model_path)
        assert model_path.exists()
        print(f"   ✓ Model trained for 512 steps and saved ({model_path.stat().st_size:,} bytes).")

        # Step 2: Multi-Episode Evaluation Benchmark
        print("\n2. Executing 20-episode evaluation benchmark...")
        evaluator = Evaluator(algorithm=algo, env=env)
        metrics = evaluator.evaluate(num_episodes=20, deterministic=True, base_seed=12345)

        print(f"   ✓ Episodes Evaluated: {metrics.episodes}")
        print(f"   ✓ Mean Reward: {metrics.mean_reward:.2f} ± {metrics.std_reward:.2f}")
        print(f"   ✓ Min / Max Reward: {metrics.min_reward:.2f} / {metrics.max_reward:.2f}")
        print(f"   ✓ Success Rate: {metrics.success_rate * 100:.1f}%")
        print(f"   ✓ Collision Rate: {metrics.collision_rate * 100:.1f}%")
        print(
            f"   ✓ Mean Episode Length: {metrics.mean_episode_length:.1f} ± {metrics.std_episode_length:.1f}"
        )

        assert metrics.episodes == 20
        assert 0.0 <= metrics.success_rate <= 1.0
        assert 0.0 <= metrics.collision_rate <= 1.0
        assert metrics.min_reward <= metrics.mean_reward <= metrics.max_reward

        # Step 3: Deterministic Reproducibility
        print("\n3. Testing deterministic reproducibility of evaluation...")
        metrics_repeat = evaluator.evaluate(num_episodes=20, deterministic=True, base_seed=12345)
        assert metrics.mean_reward == metrics_repeat.mean_reward
        assert metrics.mean_episode_length == metrics_repeat.mean_episode_length
        assert (
            metrics.additional_metrics["all_rewards"]
            == metrics_repeat.additional_metrics["all_rewards"]
        )
        print("   ✓ Identical base_seed produces bitwise-identical evaluation trajectory.")

        # Step 4: Scenario Benchmarks
        print("\n4. Running multi-scenario benchmark...")
        scenarios = [
            EvaluationScenario(
                name="sparse_obstacles",
                seed=101,
                environment_overrides={"num_obstacles": 1},
            ),
            EvaluationScenario(
                name="dense_obstacles",
                seed=202,
                environment_overrides={"num_obstacles": 4},
            ),
        ]
        scenario_results = evaluator.evaluate_scenarios(scenarios=scenarios, deterministic=True)
        assert "sparse_obstacles" in scenario_results
        assert "dense_obstacles" in scenario_results
        print(
            f"   ✓ Scenario 'sparse_obstacles' Return: {scenario_results['sparse_obstacles'].mean_reward:.2f}"
        )
        print(
            f"   ✓ Scenario 'dense_obstacles' Return:  {scenario_results['dense_obstacles'].mean_reward:.2f}"
        )

        # Step 5: JSON Report Serialization
        print("\n5. Serializing evaluation report to JSON...")
        report_path = test_dir / "evaluation_report.json"
        saved_path = evaluator.save_report(metrics, report_path)
        assert saved_path.exists()
        loaded = json.loads(saved_path.read_text(encoding="utf-8"))
        assert loaded["episodes"] == 20
        assert "mean_reward" in loaded
        assert "success_rate" in loaded
        print(f"   ✓ Evaluation report successfully saved ({saved_path.stat().st_size:,} bytes).")

        env.close()
        print(
            "\n=== Phase 5 Evaluation Engine Verification COMPLETE: All Quality Gates Passed! ==="
        )
        return True
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    success = run_phase_5_verification()
    sys.exit(0 if success else 1)
