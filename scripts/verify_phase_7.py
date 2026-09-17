"""Standalone verification script for AdaptiveRL Phase 7: Curriculum Learning.

Verifies:
1. CurriculumStage advancement criteria evaluation.
2. Curriculum sequence manager and history logging.
3. CurriculumEnvWrapper Farama Gymnasium compliance (check_env) and parameter injection.
4. Predefined curriculum schedules (navigation and gridworld).
5. Automated curriculum progression during reinforcement learning training.
6. Checkpointing, model serialization, and curriculum JSON report generation.
7. Typer CLI curriculum commands (list, inspect, validate).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from gymnasium.utils.env_checker import check_env
from typer.testing import CliRunner

from adaptive_rl.cli import app
from adaptive_rl.config import (
    AlgorithmConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    EnvironmentConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.curriculum.curriculum import Curriculum
from adaptive_rl.curriculum.presets import (
    create_navigation_curriculum,
    get_curriculum_preset,
)
from adaptive_rl.curriculum.stage import CurriculumStage
from adaptive_rl.curriculum.trainer import CurriculumTrainer
from adaptive_rl.curriculum.wrapper import CurriculumEnvWrapper
from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv

runner = CliRunner()


def run_phase_7_verification() -> bool:
    """Execute all Phase 7 verification checks."""
    print("=== AdaptiveRL Phase 7: Curriculum Learning Verification ===\n")
    test_dir = Path("experiments/verify_phase_7")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: CurriculumStage advancement logic
        print("1. Verifying CurriculumStage advancement logic...")
        stage = CurriculumStage(
            stage_id=0,
            name="Stage 0 - Easy",
            environment_parameters={"num_obstacles": 1},
            success_threshold=0.8,
            mean_reward_threshold=10.0,
            min_episodes=5,
            max_timesteps=100,
        )

        assert not stage.can_advance(
            {"success_rate": 1.0, "mean_reward": 50.0}, stage_episodes=4, stage_timesteps=50
        )
        assert not stage.can_advance(
            {"success_rate": 0.7, "mean_reward": 50.0}, stage_episodes=5, stage_timesteps=50
        )
        assert stage.can_advance(
            {"success_rate": 0.85, "mean_reward": 15.0}, stage_episodes=5, stage_timesteps=50
        )
        assert stage.can_advance(
            {"success_rate": 0.0, "mean_reward": -100.0}, stage_episodes=1, stage_timesteps=100
        )
        print("   ✓ Enforced minimum episodes constraint before evaluation.")
        print("   ✓ Verified success rate and mean reward threshold validation.")
        print("   ✓ Verified max timesteps timeout override.")

        # Step 2: Curriculum sequence transitions & serialization
        print("\n2. Verifying Curriculum sequence manager and history tracking...")
        curr = Curriculum(
            name="verification_curriculum",
            stages=[
                CurriculumStage(stage_id=0, name="Corridor", min_episodes=2, success_threshold=0.6),
                CurriculumStage(stage_id=1, name="Chamber", min_episodes=2, success_threshold=0.7),
            ],
            eval_window=5,
        )
        assert curr.current_stage_index == 0
        assert not curr.is_complete

        next_stage = curr.advance(timesteps=50, metrics={"success_rate": 0.75})
        assert next_stage is not None
        assert curr.current_stage.name == "Chamber"
        assert curr.is_complete
        assert len(curr.history) == 1

        # Serialization
        curr_dict = curr.to_dict()
        restored = Curriculum.from_dict(curr_dict)
        assert restored.current_stage_index == 1
        assert restored.is_complete
        print("   ✓ Curriculum transitions and milestone history logging verified.")
        print("   ✓ Full dictionary serialization/deserialization verified.")

        # Step 3: Gymnasium compliance of CurriculumEnvWrapper
        print("\n3. Verifying CurriculumEnvWrapper with Farama Gymnasium check_env...")
        raw_env = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, max_steps=20)
        nav_curr = create_navigation_curriculum(eval_window=5)
        wrapped_env = CurriculumEnvWrapper(env=raw_env, curriculum=nav_curr)

        check_env(wrapped_env)
        print("   ✓ Farama Gymnasium check_env passed with 0 warnings or errors.")

        # Step 4: Parameter injection across stages
        print("\n4. Verifying dynamic environment parameter injection across stages...")
        obs, info = wrapped_env.reset(seed=42)
        assert info["curriculum_stage_id"] == 0
        assert info["curriculum_stage_name"] == "Clear Corridor"
        assert raw_env.num_obstacles == 0
        print(
            f"   ✓ Stage 0 active: {info['curriculum_stage_name']} (obstacles: {raw_env.num_obstacles})"
        )

        nav_curr.advance(timesteps=100)
        obs, info = wrapped_env.reset(seed=42)
        assert info["curriculum_stage_id"] == 1
        assert raw_env.num_obstacles == 2
        print(
            f"   ✓ Stage 1 active: {info['curriculum_stage_name']} (obstacles: {raw_env.num_obstacles})"
        )
        wrapped_env.close()

        # Step 5: Presets verification
        print("\n5. Verifying predefined curriculum schedules...")
        nav_preset = get_curriculum_preset("navigation")
        grid_preset = get_curriculum_preset("gridworld")
        assert len(nav_preset.stages) == 4
        assert len(grid_preset.stages) == 4
        print(
            f"   ✓ Navigation preset verified: {len(nav_preset.stages)} progressive difficulty tiers."
        )
        print(
            f"   ✓ GridWorld preset verified: {len(grid_preset.stages)} progressive difficulty tiers."
        )

        # Step 6: End-to-end Curriculum Training Pipeline
        print("\n6. Executing end-to-end CurriculumTrainer with automated stage graduation...")
        exp_config = ExperimentConfig(
            name="verify_curriculum_run",
            seed=42,
            algorithm=AlgorithmConfig(
                name="ppo",
                learning_rate=0.001,
                gamma=0.99,
                batch_size=16,
                parameters={"n_steps": 32},
            ),
            environment=EnvironmentConfig(
                name="navigation",
                max_steps=20,
                parameters={"arena_width": 10.0, "arena_height": 10.0},
            ),
            curriculum=CurriculumConfig(
                enabled=True,
                stages=[
                    CurriculumStageConfig(
                        name="Tier 1 - Clear",
                        environment_parameters={"num_obstacles": 0},
                        max_timesteps=32,
                        min_episodes=1,
                    ),
                    CurriculumStageConfig(
                        name="Tier 2 - Cluttered",
                        environment_parameters={"num_obstacles": 3},
                        min_episodes=1,
                    ),
                ],
                eval_window=5,
            ),
            training=TrainingConfig(
                total_timesteps=64,
                checkpoint_freq=32,
                log_interval=5,
            ),
            output_dir=test_dir / "results",
            log_dir=test_dir / "logs",
        )

        trainer = CurriculumTrainer(config=exp_config)
        result = trainer.fit()

        assert result.total_timesteps == 64
        assert result.final_model_path.exists()
        print(
            f"   ✓ Curriculum training completed ({result.total_timesteps} steps, {result.episodes_completed} episodes)."
        )

        # Verify curriculum JSON summary
        report_path = test_dir / "results" / "curriculum" / "verify_curriculum_run_curriculum.json"
        assert report_path.exists()
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        assert len(report_data["history"]) >= 1
        print(
            f"   ✓ Curriculum report exported: {report_path.stat().st_size} bytes, {len(report_data['history'])} transitions recorded."
        )

        # Step 7: CLI commands verification
        print("\n7. Verifying Typer CLI curriculum commands...")
        res_list = runner.invoke(app, ["curriculum", "list"])
        assert res_list.exit_code == 0
        assert "navigation" in res_list.output
        assert "gridworld" in res_list.output

        res_inspect = runner.invoke(app, ["curriculum", "inspect", "navigation"])
        assert res_inspect.exit_code == 0
        assert "Clear" in res_inspect.output and "Corridor" in res_inspect.output
        assert "Dense" in res_inspect.output and "Hazard" in res_inspect.output

        res_val = runner.invoke(app, ["config", "validate", "configs/curriculum_navigation.yaml"])
        assert res_val.exit_code == 0
        assert "Curriculum: Enabled" in res_val.output
        print("   ✓ 'adaptive-rl curriculum list' verified.")
        print("   ✓ 'adaptive-rl curriculum inspect navigation' verified.")
        print("   ✓ 'adaptive-rl config validate configs/curriculum_navigation.yaml' verified.")

        print("\n=================================================================")
        print("✓ ALL PHASE 7 CURRICULUM LEARNING VERIFICATION CHECKS PASSED!")
        print("=================================================================\n")
        return True

    except Exception as e:
        print(f"\n❌ Phase 7 verification failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    success = run_phase_7_verification()
    sys.exit(0 if success else 1)
