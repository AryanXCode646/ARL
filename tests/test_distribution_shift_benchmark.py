"""Tests for the Issue #105 controlled distribution-shift benchmark.

Covers:
- Benchmark spec validation (seed overlap, TRAIN count, empty seeds, duplicate names,
  duplicate seeds within a scenario)
- Environment overrides actually changing dynamics (real registered environments)
- TRAIN-only training containment and frozen-policy evaluation
- Success/collision/reward extraction and recovery-time measurement/censoring
- Event-weighted, censoring-aware recovery aggregation (no false-positive gaps)
- Fixed cross-scenario event threshold with recorded provenance
- Raw per-episode records reconstructing every aggregate
- Preflight executing a real environment step
- Generalization gap math with explicit directionality
- Backward compatibility (generalization pipeline, drone defaults)
- CLI benchmark-shifts command and end-to-end JSON artifact smoke test
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import numpy as np
import pytest
from typer.testing import CliRunner

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.cli import app
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.environments import make_env
from adaptive_rl.evaluation.generalization import (
    GeneralizationDistribution,
    GeneralizationEvaluator,
    aggregate_seed_evaluation,
)
from adaptive_rl.evaluation.metrics import EvaluationMetrics
from adaptive_rl.evaluation.shift_benchmark import (
    DistributionShiftBenchmarkSpec,
    RecoverySummary,
    ShiftScenario,
    compute_scenario_gaps,
    load_shift_benchmark_config,
)
from adaptive_rl.experiments.shift_runner import DistributionShiftBenchmarkRunner
from adaptive_rl.metrics import EpisodeMetrics, compute_rate

runner = CliRunner()


def _gridworld_spec(
    train_seeds: List[int] = [101, 102, 103],
    test_a_seeds: List[int] = [201, 202, 203],
    test_b_seeds: List[int] = [301, 302, 303],
    train_overrides: Dict[str, Any] | None = None,
    test_a_overrides: Dict[str, Any] | None = None,
    test_b_overrides: Dict[str, Any] | None = None,
) -> DistributionShiftBenchmarkSpec:
    """Build a small gridworld shift spec with real environment parameter changes."""
    return DistributionShiftBenchmarkSpec(
        name="test_shift",
        environment_name="gridworld",
        base_environment_parameters={"width": 5, "height": 5},
        scenarios=[
            ShiftScenario(
                name="TRAIN",
                role="train",
                description="2 obstacles",
                seeds=train_seeds,
                environment_overrides=train_overrides or {"num_obstacles": 2},
            ),
            ShiftScenario(
                name="TEST-A",
                role="test",
                description="4 obstacles",
                seeds=test_a_seeds,
                environment_overrides=test_a_overrides or {"num_obstacles": 4},
            ),
            ShiftScenario(
                name="TEST-B",
                role="test",
                description="6 obstacles",
                seeds=test_b_seeds,
                environment_overrides=test_b_overrides or {"num_obstacles": 6},
            ),
        ],
    )


def _gridworld_experiment_config(tmp_path: Path, timesteps: int = 128) -> ExperimentConfig:
    """Build a tiny deterministic PPO/gridworld experiment config."""
    return ExperimentConfig(
        name="verify_shift_benchmark",
        seed=42,
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
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
            total_timesteps=timesteps,
            checkpoint_freq=0,
            log_interval=1,
        ),
        evaluation=EvaluationConfig(
            eval_episodes=3,
            deterministic=True,
        ),
    )


def test_shift_spec_rejects_train_test_seed_overlap() -> None:
    """TRAIN/TEST seed overlap is rejected (reuses GeneralizationDistribution)."""
    with pytest.raises(ValueError, match="overlapping seeds"):
        _gridworld_spec(train_seeds=[1, 2, 3], test_a_seeds=[3, 4, 5])


def test_shift_spec_requires_exactly_one_train_scenario() -> None:
    """Zero or multiple TRAIN scenarios are rejected."""
    base = dict(name="test_shift", environment_name="gridworld")
    no_train = dict(
        base,
        scenarios=[
            ShiftScenario(name="TEST-A", role="test", seeds=[1]),
            ShiftScenario(name="TEST-B", role="test", seeds=[2]),
        ],
    )
    with pytest.raises(ValueError, match="exactly one TRAIN"):
        DistributionShiftBenchmarkSpec(**no_train)  # type: ignore[arg-type]

    two_train = dict(
        base,
        scenarios=[
            ShiftScenario(name="TRAIN-1", role="train", seeds=[1]),
            ShiftScenario(name="TRAIN-2", role="train", seeds=[2]),
            ShiftScenario(name="TEST-A", role="test", seeds=[3]),
        ],
    )
    with pytest.raises(ValueError, match="exactly one TRAIN"):
        DistributionShiftBenchmarkSpec(**two_train)  # type: ignore[arg-type]

    no_test = dict(
        base,
        scenarios=[ShiftScenario(name="TRAIN", role="train", seeds=[1])],
    )
    with pytest.raises(ValueError, match="at least one TEST"):
        DistributionShiftBenchmarkSpec(**no_test)  # type: ignore[arg-type]


def test_shift_spec_rejects_empty_seed_lists() -> None:
    """Empty scenario seed lists are rejected."""
    with pytest.raises(ValueError, match="non-empty seed list"):
        _gridworld_spec(train_seeds=[])
    with pytest.raises(ValueError, match="non-empty seed list"):
        _gridworld_spec(test_a_seeds=[])


def test_shift_spec_rejects_duplicate_scenario_names() -> None:
    """Duplicate scenario names are rejected."""
    spec_data = dict(
        name="test_shift",
        environment_name="gridworld",
        scenarios=[
            ShiftScenario(name="TRAIN", role="train", seeds=[1]),
            ShiftScenario(name="TEST-A", role="test", seeds=[2]),
            ShiftScenario(name="TEST-A", role="test", seeds=[3]),
        ],
    )
    with pytest.raises(ValueError, match="unique"):
        DistributionShiftBenchmarkSpec(**spec_data)  # type: ignore[arg-type]


def test_environment_overrides_actually_applied() -> None:
    """Scenario overrides change real environment behavior (drone obstacle count, wind)."""
    from adaptive_rl.environments.drone.disturbed_drone import DroneDisturbance3DEnv

    train_env = DroneDisturbance3DEnv(
        num_obstacles=8,
        num_dynamic_obstacles=0,
        wind_speed=0.5,
        gust_sigma=0.0,  # deterministic drift comparison
    )
    test_env = DroneDisturbance3DEnv(
        num_obstacles=12,
        num_dynamic_obstacles=0,
        wind_speed=4.0,
        gust_sigma=0.0,
    )
    _, train_info = train_env.reset(seed=11)
    _, test_info = test_env.reset(seed=11)
    assert train_info["num_static_obstacles"] == 8
    assert test_info["num_static_obstacles"] == 12
    assert float(np.linalg.norm(train_env.wind_field.steady_wind)) == pytest.approx(0.5)
    assert float(np.linalg.norm(test_env.wind_field.steady_wind)) == pytest.approx(4.0)

    # Stronger steady wind produces measurably stronger downwind drift.
    zero = np.zeros(3, dtype=np.float32)
    for _ in range(10):
        _, _, _, _, train_info = train_env.step(zero)
        _, _, _, _, test_info = test_env.step(zero)
    assert test_info["position"][0] - train_info["position"][0] > 0.05
    train_env.close()
    test_env.close()


def test_train_env_uses_train_overrides_only() -> None:
    """Scenario parameter merge: base + TRAIN overrides, TEST values excluded."""
    spec = _gridworld_spec()
    merged = spec.scenario_environment_parameters(spec.train_scenario)
    assert merged["width"] == 5  # base preserved
    assert merged["num_obstacles"] == 2  # TRAIN override applied
    assert merged["num_obstacles"] != spec.test_scenarios[0].environment_overrides["num_obstacles"]


def test_test_env_uses_test_overrides_only() -> None:
    """Each TEST scenario merges base + its own overrides only."""
    spec = _gridworld_spec()
    merged_b = spec.scenario_environment_parameters(spec.test_scenarios[1])
    assert merged_b["width"] == 5
    assert merged_b["num_obstacles"] == 6
    assert merged_b["num_obstacles"] != spec.train_scenario.environment_overrides["num_obstacles"]


def test_unsupported_override_fails_clearly() -> None:
    """Unsupported environment parameters fail before execution, naming the scenario."""
    spec = _gridworld_spec(test_a_overrides={"wind_speed": 4.0})
    with pytest.raises(ValueError, match="TEST-A"):
        spec.validate_environments()


def test_end_to_end_smoke_produces_valid_json_artifact(tmp_path: Path) -> None:
    """Lightweight end-to-end run produces a valid JSON artifact on a real env."""
    spec = _gridworld_spec()
    config = _gridworld_experiment_config(tmp_path)
    report = DistributionShiftBenchmarkRunner(spec=spec).run(config)

    assert [s.scenario_name for s in report.scenarios] == ["TRAIN", "TEST-A", "TEST-B"]
    assert [s.role for s in report.scenarios] == ["train", "test", "test"]
    assert report.train_seeds == [101, 102, 103]
    assert report.test_seeds == [201, 202, 203, 301, 302, 303]
    assert report.total_training_timesteps == 128
    assert report.deterministic is True

    for res in report.scenarios:
        assert res.metrics.episodes == 3
        assert res.effective_environment_parameters["num_obstacles"] in (2, 4, 6)
    assert report.scenarios[0].gaps is None
    assert report.scenarios[1].gaps is not None
    assert report.scenarios[2].gaps is not None
    assert report.recovery_definition["unit"] == "environment_steps"

    artifact = (
        tmp_path / "results" / "distribution_shift" / "verify_shift_benchmark_shift_report.json"
    )
    assert artifact.exists()
    with open(artifact, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == "1.1"
    assert [s["scenario_name"] for s in data["scenarios"]] == ["TRAIN", "TEST-A", "TEST-B"]
    assert data["training_provenance"]["test_seed_contamination"] is False


def test_test_conditions_not_used_during_training(tmp_path: Path) -> None:
    """Training observes only TRAIN seeds and TRAIN environment parameters."""
    spec = _gridworld_spec()
    config = _gridworld_experiment_config(tmp_path)
    report = DistributionShiftBenchmarkRunner(spec=spec).run(config)

    prov = report.training_provenance
    sampled = prov["sampled_training_seeds"]
    assert len(sampled) > 0
    assert set(sampled).issubset(set(spec.train_seeds))
    assert set(sampled).isdisjoint(set(report.test_seeds))
    assert prov["training_environment_parameters"]["num_obstacles"] == 2


def test_same_frozen_policy_evaluated_across_all_scenarios(tmp_path: Path) -> None:
    """One fingerprint across scenarios proves a single frozen policy was evaluated."""
    spec = _gridworld_spec()
    config = _gridworld_experiment_config(tmp_path)
    report = DistributionShiftBenchmarkRunner(spec=spec).run(config)

    fingerprints = {res.policy_fingerprint for res in report.scenarios}
    assert len(fingerprints) == 1
    assert report.training_provenance["policy_fingerprint"] in fingerprints
    assert report.training_provenance["policy_frozen"] is True


def test_scenario_effective_parameters_present_in_report(tmp_path: Path) -> None:
    """Every scenario records the concrete parameters its environment was built with."""
    spec = _gridworld_spec()
    config = _gridworld_experiment_config(tmp_path)
    report = DistributionShiftBenchmarkRunner(spec=spec).run(config)

    expected = {"TRAIN": 2, "TEST-A": 4, "TEST-B": 6}
    for res in report.scenarios:
        eff = res.effective_environment_parameters
        assert eff["num_obstacles"] == expected[res.scenario_name]
        assert eff["width"] == 5
        assert eff["max_steps"] == 20


def test_success_collision_reward_correctly_extracted() -> None:
    """Scenario metrics match the canonical per-episode outcomes (no invented values)."""
    env = make_env("gridworld", width=5, height=5, num_obstacles=1, max_steps=20)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = GeneralizationEvaluator(algorithm=algo, env=env)
    metrics = evaluator.evaluate_seeds([5, 6, 7], deterministic=True)

    episodes = evaluator.last_episode_metrics
    assert metrics.episodes == 3
    assert metrics.success_rate == compute_rate([m.success for m in episodes])
    assert metrics.collision_rate == compute_rate([m.collision for m in episodes])
    assert metrics.mean_reward == pytest.approx(float(np.mean([m.reward for m in episodes])))
    assert metrics.additional_metrics["evaluated_seeds"] == [5, 6, 7]
    evaluator.close()


def test_recovery_time_measured_on_disturbance_and_recovery() -> None:
    """Exact recovery-time accounting: onset step 2, hold 3 -> recovery at step 6, time 4."""
    from adaptive_rl.environments.drone.disturbed_drone import DisturbanceRecoveryTracker

    tracker = DisturbanceRecoveryTracker()
    magnitudes = [0.2, 1.5, 1.6, 0.3, 0.2, 0.2, 0.2]
    speeds = [1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 1.0]
    teles = []
    for i, (mag, spd) in enumerate(zip(magnitudes, speeds), start=1):
        tracker.update(
            step=i,
            magnitude=mag,
            speed=spd,
            threshold=1.0,
            speed_tolerance=0.5,
            hold_steps=3,
        )
        teles.append(tracker.telemetry())

    assert teles[0]["disturbance_active"] is False
    assert teles[1]["disturbance_active"] is True
    assert teles[1]["disturbance_onset_step"] == 2
    assert teles[4]["disturbance_active"] is True  # hold not yet complete
    assert teles[4]["recovered"] is False
    assert teles[5]["recovered"] is True
    assert teles[5]["recovery_time"] == 4
    assert teles[5]["recovery_time"] >= 1  # never falsely zero
    assert teles[5]["episode_recovery_time"] == pytest.approx(4.0)
    assert teles[5]["recovery_events"] == 1
    assert teles[5]["recovery_completed"] == 1
    assert teles[5]["recovery_censored"] == 0


def test_recovery_unavailable_or_censored_never_zero() -> None:
    """Unrecovered events are censored (None), event-free episodes unavailable (None)."""
    from adaptive_rl.environments.drone.disturbed_drone import (
        DisturbanceRecoveryTracker,
        DroneDisturbance3DEnv,
    )

    # Censored: onset occurs but the episode ends before recovery completes.
    tracker = DisturbanceRecoveryTracker()
    tracker.update(
        step=1, magnitude=2.0, speed=1.0, threshold=1.0, speed_tolerance=0.5, hold_steps=5
    )
    tele = tracker.telemetry()
    assert tele["recovery_events"] == 1
    assert tele["recovery_completed"] == 0
    assert tele["recovery_censored"] == 1
    assert tele["episode_recovery_time"] is None

    # Unavailable: no stochastic disturbance configured -> no events, None (not 0.0).
    env = DroneDisturbance3DEnv(gust_sigma=0.0, disturbance_strength=0.0, max_steps=20)
    _, info = env.reset(seed=3)
    zero = np.zeros(3, dtype=np.float32)
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(zero)
        if terminated or truncated:
            break
    assert info["recovery_events"] == 0
    assert info["episode_recovery_time"] is None
    env.close()


def test_recovery_telemetry_consistent_on_real_disturbed_episodes() -> None:
    """Real disturbed episodes: completed recoveries average exactly the recorded times."""
    from adaptive_rl.environments.drone.disturbed_drone import DroneDisturbance3DEnv

    env = DroneDisturbance3DEnv(
        num_obstacles=8,
        num_dynamic_obstacles=0,
        wind_speed=0.5,
        gust_sigma=0.15,
        disturbance_strength=6.0,
        disturbance_event_factor=1.5,
        max_steps=300,
    )
    zero = np.zeros(3, dtype=np.float32)
    for seed in range(8):
        _, info = env.reset(seed=seed)
        for _ in range(300):
            _, _, terminated, truncated, info = env.step(zero)
            if terminated or truncated:
                break
        if info["recovery_completed"] > 0:
            assert info["episode_recovery_time"] is not None
            assert info["episode_recovery_time"] == pytest.approx(
                float(np.mean(info["recovery_times"]))
            )
            assert info["episode_recovery_time"] >= 5  # hold_steps lower bound
        else:
            assert info["episode_recovery_time"] is None
    env.close()


def test_generalization_gaps_mathematically_correct() -> None:
    """Gap directionality: positive means TEST worse; recovery time gap needs full completion."""

    def _metrics(
        success: float | None,
        collision: float | None,
        reward: float,
        recovery: float | None,
        completion: float | None,
        censoring: float | None,
    ) -> EvaluationMetrics:
        return EvaluationMetrics(
            episodes=4,
            mean_reward=reward,
            success_rate=success,
            collision_rate=collision,
            recovery_time=recovery,
            recovery_events=10 if completion is not None else None,
            recovery_completed_events=(int(10 * completion) if completion is not None else None),
            recovery_censored_events=int(10 * censoring) if censoring is not None else None,
            recovery_completion_rate=completion,
            recovery_censoring_rate=censoring,
            mean_episode_length=10.0,
        )

    # Both sides fully recovered: time gap defined (test - train).
    gaps = compute_scenario_gaps(
        _metrics(0.8, 0.1, 10.0, 5.0, 1.0, 0.0),
        _metrics(0.5, 0.4, 4.0, 12.0, 1.0, 0.0),
    )
    assert gaps.success_gap == pytest.approx(0.3)  # train - test
    assert gaps.reward_gap == pytest.approx(6.0)  # train - test
    assert gaps.collision_gap == pytest.approx(0.3)  # test - train
    assert gaps.recovery_time_gap == pytest.approx(7.0)  # test - train
    assert gaps.recovery_completion_gap == pytest.approx(0.0)
    assert gaps.recovery_censoring_gap == pytest.approx(0.0)

    # TEST censored: time gap suppressed, completion/censoring gaps carry the signal.
    gaps_cens = compute_scenario_gaps(
        _metrics(0.8, 0.1, 10.0, 5.0, 1.0, 0.0),
        _metrics(0.5, 0.4, 4.0, 4.0, 0.2, 0.8),
    )
    assert gaps_cens.recovery_time_gap is None
    assert gaps_cens.recovery_completion_gap == pytest.approx(0.8)  # train - test
    assert gaps_cens.recovery_censoring_gap == pytest.approx(0.8)  # test - train

    # Unavailable inputs propagate None instead of collapsing into 0.0.
    gaps_none = compute_scenario_gaps(
        _metrics(None, None, 10.0, None, None, None),
        _metrics(0.5, 0.4, 4.0, 12.0, 1.0, 0.0),
    )
    assert gaps_none.success_gap is None
    assert gaps_none.collision_gap is None
    assert gaps_none.recovery_time_gap is None
    assert gaps_none.recovery_completion_gap is None
    assert gaps_none.recovery_censoring_gap is None
    assert gaps_none.reward_gap == pytest.approx(6.0)


def test_existing_generalization_behavior_compatible() -> None:
    """The pre-existing generalization pipeline still behaves as before (plus recovery=None)."""
    env = make_env("gridworld", width=5, height=5, num_obstacles=1, max_steps=20)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    evaluator = GeneralizationEvaluator(algorithm=algo, env=env)
    distribution = GeneralizationDistribution(train_seeds=[10, 11], test_seeds=[20, 21])
    report = evaluator.evaluate_generalization(distribution, experiment_name="compat")
    assert report.train_metrics.episodes == 2
    assert report.test_metrics.episodes == 2
    assert report.generalization_gap_reward == pytest.approx(
        report.train_metrics.mean_reward - report.test_metrics.mean_reward
    )
    # Gridworld exposes no disturbance telemetry: recovery stays unavailable, not zero.
    assert report.train_metrics.recovery_time is None
    assert report.test_metrics.recovery_time is None
    evaluator.close()


def test_existing_drone_defaults_unchanged() -> None:
    """New optional parameters default to exact legacy behavior."""
    from adaptive_rl.environments.drone.disturbed_drone import DroneDisturbance3DEnv

    env = DroneDisturbance3DEnv()
    assert env.num_obstacles == 6
    assert env.num_dynamic_obstacles == 3
    assert env.wind_speed is None
    assert env.disturbance_strength == 0.0
    assert env.disturbance_theta == 0.15
    assert env.disturbance_event_factor == 1.5
    assert env.recovery_speed_tolerance == 1.0
    assert env.recovery_hold_steps == 5
    assert list(env.wind_field.steady_wind) == [1.5, 0.5, 0.0]
    assert env.wind_field.gust_sigma == 0.4
    assert env.observation_space.shape == (33,)

    env.reset(seed=42)
    zero = np.zeros(3, dtype=np.float32)
    for _ in range(10):
        env.step(zero)
    assert np.all(env._injected_disturbance == 0.0)

    # Legacy curriculum parameter hooks keep working.
    env.set_parameters(wind_base_speed=2.0, wind_gusts=False, battery_capacity=50.0)
    assert env.wind_base_speed == pytest.approx(2.0 * np.sqrt(1.09))
    assert env.wind_gusts is False
    assert env.battery_capacity == pytest.approx(50.0)
    env.close()


def test_example_config_loads_and_validates() -> None:
    """The shipped Issue #105 config loads and every scenario environment validates."""
    config_path = Path("configs/drone_distribution_shift.yaml")
    exp_config, spec = load_shift_benchmark_config(config_path)
    assert exp_config.name == "drone_distribution_shift"
    assert [s.name for s in spec.scenarios] == ["TRAIN", "TEST-A", "TEST-B", "TEST-C", "TEST-D"]
    assert spec.train_scenario.environment_overrides["num_obstacles"] == 8
    effective = spec.validate_environments()
    assert set(effective) == {"TRAIN", "TEST-A", "TEST-B", "TEST-C", "TEST-D"}
    assert effective["TEST-D"]["disturbance_strength"] == 6.0
    assert effective["TEST-C"]["num_dynamic_obstacles"] == 4


def test_cli_benchmark_shifts_command(tmp_path: Path) -> None:
    """The benchmark-shifts CLI runs end-to-end on a small derived config."""
    import yaml

    with open("configs/drone_distribution_shift.yaml", "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # Derive a fast smoke config from the real example (same loader, same shape).
    raw["training"]["total_timesteps"] = 64
    raw["training"]["checkpoint_freq"] = 0
    raw["output_dir"] = str(tmp_path / "out")
    raw["log_dir"] = str(tmp_path / "logs")
    for scenario in raw["benchmark"]["scenarios"]:
        scenario["seeds"] = scenario["seeds"][:2]
    cfg_file = tmp_path / "shift_smoke.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f)

    out_report = tmp_path / "shift_cli_report.json"
    res = runner.invoke(
        app,
        [
            "benchmark-shifts",
            "--config",
            str(cfg_file),
            "--output-report",
            str(out_report),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Distribution-Shift Results" in res.output
    assert "TRAIN" in res.output and "TEST-D" in res.output
    assert out_report.exists()
    with open(out_report, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert [s["scenario_name"] for s in data["scenarios"]] == [
        "TRAIN",
        "TEST-A",
        "TEST-B",
        "TEST-C",
        "TEST-D",
    ]


def _episode(
    reward: float,
    success: bool | None,
    collision: bool | None,
    times: List[int],
    events: int,
    censored: int,
    length: int = 10,
) -> EpisodeMetrics:
    """Build a synthetic canonical episode with drone-style recovery telemetry."""
    mean_time = float(sum(times) / len(times)) if times else None
    return EpisodeMetrics(
        reward=reward,
        length=length,
        success=success,
        collision=collision,
        terminated=True,
        truncated=False,
        additional_metrics={
            "recovery_times": list(times),
            "recovery_events": events,
            "recovery_completed": len(times),
            "recovery_censored": censored,
            "episode_recovery_time": mean_time,
        },
    )


def test_recovery_time_is_event_weighted() -> None:
    """Episodes with many events weigh proportionally more (no mean-of-means)."""
    episodes = [
        _episode(1.0, True, False, [30], 1, 0),
        _episode(2.0, True, False, [5] * 10, 10, 0),
    ]
    metrics = aggregate_seed_evaluation(episodes, [1, 2])
    # Event-weighted: (30 + 10*5) / 11 = 80/11; mean-of-means would be 17.5.
    assert metrics.recovery_time == pytest.approx(80.0 / 11.0)
    assert metrics.recovery_time != pytest.approx(17.5)
    assert metrics.recovery_events == 11
    assert metrics.recovery_completed_events == 11
    assert metrics.recovery_censored_events == 0
    assert metrics.recovery_completion_rate == pytest.approx(1.0)
    assert metrics.recovery_censoring_rate == pytest.approx(0.0)


def test_recovery_censoring_does_not_create_false_positive_gap() -> None:
    """A censored TEST with a low conditional mean must not look spuriously faster."""
    train_episodes = [_episode(5.0, True, False, [10] * 10, 10, 0)]
    test_episodes = [_episode(1.0, False, True, [4, 4], 10, 8)]
    train_metrics = aggregate_seed_evaluation(train_episodes, [1])
    test_metrics = aggregate_seed_evaluation(test_episodes, [2])

    assert train_metrics.recovery_time == pytest.approx(10.0)
    assert test_metrics.recovery_time == pytest.approx(4.0)  # conditional only
    assert test_metrics.recovery_completion_rate == pytest.approx(0.2)
    assert test_metrics.recovery_censoring_rate == pytest.approx(0.8)

    gaps = compute_scenario_gaps(train_metrics, test_metrics)
    assert gaps.recovery_time_gap is None  # 4.0 < 10.0 must never read as "better"
    assert gaps.recovery_completion_gap == pytest.approx(0.8)  # TEST worse
    assert gaps.recovery_censoring_gap == pytest.approx(0.8)  # TEST worse


def test_recovery_completion_rate_is_reported() -> None:
    """Completion/censoring rates and counts are first-class scenario metrics."""
    episodes = [
        _episode(1.0, True, False, [6, 8], 2, 0),
        _episode(2.0, False, True, [], 1, 1),
        _episode(3.0, False, False, [], 0, 0),
    ]
    metrics = aggregate_seed_evaluation(episodes, [1, 2, 3])
    assert metrics.recovery_events == 3
    assert metrics.recovery_completed_events == 2
    assert metrics.recovery_censored_events == 1
    assert metrics.recovery_completion_rate == pytest.approx(2.0 / 3.0)
    assert metrics.recovery_censoring_rate == pytest.approx(1.0 / 3.0)
    assert metrics.recovery_time == pytest.approx(7.0)
    assert metrics.additional_metrics["recovery_episodes_measured"] == 1
    assert metrics.additional_metrics["recovery_episodes_unavailable"] == 1
    assert metrics.additional_metrics["recovery_episodes_censored"] == 1

    summary = RecoverySummary.from_metrics(metrics)
    assert summary.total_events == 3
    assert summary.completed_events == 2
    assert summary.censored_events == 1
    assert summary.completion_rate == pytest.approx(2.0 / 3.0)
    assert summary.censoring_rate == pytest.approx(1.0 / 3.0)
    assert summary.mean_completed_recovery_time == pytest.approx(7.0)


def test_recovery_time_gap_none_when_test_has_censoring() -> None:
    """Time gap requires 100% completion on both sides; completion gap stays defined."""
    train_metrics = aggregate_seed_evaluation([_episode(1.0, True, False, [9], 1, 0)], [1])
    test_metrics = aggregate_seed_evaluation(
        [_episode(1.0, True, False, [9], 1, 0), _episode(0.0, False, True, [], 1, 1)],
        [2, 3],
    )
    assert test_metrics.recovery_completion_rate == pytest.approx(0.5)
    gaps = compute_scenario_gaps(train_metrics, test_metrics)
    assert gaps.recovery_time_gap is None
    assert gaps.recovery_completion_gap == pytest.approx(0.5)
    assert gaps.recovery_censoring_gap == pytest.approx(0.5)


def test_fixed_recovery_threshold_is_identical_across_scenarios() -> None:
    """The benchmark fixed threshold reaches every scenario with the same value."""
    _, spec = load_shift_benchmark_config("configs/drone_distribution_shift.yaml")
    assert spec.recovery_event_threshold == pytest.approx(0.7)
    from adaptive_rl.environments.drone.disturbed_drone import DroneDisturbance3DEnv

    for scenario in spec.scenarios:
        params = spec.effective_scenario_parameters(scenario)
        assert params["disturbance_event_threshold_override"] == pytest.approx(0.7)
        env = DroneDisturbance3DEnv(**{k: v for k, v in params.items() if k != "max_steps"})
        assert env.disturbance_event_threshold == pytest.approx(0.7)
        env.close()


def test_recovery_threshold_is_recorded_in_effective_parameters() -> None:
    """Effective parameters record both the threshold value and its source."""
    from adaptive_rl.environments.drone.disturbed_drone import DroneDisturbance3DEnv

    fixed = DroneDisturbance3DEnv(disturbance_event_threshold_override=0.7)
    eff = fixed.get_effective_parameters()
    assert eff["disturbance_event_threshold"] == pytest.approx(0.7)
    assert eff["disturbance_event_threshold_source"] == "benchmark_override"
    fixed.close()

    derived = DroneDisturbance3DEnv()
    eff_derived = derived.get_effective_parameters()
    assert eff_derived["disturbance_event_threshold_source"] == "derived"
    assert eff_derived["disturbance_event_threshold"] == pytest.approx(
        derived.disturbance_event_threshold
    )
    # Legacy derived behavior is unchanged when the override is absent.
    import math

    gust_stat = 0.4 / math.sqrt(2.0 * 0.15)
    assert eff_derived["disturbance_event_threshold"] == pytest.approx(
        1.5 * math.sqrt(3.0 * gust_stat**2)
    )
    derived.close()


def test_per_episode_records_are_written_to_report(tmp_path: Path) -> None:
    """Every scenario stores one raw record per evaluated seed, in seed order."""
    spec = _gridworld_spec()
    config = _gridworld_experiment_config(tmp_path)
    report = DistributionShiftBenchmarkRunner(spec=spec).run(config)

    for res in report.scenarios:
        assert len(res.episodes) == len(res.seeds)
        assert [rec.seed for rec in res.episodes] == res.seeds
        for rec in res.episodes:
            assert rec.recovery_completed == len(rec.recovery_times)
            assert rec.recovery_events == rec.recovery_completed + rec.recovery_censored


def test_per_episode_records_reconstruct_aggregate_metrics(tmp_path: Path) -> None:
    """Aggregates recomputed from raw records match the reported scenario metrics."""
    spec = _gridworld_spec()
    config = _gridworld_experiment_config(tmp_path)
    report = DistributionShiftBenchmarkRunner(spec=spec).run(config)

    for res in report.scenarios:
        recs = res.episodes
        assert res.metrics.success_rate == compute_rate([r.success for r in recs])
        assert res.metrics.collision_rate == compute_rate([r.collision for r in recs])
        assert res.metrics.mean_reward == pytest.approx(float(np.mean([r.reward for r in recs])))
        all_times = [t for r in recs for t in r.recovery_times]
        if all_times:
            assert res.metrics.recovery_time == pytest.approx(float(np.mean(all_times)))
            assert res.recovery.total_events == sum(r.recovery_events for r in recs)
            assert res.recovery.completed_events == len(all_times)
        else:
            assert res.metrics.recovery_time is None
            assert res.recovery.total_events == 0
            assert res.recovery.completion_rate is None


def test_preflight_executes_real_environment_step() -> None:
    """Preflight validation executes step() dynamics (spy passes through to real code)."""
    from adaptive_rl.environments.drone.disturbed_drone import DroneDisturbance3DEnv

    spec = DistributionShiftBenchmarkSpec(
        name="preflight_probe",
        environment_name="drone_disturbed",
        scenarios=[
            ShiftScenario(
                name="TRAIN",
                role="train",
                seeds=[11],
                environment_overrides={"num_obstacles": 8, "num_dynamic_obstacles": 0},
            ),
            ShiftScenario(
                name="TEST-A",
                role="test",
                seeds=[22],
                environment_overrides={"num_obstacles": 10, "num_dynamic_obstacles": 0},
            ),
        ],
    )
    calls: List[int] = []
    original_step = DroneDisturbance3DEnv.step

    def spy_step(self: DroneDisturbance3DEnv, action: Any) -> Any:
        calls.append(1)
        return original_step(self, action)

    with mock.patch.object(DroneDisturbance3DEnv, "step", spy_step):
        effective = spec.validate_environments(max_steps=50)
    assert len(calls) == 2  # one real step per scenario
    assert set(effective) == {"TRAIN", "TEST-A"}

    # A construct-time failure names the offending scenario.
    bad = DistributionShiftBenchmarkSpec(
        name="preflight_bad",
        environment_name="drone_disturbed",
        scenarios=[
            ShiftScenario(name="TRAIN", role="train", seeds=[11]),
            ShiftScenario(
                name="TEST-BROKEN",
                role="test",
                seeds=[22],
                environment_overrides={"bounds": [-1.0, 5.0, 5.0]},
            ),
        ],
    )
    with pytest.raises(ValueError, match="TEST-BROKEN"):
        bad.validate_environments(max_steps=50)


def test_duplicate_seeds_within_scenario_are_rejected() -> None:
    """Seed lists must contain unique integers inside each scenario."""
    with pytest.raises(ValueError, match="must be unique"):
        _gridworld_spec(train_seeds=[100, 100, 101])
    with pytest.raises(ValueError, match="must be unique"):
        _gridworld_spec(test_a_seeds=[200, 201, 200])
