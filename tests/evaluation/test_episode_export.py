# -*- coding: utf-8 -*-
"""Comprehensive behavior tests for episode-level evaluation export.

Verifies:
* Normal evaluation: exactly one record per episode matching canonical EpisodeMetrics.
* Multiple scenarios: records from all scenarios aggregated into parent evaluator.
* Record reset: evaluator resets records on subsequent evaluation passes without leakage.
* Environment ownership: reused environments are not closed, new override environments are closed in finally.
* Export schema: JSON schema and CSV column order, formats equivalence.
* Validation and error handling: empty records, unsupported extensions, nested paths, malformed records.
* Deterministic reproducibility: identical seeds produce identical episodic metrics.
* Public API: EpisodeEvaluationRecord import and factory method.
* CLI integration: --episode-report flag in JSON and CSV formats.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.cli import app
from adaptive_rl.environments.gridworld.grid import GridWorldEnv
from adaptive_rl.environments.testing import DummyTestEnv
from adaptive_rl.evaluation import (
    EpisodeEvaluationRecord,
    EvaluationMetrics,
    EvaluationScenario,
    Evaluator,
)
from adaptive_rl.evaluation.evaluator import EPISODE_RECORD_FIELDS
from adaptive_rl.evaluation.seeding import derive_evaluation_seed
from adaptive_rl.metrics import EpisodeMetrics


def _create_evaluator(
    step_limit: int = 5,
    reward_step: float = 1.0,
    seed: int = 123,
) -> tuple[Evaluator, DummyTestEnv]:
    env = DummyTestEnv(step_limit=step_limit, reward_step=reward_step)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=seed)
    evaluator = Evaluator(algorithm=algo, env=env)
    return evaluator, env


def test_public_api_and_data_model() -> None:
    """Verify EpisodeEvaluationRecord is public and has canonical conversion."""
    m = EpisodeMetrics(
        reward=42.5,
        length=10,
        success=True,
        collision=False,
        terminated=True,
        truncated=False,
    )
    record = EpisodeEvaluationRecord.from_episode_metrics(
        m,
        episode_index=3,
        seed=1003,
        environment="gridworld",
        scenario="dense_obstacles",
    )

    assert record.episode_index == 3
    assert record.seed == 1003
    assert record.environment == "gridworld"
    assert record.scenario == "dense_obstacles"
    assert record.return_value == 42.5
    assert record.episode_length == 10
    assert record.success is True
    assert record.collision is False

    d = record.to_dict()
    assert d["episode_index"] == 3
    assert d["seed"] == 1003
    assert d["environment"] == "gridworld"
    assert d["scenario"] == "dense_obstacles"
    assert d["return"] == 42.5
    assert d["episode_length"] == 10
    assert d["success"] is True
    assert d["collision"] is False
    assert tuple(d.keys()) == EPISODE_RECORD_FIELDS


def test_normal_evaluation_episode_records() -> None:
    """Verify exactly one record per episode with canonical EpisodeMetrics correspondence."""
    evaluator, env = _create_evaluator(step_limit=6, reward_step=2.0)
    try:
        metrics = evaluator.evaluate(num_episodes=4, deterministic=True, base_seed=77)
        records = evaluator.last_episode_records

        assert len(records) == 4
        assert len(evaluator.last_episode_metrics) == 4

        for ep, (rec, ep_m) in enumerate(zip(records, evaluator.last_episode_metrics)):
            assert isinstance(rec, EpisodeEvaluationRecord)
            assert rec.episode_index == ep
            assert rec.seed == derive_evaluation_seed(77, ep)
            assert rec.environment == "DummyTestEnv"
            assert rec.scenario is None
            assert rec.return_value == ep_m.reward == 12.0  # 6 steps * 2.0
            assert rec.episode_length == ep_m.length == 6
            assert rec.success == ep_m.success
            assert rec.collision == ep_m.collision

        assert metrics.episodes == 4
    finally:
        env.close()


def test_multiple_scenarios_record_propagation() -> None:
    """Verify parent evaluator aggregates records across multiple scenarios."""
    evaluator, env = _create_evaluator(step_limit=5)
    try:
        scenarios = [
            EvaluationScenario(name="scenario_a", seed=111, environment_overrides={}),
            EvaluationScenario(name="scenario_b", seed=222, environment_overrides={}),
        ]
        results = evaluator.evaluate_scenarios(
            scenarios=scenarios,
            deterministic=True,
            num_episodes_per_scenario=2,
        )

        assert isinstance(results, dict)
        assert set(results.keys()) == {"scenario_a", "scenario_b"}
        assert all(isinstance(m, EvaluationMetrics) for m in results.values())

        records = evaluator.last_episode_records
        # 2 scenarios * 2 episodes = 4 records
        assert len(records) == 4

        # Scenario A records
        rec_a = records[:2]
        assert all(r.scenario == "scenario_a" for r in rec_a)
        assert [r.episode_index for r in rec_a] == [0, 1]
        assert rec_a[0].seed == derive_evaluation_seed(111, 0)
        assert rec_a[1].seed == derive_evaluation_seed(111, 1)

        # Scenario B records
        rec_b = records[2:]
        assert all(r.scenario == "scenario_b" for r in rec_b)
        assert [r.episode_index for r in rec_b] == [0, 1]
        assert rec_b[0].seed == derive_evaluation_seed(222, 0)
        assert rec_b[1].seed == derive_evaluation_seed(222, 1)
    finally:
        env.close()


def test_record_reset_between_evaluations() -> None:
    """Verify evaluator resets last_episode_records on every evaluate call."""
    evaluator, env = _create_evaluator(step_limit=3)
    try:
        evaluator.evaluate(num_episodes=5, deterministic=True, base_seed=10)
        assert len(evaluator.last_episode_records) == 5

        # Second evaluation with fewer episodes should not contain stale records
        evaluator.evaluate(num_episodes=2, deterministic=True, base_seed=20)
        assert len(evaluator.last_episode_records) == 2
        assert [r.episode_index for r in evaluator.last_episode_records] == [0, 1]

        # Scenario evaluation also resets parent records
        scenario = EvaluationScenario(name="reset_test", seed=30, environment_overrides={})
        evaluator.evaluate_scenarios([scenario], deterministic=True)
        assert len(evaluator.last_episode_records) == 1
        assert evaluator.last_episode_records[0].scenario == "reset_test"
    finally:
        env.close()


def test_environment_ownership_reuse_and_cleanup() -> None:
    """Verify reused env is not closed, and scenario-specific override env is closed."""
    # 1. Reused environment without overrides
    env = DummyTestEnv(step_limit=4)
    env.close = MagicMock(wraps=env.close)  # type: ignore[assignment]
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32)
    evaluator = Evaluator(algorithm=algo, env=env)

    scenario_no_overrides = EvaluationScenario(
        name="no_override", seed=42, environment_overrides={}
    )
    evaluator.evaluate_scenarios([scenario_no_overrides], deterministic=True)
    # The reused environment must NOT have been closed
    env.close.assert_not_called()

    # 2. Environment with overrides on registered environment
    grid_env = GridWorldEnv(width=5, height=5, num_obstacles=1, max_steps=10)
    grid_algo = PPOAlgorithm(env=grid_env, n_steps=64, batch_size=32)
    grid_evaluator = Evaluator(
        algorithm=grid_algo,
        env=grid_env,
        env_name="gridworld",
        env_kwargs={"width": 5, "height": 5},
    )

    scenario_override = EvaluationScenario(
        name="with_override", seed=55, environment_overrides={"num_obstacles": 3}
    )
    grid_evaluator.evaluate_scenarios([scenario_override], deterministic=True)

    # 3. Verify failure during evaluation closes newly created environment
    failing_scenario = EvaluationScenario(
        name="failing_scenario", seed=66, environment_overrides={"num_obstacles": 2}
    )
    failing_algo = MagicMock(spec=PPOAlgorithm)
    failing_algo.predict.side_effect = RuntimeError("Simulated agent prediction crash")

    failing_evaluator = Evaluator(
        algorithm=failing_algo,
        env=grid_env,
        env_name="gridworld",
        env_kwargs={"width": 5, "height": 5},
    )

    with pytest.raises(RuntimeError, match="Simulated agent prediction crash"):
        failing_evaluator.evaluate_scenarios([failing_scenario], deterministic=True)

    grid_env.close()
    env.close()


def test_export_formats_and_schema_equivalence(tmp_path: Path) -> None:
    """Verify JSON schema and CSV header order with identical underlying values."""
    evaluator, env = _create_evaluator(step_limit=5)
    try:
        evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=42)
        json_file = tmp_path / "reports" / "episodes.json"
        csv_file = tmp_path / "reports" / "episodes.csv"

        res_json = Evaluator.save_episode_report(evaluator.last_episode_records, json_file)
        res_csv = Evaluator.save_episode_report(evaluator.last_episode_records, csv_file)

        assert res_json == json_file
        assert res_csv == csv_file
        assert json_file.exists()
        assert csv_file.exists()

        # Validate JSON content
        json_data = json.loads(json_file.read_text(encoding="utf-8"))
        assert isinstance(json_data, list)
        assert len(json_data) == 3
        for row in json_data:
            assert set(row.keys()) == set(EPISODE_RECORD_FIELDS)
            assert isinstance(row["episode_index"], int)
            assert isinstance(row["seed"], int)
            assert isinstance(row["environment"], str)
            assert row["scenario"] is None
            assert isinstance(row["return"], float)
            assert isinstance(row["episode_length"], int)

        # Validate CSV content
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert tuple(reader.fieldnames or []) == EPISODE_RECORD_FIELDS
            csv_rows = list(reader)

        assert len(csv_rows) == len(json_data) == 3
        for j_row, c_row in zip(json_data, csv_rows):
            assert int(c_row["episode_index"]) == j_row["episode_index"]
            assert int(c_row["seed"]) == j_row["seed"]
            assert c_row["environment"] == j_row["environment"]
            assert (c_row["scenario"] or None) == j_row["scenario"]
            assert float(c_row["return"]) == pytest.approx(j_row["return"])
            assert int(c_row["episode_length"]) == j_row["episode_length"]
    finally:
        env.close()


def test_save_episode_report_validation_and_edge_cases(tmp_path: Path) -> None:
    """Verify handling of empty records, nested paths, unsupported extensions, and malformed inputs."""
    # 1. Empty records
    empty_json = tmp_path / "empty.json"
    empty_csv = tmp_path / "empty.csv"
    Evaluator.save_episode_report([], empty_json)
    Evaluator.save_episode_report([], empty_csv)

    assert json.loads(empty_json.read_text(encoding="utf-8")) == []
    with open(empty_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert tuple(header) == EPISODE_RECORD_FIELDS
        assert list(reader) == []

    # 2. Unsupported extension
    with pytest.raises(ValueError, match="Unsupported file extension"):
        Evaluator.save_episode_report([], tmp_path / "report.parquet")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        Evaluator.save_episode_report([], tmp_path / "report.txt")

    # 3. Deeply nested output path
    nested_path = tmp_path / "a" / "b" / "c" / "episodes.json"
    Evaluator.save_episode_report([], nested_path)
    assert nested_path.exists()

    # 4. Malformed record (dict missing required keys)
    malformed_records: list[dict[str, Any]] = [
        {"episode_index": 0, "seed": 42}  # missing return, episode_length, etc.
    ]
    with pytest.raises(ValueError, match="missing required field"):
        Evaluator.save_episode_report(malformed_records, tmp_path / "out.json")

    # 5. Invalid record type
    with pytest.raises(TypeError, match="must be EpisodeEvaluationRecord or dict"):
        Evaluator.save_episode_report(["not_a_record"], tmp_path / "out.json")  # type: ignore[list-item]

    # 6. Valid dict input
    valid_dict_records = [
        {
            "episode_index": 0,
            "seed": 10,
            "environment": "dummy",
            "scenario": None,
            "return": 5.0,
            "episode_length": 5,
            "success": True,
            "collision": False,
        }
    ]
    dict_out = tmp_path / "dict_out.json"
    Evaluator.save_episode_report(valid_dict_records, dict_out)
    assert dict_out.exists()
    assert json.loads(dict_out.read_text(encoding="utf-8")) == valid_dict_records


def test_deterministic_reproducibility() -> None:
    """Verify identical base_seed produces identical episodic records."""
    eval1, env1 = _create_evaluator(step_limit=5)
    eval2, env2 = _create_evaluator(step_limit=5)
    try:
        eval1.evaluate(num_episodes=3, deterministic=True, base_seed=999)
        eval2.evaluate(num_episodes=3, deterministic=True, base_seed=999)

        records1 = eval1.last_episode_records
        records2 = eval2.last_episode_records

        assert len(records1) == len(records2) == 3
        for r1, r2 in zip(records1, records2):
            assert r1.episode_index == r2.episode_index
            assert r1.seed == r2.seed
            assert r1.return_value == r2.return_value
            assert r1.episode_length == r2.episode_length
            assert r1.success == r2.success
            assert r1.collision == r2.collision
    finally:
        env1.close()
        env2.close()


def test_cli_episode_report_integration(tmp_path: Path) -> None:
    """Verify CLI --episode-report exports valid JSON and CSV without a second evaluation pass."""
    runner = CliRunner()

    env = GridWorldEnv(width=4, height=4, num_obstacles=1, max_steps=10)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    algo.train(64)
    model_path = tmp_path / "model.zip"
    algo.save(model_path)
    env.close()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: "cli_episode_test"
seed: 42
algorithm:
  name: "ppo"
environment:
  name: "gridworld"
  max_steps: 10
  parameters:
    width: 4
    height: 4
    num_obstacles: 1
training:
  total_timesteps: 64
evaluation:
  eval_episodes: 2
  deterministic: true
""",
        encoding="utf-8",
    )

    json_report = tmp_path / "episodes.json"
    csv_report = tmp_path / "episodes.csv"
    output_report = tmp_path / "eval_report.json"

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(config_path),
            "--model",
            str(model_path),
            "--episodes",
            "2",
            "--output-report",
            str(output_report),
            "--episode-report",
            str(json_report),
        ],
    )
    assert result.exit_code == 0
    assert "Episode report saved to:" in result.output
    assert json_report.exists()
    assert output_report.exists()

    records = json.loads(json_report.read_text(encoding="utf-8"))
    assert len(records) == 2
    assert records[0]["episode_index"] == 0
    assert records[1]["episode_index"] == 1

    # Test short option -er with CSV
    result_csv = runner.invoke(
        app,
        [
            "evaluate",
            "--config",
            str(config_path),
            "--model",
            str(model_path),
            "--episodes",
            "2",
            "-er",
            str(csv_report),
        ],
    )
    assert result_csv.exit_code == 0
    assert csv_report.exists()
    with open(csv_report, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or []) == EPISODE_RECORD_FIELDS
        assert len(list(reader)) == 2
