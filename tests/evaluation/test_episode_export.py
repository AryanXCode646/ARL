# -*- coding: utf-8 -*-
"""Tests for episode‑level export functionality.

The tests verify:
* One record per episode.
* Correct JSON schema.
* CSV header order and row count.
* Deterministic seed preservation.
* Scenario name propagation when using `evaluate_scenarios`.
"""

import csv
import json
from pathlib import Path

import pytest

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.environments.testing import DummyTestEnv
from adaptive_rl.evaluation.evaluator import Evaluator


def _run_evaluator(num_episodes: int, base_seed: int | None = 42, scenario_name: str | None = None):
    env = DummyTestEnv(step_limit=5, reward_step=1.0)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=123)
    evaluator = Evaluator(algorithm=algo, env=env)
    if scenario_name:
        evaluator._current_scenario_name = scenario_name
    metrics = evaluator.evaluate(num_episodes=num_episodes, deterministic=True, base_seed=base_seed)
    return evaluator, metrics, env


@pytest.mark.parametrize("fmt, ext", [("json", ".json"), ("csv", ".csv")])
def test_episode_export_formats(tmp_path: Path, fmt: str, ext: str):
    evaluator, metrics, env = _run_evaluator(num_episodes=3)
    out_file = tmp_path / f"episodes{ext}"
    result_path = Evaluator.save_episode_report(evaluator.last_episode_records, out_file)
    assert result_path == out_file
    assert out_file.exists()

    if fmt == "json":
        data = json.loads(out_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 3
        for rec in data:
            for key in [
                "episode_index",
                "seed",
                "environment",
                "scenario",
                "return",
                "episode_length",
                "success",
                "collision",
            ]:
                assert key in rec
    else:
        with out_file.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        expected_fields = [
            "episode_index",
            "seed",
            "environment",
            "scenario",
            "return",
            "episode_length",
            "success",
            "collision",
        ]
        assert reader.fieldnames == expected_fields

    env.close()


def test_deterministic_seed_preservation(tmp_path: Path):
    eval1, _, env1 = _run_evaluator(num_episodes=2, base_seed=777)
    out1 = tmp_path / "first.json"
    Evaluator.save_episode_report(eval1.last_episode_records, out1)
    records1 = json.loads(out1.read_text())

    eval2, _, env2 = _run_evaluator(num_episodes=2, base_seed=777)
    out2 = tmp_path / "second.json"
    Evaluator.save_episode_report(eval2.last_episode_records, out2)
    records2 = json.loads(out2.read_text())

    assert records1 == records2
    env1.close()
    env2.close()


def test_scenario_name_propagation(tmp_path: Path):
    from adaptive_rl.evaluation.scenarios import EvaluationScenario

    env = DummyTestEnv(step_limit=5)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32)
    evaluator = Evaluator(algorithm=algo, env=env)
    scenario = EvaluationScenario(name="test_scenario", seed=10, environment_overrides={})
    evaluator.evaluate_scenarios([scenario], deterministic=True)
    records = evaluator.last_episode_records
    assert len(records) == 1
    assert records[0].scenario == "test_scenario"
    env.close()
