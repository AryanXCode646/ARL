"""Regression tests for Issue #95: training episode metrics and outcome artifacts.

Verifies:
1. Terminal success/collision canonical extraction rules (missing=None, False=False, True=True).
2. Collision overrides conflicting success.
3. Precedence and equivalence of keys (e.g. episode_success, is_success, collision, had_collision).
4. compute_rate semantics: measured 0.0 is 0.0, undefined is None.
5. Preserving zero-valued additional metrics.
6. MetricLoggerCallback consuming canonical EpisodeMetrics and preserving tri-state outcomes.
7. Nullable outcome fields in EpisodeRecord, episodes.csv, and ExperimentMetadata.
8. Canonical callback forwarding parity between PPO and SAC.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.experiments.metadata import (
    EpisodeRecord,
    ExperimentMetadata,
    save_episodes_csv,
)
from adaptive_rl.metrics import (
    compute_rate,
    extract_episode_metrics,
)
from adaptive_rl.training.callbacks import (
    MetricLoggerCallback,
    SB3CallbackAdapter,
)


def test_missing_terminal_metrics_are_none() -> None:
    metrics = extract_episode_metrics(
        reward=1.0,
        length=5,
        terminated=True,
        truncated=False,
        info={},
    )

    assert metrics.success is None
    assert metrics.collision is None


def test_explicit_false_is_not_treated_as_missing() -> None:
    metrics = extract_episode_metrics(
        reward=1.0,
        length=5,
        terminated=True,
        truncated=False,
        info={
            "success": False,
            "collision": False,
        },
    )

    assert metrics.success is False
    assert metrics.collision is False


def test_terminal_success_and_collision_are_extracted() -> None:
    metrics = extract_episode_metrics(
        reward=5.0,
        length=10,
        terminated=True,
        truncated=False,
        info={
            "success": True,
            "collision": False,
        },
    )

    assert metrics.success is True
    assert metrics.collision is False


def test_collision_overrides_conflicting_positive_success() -> None:
    metrics = extract_episode_metrics(
        reward=5.0,
        length=10,
        terminated=True,
        truncated=False,
        info={
            "success": True,
            "collision": True,
        },
    )

    assert metrics.collision is True
    assert metrics.success is False


def test_equivalent_success_keys_are_supported() -> None:
    metrics = extract_episode_metrics(
        reward=1.0,
        length=5,
        terminated=True,
        truncated=False,
        info={
            "is_success": True,
        },
    )

    assert metrics.success is True


def test_zero_rate_is_zero_not_none() -> None:
    assert compute_rate([False, False]) == 0.0


def test_undefined_rate_is_none() -> None:
    assert compute_rate([None, None]) is None


def test_zero_valued_additional_metric_is_preserved() -> None:
    metrics = extract_episode_metrics(
        reward=1.0,
        length=3,
        terminated=True,
        truncated=False,
        info={
            "success": False,
            "collision": False,
            "energy": 0.0,
        },
    )

    assert metrics.additional_metrics["energy"] == 0.0


def test_metric_logger_consumes_canonical_metrics() -> None:
    logger = MetricLoggerCallback()

    metrics = extract_episode_metrics(
        reward=3.0,
        length=7,
        terminated=True,
        truncated=False,
        info={
            "success": True,
            "collision": False,
        },
    )

    logger.on_episode_end(
        episode=1,
        episode_reward=3.0,
        episode_length=7,
        info={
            "success": True,
            "collision": False,
        },
        metrics=metrics,
    )

    assert len(logger.episode_metrics) == 1
    assert logger.episode_metrics[0].success is True
    assert logger.episode_metrics[0].collision is False
    assert logger.success_rate == 1.0
    assert logger.collision_rate == 0.0


def test_metric_logger_preserves_unknown_outcomes() -> None:
    logger = MetricLoggerCallback()

    metrics = extract_episode_metrics(
        reward=2.0,
        length=4,
        terminated=True,
        truncated=False,
        info={},
    )

    logger.on_episode_end(
        episode=1,
        episode_reward=2.0,
        episode_length=4,
        info={},
        metrics=metrics,
    )

    assert logger.episode_metrics[0].success is None
    assert logger.episode_metrics[0].collision is None
    assert logger.success_rate is None
    assert logger.collision_rate is None


def test_episode_record_accepts_nullable_outcomes() -> None:
    record = EpisodeRecord(
        episode=1,
        reward=1.0,
        length=5,
        success=None,
        collision=None,
        timestep=5,
    )

    assert record.success is None
    assert record.collision is None


def test_nullable_outcomes_are_preserved_in_csv(tmp_path: Path) -> None:
    record = EpisodeRecord(
        episode=1,
        reward=1.0,
        length=5,
        success=None,
        collision=None,
        timestep=5,
    )

    path = save_episodes_csv(
        records=[record],
        output_dir=tmp_path,
        name="test",
    )

    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["success"] == ""
    assert row["collision"] == ""


def test_metadata_preserves_none_rates(tmp_path: Path) -> None:
    metadata = ExperimentMetadata(
        experiment_name="test",
        algorithm="ppo",
        environment="dummy",
        seed=0,
        total_timesteps=10,
        actual_timesteps=10,
        episodes_completed=1,
        mean_reward=1.0,
        success_rate=None,
        collision_rate=None,
        final_model_path="model.zip",
    )

    path = metadata.save(tmp_path, name="test")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["success_rate"] is None
    assert payload["collision_rate"] is None


def test_sb3_callback_adapter_feeds_metric_logger_canonical_metrics() -> None:
    logger = MetricLoggerCallback()
    adapter = SB3CallbackAdapter(callbacks=[logger])

    # Simulate episode 1: success=True, collision=False
    adapter.num_timesteps = 5
    adapter.locals = {
        "dones": [True],
        "rewards": [1.0],
        "infos": [{"success": True, "collision": False}],
    }
    adapter._on_step()

    # Simulate episode 2: missing telemetry (success=None, collision=None)
    adapter.num_timesteps = 10
    adapter.locals = {
        "dones": [True],
        "rewards": [0.0],
        "infos": [{}],
    }
    adapter._on_step()

    assert len(logger.episode_metrics) == 2
    assert logger.episode_metrics[0].success is True
    assert logger.episode_metrics[0].collision is False
    assert logger.episode_metrics[1].success is None
    assert logger.episode_metrics[1].collision is None
    assert logger.success_rate == 1.0
    assert logger.collision_rate == 0.0


@pytest.mark.parametrize(
    ("algorithm_cls", "env_id", "kwargs"),
    [
        (
            PPOAlgorithm,
            "CartPole-v1",
            {
                "n_steps": 8,
                "batch_size": 4,
                "n_epochs": 1,
            },
        ),
        (
            SACAlgorithm,
            "Pendulum-v1",
            {
                "buffer_size": 16,
                "learning_starts": 1,
                "batch_size": 4,
            },
        ),
    ],
)
def test_ppo_and_sac_forward_same_callback(
    monkeypatch: pytest.MonkeyPatch,
    algorithm_cls: type[PPOAlgorithm] | type[SACAlgorithm],
    env_id: str,
    kwargs: dict[str, Any],
) -> None:
    env = gym.make(env_id)

    algorithm = algorithm_cls(
        env=env,
        seed=0,
        verbose=0,
        device="cpu",
        **kwargs,
    )

    received: dict[str, Any] = {}

    def fake_learn(
        *,
        total_timesteps: int,
        callback: Any,
        reset_num_timesteps: bool,
    ) -> None:
        received["callback"] = callback
        received["total_timesteps"] = total_timesteps
        received["reset_num_timesteps"] = reset_num_timesteps

    monkeypatch.setattr(
        algorithm.model,
        "learn",
        fake_learn,
    )

    marker = object()

    algorithm.train(
        total_timesteps=1,
        callback=marker,
    )

    assert received["callback"] is marker
    assert received["total_timesteps"] == 1

    env.close()
