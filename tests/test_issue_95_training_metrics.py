"""Issue #95 regression coverage at the training and artifact boundaries."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest

from adaptive_rl.config import AlgorithmConfig, EnvironmentConfig, ExperimentConfig, TrainingConfig
from adaptive_rl.experiments.metadata import EpisodeRecord, load_episodes_csv
from adaptive_rl.metrics import EpisodeMetrics, compute_rate
from adaptive_rl.training.callbacks import MetricLoggerCallback, SB3CallbackAdapter
from adaptive_rl.training.trainer import PPOTrainer, SACTrainer, TrainingResult


class DeterministicOutcomeEnv(gym.Env):
    """One-step continuous-action environment with deterministic terminal telemetry."""

    metadata = {"render_modes": []}

    def __init__(self, terminal_info: Mapping[str, Any], truncated: bool = False) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.terminal_info = dict(terminal_info)
        self.truncated = truncated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return (
            np.zeros(2, dtype=np.float32),
            1.0,
            not self.truncated,
            self.truncated,
            dict(self.terminal_info),
        )


def _run_training(
    trainer_cls: type[PPOTrainer] | type[SACTrainer],
    terminal_info: Mapping[str, Any],
    tmp_path: Path,
    *,
    truncated: bool = False,
) -> TrainingResult:
    algorithm = "ppo" if trainer_cls is PPOTrainer else "sac"
    parameters = (
        {"n_steps": 2, "n_epochs": 1}
        if algorithm == "ppo"
        else {"buffer_size": 8, "learning_starts": 8, "train_freq": 1}
    )
    config = ExperimentConfig(
        name=f"issue95_{algorithm}",
        seed=7,
        algorithm=AlgorithmConfig(name=algorithm, batch_size=2, parameters=parameters),
        environment=EnvironmentConfig(name="navigation"),
        training=TrainingConfig(total_timesteps=2),
        output_dir=tmp_path,
    )
    trainer = trainer_cls(
        config=config,
        env=DeterministicOutcomeEnv(terminal_info, truncated=truncated),
    )
    try:
        return trainer.fit()
    finally:
        trainer.close()


def _read_artifacts(
    result: TrainingResult,
) -> tuple[list[EpisodeRecord], dict[str, Any], list[dict[str, str]]]:
    assert result.episodes_csv_path is not None
    assert result.metadata_path is not None
    records = load_episodes_csv(result.episodes_csv_path)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    with result.episodes_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return records, metadata, rows


def test_metric_logger_treats_supplied_canonical_metrics_as_authoritative() -> None:
    """Conflicting raw info must not replace the canonical metrics object."""
    logger = MetricLoggerCallback()
    metrics = EpisodeMetrics(
        reward=3.0,
        length=7,
        success=True,
        collision=False,
        terminated=True,
        truncated=False,
    )

    logger.on_episode_end(
        episode=1,
        episode_reward=3.0,
        episode_length=7,
        info={"success": False, "collision": True},
        metrics=metrics,
    )

    assert logger.episode_metrics == [metrics]
    assert logger.successes == 1
    assert logger.collisions == 0
    assert logger.success_rate == 1.0
    assert logger.collision_rate == 0.0


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
@pytest.mark.parametrize(
    (
        "terminal_info",
        "expected_success",
        "expected_collision",
        "expected_success_rate",
        "expected_collision_rate",
    ),
    [
        ({"success": True, "collision": False}, True, False, 1.0, 0.0),
        ({"success": False, "collision": False}, False, False, 0.0, 0.0),
        ({}, None, None, None, None),
        ({"success": True, "collision": True}, False, True, 0.0, 1.0),
    ],
)
def test_ppo_and_sac_persist_canonical_outcomes(
    trainer_cls: type[PPOTrainer] | type[SACTrainer],
    terminal_info: Mapping[str, Any],
    expected_success: Optional[bool],
    expected_collision: Optional[bool],
    expected_success_rate: Optional[float],
    expected_collision_rate: Optional[float],
    tmp_path: Path,
) -> None:
    """Real PPO/SAC training must use canonical outcomes in generated artifacts."""
    result = _run_training(trainer_cls, terminal_info, tmp_path)
    records, metadata, rows = _read_artifacts(result)

    assert result.episodes_completed == 2
    assert len(records) == 2
    assert all(record.success is expected_success for record in records)
    assert all(record.collision is expected_collision for record in records)
    assert result.success_rate == expected_success_rate
    assert result.collision_rate == expected_collision_rate
    assert len(rows) == 2
    assert all(
        row["success"] == ("" if expected_success is None else str(expected_success))
        for row in rows
    )
    assert all(
        row["collision"] == ("" if expected_collision is None else str(expected_collision))
        for row in rows
    )
    assert metadata["success_rate"] == expected_success_rate
    assert metadata["collision_rate"] == expected_collision_rate


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
def test_truncated_training_episode_preserves_terminal_flags(
    trainer_cls: type[PPOTrainer] | type[SACTrainer], tmp_path: Path
) -> None:
    """Truncation is an episode boundary and retains explicit terminal telemetry."""
    result = _run_training(
        trainer_cls,
        {"success": True, "collision": False},
        tmp_path,
        truncated=True,
    )
    records, _, _ = _read_artifacts(result)

    assert result.episodes_completed == 2
    assert len(records) == 2
    assert all(record.success is True for record in records)
    assert all(record.collision is False for record in records)


def test_compute_rate_distinguishes_unknown_from_measured_zero() -> None:
    assert compute_rate([None, None]) is None
    assert compute_rate([False, False, None]) == 0.0


def test_sb3_adapter_delivers_episode_metrics_to_logger() -> None:
    """The callback boundary delivers an EpisodeMetrics instance, not raw info."""
    logger = MetricLoggerCallback()
    adapter = SB3CallbackAdapter(callbacks=[logger])
    adapter.locals = {
        "dones": [True],
        "rewards": [1.0],
        "infos": [{"success": True, "collision": True}],
    }
    adapter.num_timesteps = 1

    assert adapter._on_step() is True
    assert len(logger.episode_metrics) == 1
    assert isinstance(logger.episode_metrics[0], EpisodeMetrics)
    assert logger.episode_metrics[0].success is False
    assert logger.episode_metrics[0].collision is True
