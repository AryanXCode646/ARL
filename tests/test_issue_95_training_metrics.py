"""Issue #95 regression coverage at the training and artifact boundaries."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest

from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.experiments.metadata import (
    EpisodeRecord,
    ExperimentMetadata,
    load_episodes_csv,
    save_episodes_csv,
)
from adaptive_rl.metrics import (
    EpisodeMetrics,
    compute_rate,
    extract_episode_metrics,
)
from adaptive_rl.training.callbacks import (
    MetricLoggerCallback,
    SB3CallbackAdapter,
)
from adaptive_rl.training.trainer import PPOTrainer, SACTrainer, TrainingResult


class DeterministicOutcomeEnv(gym.Env):
    """Deterministic, lightweight continuous-action environment with deterministic terminal telemetry.

    Features:
    - 2D continuous observation space and 1D continuous action space, compatible
      with both PPO (MlpPolicy) and SAC (which requires continuous actions).
    - Sequential replay of supplied episode terminal info mappings or single mapping.
    - Configurable steps per episode for single-step and multi-step verification.
    - Optional truncated flag to test Gymnasium truncated episodes.
    - Optional non_terminal_info to emit telemetry on non‑terminal steps.
    - Fully deterministic, fast, CPU‑only, and Gymnasium compliant.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        terminal_info: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        steps_per_episode: int = 1,
        *,
        truncated: bool = False,
        non_terminal_info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        if isinstance(terminal_info, Mapping):
            self.episode_infos = [dict(terminal_info)]
        else:
            self.episode_infos = [dict(info) for info in terminal_info]
        if steps_per_episode < 1:
            raise ValueError("steps_per_episode must be >= 1")
        self.steps_per_episode = steps_per_episode
        self.truncated = truncated
        self.truncated_flag = truncated
        self.non_terminal_info = non_terminal_info
        self._current_step = 0
        self._episode_index = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._current_step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._current_step += 1
        terminated = self._current_step >= self.steps_per_episode
        truncated = False
        reward = 1.0
        if terminated:
            terminal_info = {}
            if self.episode_infos:
                terminal_info = dict(
                    self.episode_infos[self._episode_index % len(self.episode_infos)]
                )
            self._episode_index += 1
            self._current_step = 0
            if self.truncated:
                terminated = False
                truncated = True
            info = terminal_info
        else:
            info = dict(self.non_terminal_info) if self.non_terminal_info else {}
        return (
            np.zeros(2, dtype=np.float32),
            reward,
            terminated,
            truncated,
            info,
        )


def _run_training(
    trainer_cls: type[PPOTrainer] | type[SACTrainer],
    terminal_info: Sequence[Mapping[str, Any]] | Mapping[str, Any],
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


def _read_training_artifacts(
    result: TrainingResult,
) -> tuple[list[EpisodeRecord], dict[str, Any]]:
    records, metadata, _ = _read_artifacts(result)
    return records, metadata


def _run_trainer_e2e(
    trainer_cls: type[PPOTrainer] | type[SACTrainer],
    env: gym.Env,
    tmp_path: Path,
    name: str,
    total_timesteps: int = 4,
) -> TrainingResult:
    """Execute actual end‑to‑end training without mocking."""
    is_ppo = trainer_cls is PPOTrainer
    algo_name = "ppo" if is_ppo else "sac"
    algo_params = (
        {"n_steps": total_timesteps, "n_epochs": 1}
        if is_ppo
        else {"buffer_size": 100, "learning_starts": 2}
    )
    config = ExperimentConfig(
        name=name,
        seed=42,
        algorithm=AlgorithmConfig(name=algo_name, batch_size=4, parameters=algo_params),
        environment=EnvironmentConfig(name="navigation"),
        training=TrainingConfig(total_timesteps=total_timesteps),
        output_dir=tmp_path / name,
    )
    trainer = trainer_cls(config=config, env=env)
    try:
        return trainer.fit()
    finally:
        trainer.close()


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


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
        info={"success": False, "collision": False},
    )
    assert metrics.success is False
    assert metrics.collision is False


def test_terminal_success_and_collision_are_extracted() -> None:
    metrics = extract_episode_metrics(
        reward=5.0,
        length=10,
        terminated=True,
        truncated=False,
        info={"success": True, "collision": False},
    )
    assert metrics.success is True
    assert metrics.collision is False


def test_collision_overrides_conflicting_positive_success() -> None:
    metrics = extract_episode_metrics(
        reward=5.0,
        length=10,
        terminated=True,
        truncated=False,
        info={"success": True, "collision": True},
    )
    assert metrics.collision is True
    assert metrics.success is False


def test_equivalent_success_keys_are_supported() -> None:
    metrics = extract_episode_metrics(
        reward=1.0,
        length=5,
        terminated=True,
        truncated=False,
        info={"is_success": True},
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
        info={"success": False, "collision": False, "energy": 0.0},
    )
    assert metrics.additional_metrics["energy"] == 0.0


def test_compute_rate_distinguishes_unknown_from_measured_zero() -> None:
    assert compute_rate([None, None]) is None
    assert compute_rate([False, False, None]) == 0.0


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
    path = save_episodes_csv(records=[record], output_dir=tmp_path, name="test")
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


def test_sb3_callback_adapter_feeds_metric_logger_canonical_metrics() -> None:
    logger = MetricLoggerCallback()
    adapter = SB3CallbackAdapter(callbacks=[logger])
    adapter.num_timesteps = 5
    adapter.locals = {
        "dones": [True],
        "rewards": [1.0],
        "infos": [{"success": True, "collision": False}],
    }
    adapter._on_step()
    adapter.num_timesteps = 10
    adapter.locals = {"dones": [True], "rewards": [0.0], "infos": [{}]}
    adapter._on_step()
    assert len(logger.episode_metrics) == 2
    assert logger.episode_metrics[0].success is True
    assert logger.episode_metrics[0].collision is False
    assert logger.episode_metrics[1].success is None
    assert logger.episode_metrics[1].collision is None
    assert logger.success_rate == 1.0
    assert logger.collision_rate == 0.0


# ---------------------------------------------------------------------------
# Real End-to-End PPO and SAC Training Tests
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
def test_e2e_training_numpy_telemetry_payload_artifact(
    trainer_cls: type[PPOTrainer] | type[SACTrainer], tmp_path: Path
) -> None:
    """Terminal info containing multi-element NumPy arrays does not crash training."""
    terminal_info = {
        "success": True,
        "collision": False,
        "array_data": np.array([1.0, 2.0, 3.0]),
    }
    result = _run_training(trainer_cls, terminal_info, tmp_path)
    records, metadata, rows = _read_artifacts(result)
    assert result.episodes_completed == 2
    assert len(records) == 2
    assert result.success_rate == 1.0
    assert result.collision_rate == 0.0


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
def test_e2e_training_multi_episode_mixed_outcomes_and_denominator(
    trainer_cls: type[PPOTrainer] | type[SACTrainer], tmp_path: Path
) -> None:
    """Multi‑episode sequence verifying denominator semantics and tri‑state outcomes.

    Episodes:
    1: success=True, collision=False
    2: success=False, collision=False
    3: {} -> undefined
    4: success=True, collision=True -> conflict
    """
    infos = [
        {"success": True, "collision": False},
        {"success": False, "collision": False},
        {},
        {"success": True, "collision": True},
    ]
    env = DeterministicOutcomeEnv(infos)
    prefix = "ppo" if trainer_cls is PPOTrainer else "sac"
    result = _run_trainer_e2e(trainer_cls, env, tmp_path, f"{prefix}_multi", total_timesteps=4)
    assert result.episodes_completed == 4
    expected = 1.0 / 3.0
    assert result.success_rate == pytest.approx(expected)
    assert result.collision_rate == pytest.approx(expected)
    records, metadata = _read_training_artifacts(result)
    assert len(records) == 4
    assert records[0].success is True and records[0].collision is False
    assert records[1].success is False and records[1].collision is False
    assert records[2].success is None and records[2].collision is None
    assert records[3].success is False and records[3].collision is True
    rows = list(csv.DictReader(result.episodes_csv_path.open(newline="", encoding="utf-8")))
    assert rows[0]["success"] == "True" and rows[0]["collision"] == "False"
    assert rows[1]["success"] == "False" and rows[1]["collision"] == "False"
    assert rows[2]["success"] == "" and rows[2]["collision"] == ""
    assert rows[3]["success"] == "False" and rows[3]["collision"] == "True"
    assert metadata["episodes_completed"] == 4
    assert metadata["success_rate"] == pytest.approx(expected)
    assert metadata["collision_rate"] == pytest.approx(expected)


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
def test_e2e_training_multi_step_episode_accumulation(
    trainer_cls: type[PPOTrainer] | type[SACTrainer], tmp_path: Path
) -> None:
    """Verify multi‑step episode accumulation.

    Two episodes, each with two steps (total_timesteps=4).
    Episode 1: success=True, collision=False
    Episode 2: success=False, collision=False
    """
    infos = [{"success": True, "collision": False}, {"success": False, "collision": False}]
    env = DeterministicOutcomeEnv(infos, steps_per_episode=2)
    prefix = "ppo" if trainer_cls is PPOTrainer else "sac"
    result = _run_trainer_e2e(trainer_cls, env, tmp_path, f"{prefix}_step2", total_timesteps=4)
    assert result.episodes_completed == 2
    assert result.success_rate == 0.5
    assert result.collision_rate == 0.0
    records, metadata = _read_training_artifacts(result)
    assert len(records) == 2
    assert (
        records[0].episode == 1
        and records[0].length == 2
        and records[0].reward == 2.0
        and records[0].timestep == 2
    )
    assert records[0].success is True and records[0].collision is False
    assert (
        records[1].episode == 2
        and records[1].length == 2
        and records[1].reward == 2.0
        and records[1].timestep == 4
    )
    assert records[1].success is False and records[1].collision is False


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
def test_e2e_training_misleading_nonterminal_telemetry_artifact(
    trainer_cls: type[PPOTrainer] | type[SACTrainer], tmp_path: Path
) -> None:
    """Test that non‑terminal telemetry does not affect the canonical outcome.

    First step (non‑terminal) emits success=True, then terminal step has no telemetry.
    """

    class MisleadingEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self._step = 0

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
            super().reset(seed=seed)
            self._step = 0
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action: Any):
            self._step += 1
            reward = 1.0
            if self._step == 1:
                return (
                    np.zeros(2, dtype=np.float32),
                    reward,
                    False,
                    False,
                    {"success": True, "collision": False},
                )
            else:
                return np.zeros(2, dtype=np.float32), reward, True, False, {}

    env = MisleadingEnv()
    prefix = "ppo" if trainer_cls is PPOTrainer else "sac"
    result = _run_trainer_e2e(trainer_cls, env, tmp_path, f"{prefix}_mislead")
    # Outcome should be undefined because terminal info is missing
    assert result.success_rate is None
    assert result.collision_rate is None
    records, metadata = _read_training_artifacts(result)
    assert result.episodes_completed == 2
    assert len(records) == 2
    for record in records:
        assert record.success is None
        assert record.collision is None
    assert metadata["success_rate"] is None
    assert metadata["collision_rate"] is None
