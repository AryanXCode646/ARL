"""Regression tests for Issue #95: training episode metrics and outcome artifacts.

Verifies:
1. Terminal success/collision canonical extraction rules (missing=None, False=False, True=True).
2. Collision overrides conflicting success.
3. Precedence and equivalence of keys (e.g. episode_success, is_success, collision, had_collision).
4. compute_rate semantics: measured 0.0 is 0.0, undefined is None, None excluded from denominator.
5. Preserving zero-valued additional metrics.
6. MetricLoggerCallback consuming canonical EpisodeMetrics and preserving tri-state outcomes.
7. Nullable outcome fields in EpisodeRecord, episodes.csv, and ExperimentMetadata.
8. Real end-to-end PPO and SAC training flows verifying terminal outcome extraction and persisted
   artifacts (episodes.csv, metadata.json) without mocking model.learn().
9. Multi-episode mixed sequences ensuring undefined outcomes do not inflate rate denominators.
10. Multi-step episode accumulation and persistence across both PPO and SAC.
"""

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
    compute_rate,
    extract_episode_metrics,
)
from adaptive_rl.training.callbacks import (
    MetricLoggerCallback,
    SB3CallbackAdapter,
)
from adaptive_rl.training.trainer import (
    PPOTrainer,
    SACTrainer,
    TrainingResult,
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


# ---------------------------------------------------------------------------
# Deterministic Test Environment for Real End-to-End PPO & SAC Training
# ---------------------------------------------------------------------------


class DeterministicOutcomeEnv(gym.Env):
    """Deterministic, lightweight environment for testing episode terminal metrics.

    Features:
    - 2D continuous observation space and 1D continuous action space, compatible
      with both PPO (MlpPolicy) and SAC (which requires continuous actions).
    - Sequential replay of supplied episode terminal info mappings.
    - Configurable steps per episode for single-step and multi-step verification.
    - Fully deterministic, fast, CPU-only, and Gymnasium compliant.
    """

    def __init__(
        self,
        episode_infos: Sequence[Mapping[str, Any]],
        steps_per_episode: int = 1,
    ) -> None:
        super().__init__()
        self.episode_infos = list(episode_infos)
        self.steps_per_episode = max(1, steps_per_episode)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._current_step: int = 0
        self._episode_index: int = 0

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
            if self.episode_infos:
                terminal_info = dict(
                    self.episode_infos[self._episode_index % len(self.episode_infos)]
                )
            else:
                terminal_info = {}
            self._episode_index += 1
            self._current_step = 0
            info = terminal_info
        else:
            info = {}

        return np.zeros(2, dtype=np.float32), reward, terminated, truncated, info


def _run_trainer_e2e(
    trainer_cls: type[PPOTrainer] | type[SACTrainer],
    env: gym.Env,
    tmp_path: Path,
    name: str,
    total_timesteps: int = 4,
) -> TrainingResult:
    """Execute actual end-to-end training without mocking."""
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
        algorithm=AlgorithmConfig(
            name=algo_name,
            batch_size=4,
            parameters=algo_params,
        ),
        environment=EnvironmentConfig(name="navigation"),
        training=TrainingConfig(total_timesteps=total_timesteps),
        output_dir=tmp_path / name,
    )

    trainer = trainer_cls(config=config, env=env)
    return trainer.fit()


# ---------------------------------------------------------------------------
# Real End-to-End PPO and SAC Training Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
def test_e2e_training_positive_outcome_artifact(
    trainer_cls: type[PPOTrainer] | type[SACTrainer],
    tmp_path: Path,
) -> None:
    """Case A: info={"success": True, "collision": False}.

    Proves real PPO and SAC training captures positive outcomes and writes
    success=True, collision=False, success_rate=1.0, collision_rate=0.0 to
    both episodes.csv and metadata.json.
    """
    env = DeterministicOutcomeEnv([{"success": True, "collision": False}])
    prefix = "ppo" if trainer_cls is PPOTrainer else "sac"
    result = _run_trainer_e2e(trainer_cls, env, tmp_path, f"{prefix}_pos")

    assert result.success_rate == 1.0
    assert result.collision_rate == 0.0

    # Verify real generated episodes.csv
    assert result.episodes_csv_path is not None
    assert result.episodes_csv_path.exists()
    records = load_episodes_csv(result.episodes_csv_path)
    assert len(records) > 0
    for record in records:
        assert record.success is True
        assert record.collision is False

    with result.episodes_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(records)
    for row in rows:
        assert row["success"] == "True"
        assert row["collision"] == "False"

    # Verify real generated metadata.json
    assert result.metadata_path is not None
    assert result.metadata_path.exists()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["success_rate"] == 1.0
    assert metadata["collision_rate"] == 0.0


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
def test_e2e_training_missing_telemetry_artifact(
    trainer_cls: type[PPOTrainer] | type[SACTrainer],
    tmp_path: Path,
) -> None:
    """Case C: info={}.

    Proves real PPO and SAC training preserves missing telemetry as None/null,
    writing empty string representation in CSV and null in metadata.json.
    Missing telemetry is NEVER converted into False or 0.0.
    """
    env = DeterministicOutcomeEnv([{}])
    prefix = "ppo" if trainer_cls is PPOTrainer else "sac"
    result = _run_trainer_e2e(trainer_cls, env, tmp_path, f"{prefix}_missing")

    assert result.success_rate is None
    assert result.collision_rate is None

    # Verify real generated episodes.csv
    assert result.episodes_csv_path is not None
    assert result.episodes_csv_path.exists()
    records = load_episodes_csv(result.episodes_csv_path)
    assert len(records) > 0
    for record in records:
        assert record.success is None
        assert record.collision is None

    with result.episodes_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(records)
    for row in rows:
        assert row["success"] == ""
        assert row["collision"] == ""

    # Verify real generated metadata.json
    assert result.metadata_path is not None
    assert result.metadata_path.exists()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["success_rate"] is None
    assert metadata["collision_rate"] is None


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
def test_e2e_training_explicit_zero_rate_artifact(
    trainer_cls: type[PPOTrainer] | type[SACTrainer],
    tmp_path: Path,
) -> None:
    """Case B: info={"success": False, "collision": False}.

    Proves real PPO and SAC training distinguishes explicit zero (0.0) from
    undefined (None). Explicit False yields rate 0.0, NOT None.
    """
    env = DeterministicOutcomeEnv([{"success": False, "collision": False}])
    prefix = "ppo" if trainer_cls is PPOTrainer else "sac"
    result = _run_trainer_e2e(trainer_cls, env, tmp_path, f"{prefix}_zero")

    assert result.success_rate is not None
    assert result.success_rate == 0.0
    assert result.collision_rate is not None
    assert result.collision_rate == 0.0

    # Verify real generated episodes.csv
    assert result.episodes_csv_path is not None
    assert result.episodes_csv_path.exists()
    records = load_episodes_csv(result.episodes_csv_path)
    assert len(records) > 0
    for record in records:
        assert record.success is False
        assert record.collision is False

    with result.episodes_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(records)
    for row in rows:
        assert row["success"] == "False"
        assert row["collision"] == "False"

    # Verify real generated metadata.json
    assert result.metadata_path is not None
    assert result.metadata_path.exists()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["success_rate"] == 0.0
    assert metadata["collision_rate"] == 0.0


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
def test_e2e_training_collision_overrides_success_artifact(
    trainer_cls: type[PPOTrainer] | type[SACTrainer],
    tmp_path: Path,
) -> None:
    """Case D: info={"success": True, "collision": True}.

    Proves collision overrides conflicting positive success in the canonical
    pipeline during real training, resulting in success=False, collision=True.
    """
    env = DeterministicOutcomeEnv([{"success": True, "collision": True}])
    prefix = "ppo" if trainer_cls is PPOTrainer else "sac"
    result = _run_trainer_e2e(trainer_cls, env, tmp_path, f"{prefix}_conflict")

    assert result.success_rate == 0.0
    assert result.collision_rate == 1.0

    # Verify real generated episodes.csv
    assert result.episodes_csv_path is not None
    assert result.episodes_csv_path.exists()
    records = load_episodes_csv(result.episodes_csv_path)
    assert len(records) > 0
    for record in records:
        assert record.success is False
        assert record.collision is True

    with result.episodes_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        assert row["success"] == "False"
        assert row["collision"] == "True"

    # Verify real generated metadata.json
    assert result.metadata_path is not None
    assert result.metadata_path.exists()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["success_rate"] == 0.0
    assert metadata["collision_rate"] == 1.0


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
def test_e2e_training_multi_episode_mixed_outcomes_and_denominator(
    trainer_cls: type[PPOTrainer] | type[SACTrainer],
    tmp_path: Path,
) -> None:
    """Multi-episode sequence verifying denominator semantics and tri-state outcomes.

    Sequence:
    Episode 1: success=True, collision=False
    Episode 2: success=False, collision=False
    Episode 3: info={} -> success=None, collision=None (undefined)
    Episode 4: success=True, collision=True -> success=False, collision=True (conflict)

    Expected rates:
    - Success: 1 True out of 3 defined episodes (Ep 1, 2, 4) -> 1/3
      (Episode 3 is undefined and must NOT inflate the denominator to 4).
    - Collision: 1 True out of 3 defined episodes (Ep 1, 2, 4) -> 1/3.
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
    expected_rate = 1.0 / 3.0
    assert result.success_rate == pytest.approx(expected_rate)
    assert result.collision_rate == pytest.approx(expected_rate)

    # Verify real generated episodes.csv
    assert result.episodes_csv_path is not None
    assert result.episodes_csv_path.exists()
    records = load_episodes_csv(result.episodes_csv_path)
    assert len(records) == 4

    assert records[0].success is True
    assert records[0].collision is False

    assert records[1].success is False
    assert records[1].collision is False

    assert records[2].success is None
    assert records[2].collision is None

    assert records[3].success is False
    assert records[3].collision is True

    # Verify CSV raw text representation
    with result.episodes_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["success"] == "True" and rows[0]["collision"] == "False"
    assert rows[1]["success"] == "False" and rows[1]["collision"] == "False"
    assert rows[2]["success"] == "" and rows[2]["collision"] == ""
    assert rows[3]["success"] == "False" and rows[3]["collision"] == "True"

    # Verify real generated metadata.json
    assert result.metadata_path is not None
    assert result.metadata_path.exists()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["episodes_completed"] == 4
    assert metadata["success_rate"] == pytest.approx(expected_rate)
    assert metadata["collision_rate"] == pytest.approx(expected_rate)


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
def test_e2e_training_multi_step_episode_accumulation(
    trainer_cls: type[PPOTrainer] | type[SACTrainer],
    tmp_path: Path,
) -> None:
    """Verify multi-step episode accumulation and persistence for PPO and SAC.

    2 episodes with 2 steps each (total_timesteps=4).
    Episode 1: success=True, collision=False (reward=2.0, length=2, cumulative_ts=2)
    Episode 2: success=False, collision=False (reward=2.0, length=2, cumulative_ts=4)
    """
    infos = [
        {"success": True, "collision": False},
        {"success": False, "collision": False},
    ]
    env = DeterministicOutcomeEnv(infos, steps_per_episode=2)
    prefix = "ppo" if trainer_cls is PPOTrainer else "sac"
    result = _run_trainer_e2e(trainer_cls, env, tmp_path, f"{prefix}_step2", total_timesteps=4)

    assert result.episodes_completed == 2
    assert result.success_rate == 0.5
    assert result.collision_rate == 0.0

    assert result.episodes_csv_path is not None
    records = load_episodes_csv(result.episodes_csv_path)
    assert len(records) == 2

    assert records[0].episode == 1
    assert records[0].length == 2
    assert records[0].reward == 2.0
    assert records[0].timestep == 2
    assert records[0].success is True
    assert records[0].collision is False

    assert records[1].episode == 2
    assert records[1].length == 2
    assert records[1].reward == 2.0
    assert records[1].timestep == 4
    assert records[1].success is False
    assert records[1].collision is False
