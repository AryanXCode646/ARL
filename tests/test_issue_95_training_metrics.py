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
    EpisodeMetricsAccumulator,
    TrafficOutcomePolicy,
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
    # Success is terminal-authoritative: terminal step has no success key -> None.
    assert result.success_rate is None
    # Collision is episode-level: non-terminal step explicitly reported collision=False,
    # so collision is defined (False), not unknown (None).
    assert result.collision_rate == 0.0
    records, metadata = _read_training_artifacts(result)
    assert result.episodes_completed == 2
    assert len(records) == 2
    for record in records:
        assert record.success is None
        assert record.collision is False
    assert metadata["success_rate"] is None
    assert metadata["collision_rate"] == 0.0


# ---------------------------------------------------------------------------
# Phase 8 & Phase 6/7/19 Comprehensive Verification Matrix
# ---------------------------------------------------------------------------


class TestCanonicalSemanticsMatrix:
    """Explicit regression coverage for TEST A through TEST F and Phase 19."""

    def test_matrix_a_nonterminal_success_terminal_empty(self) -> None:
        """TEST A: Non-terminal success=True, terminal info empty -> success=None."""
        acc = EpisodeMetricsAccumulator()
        acc.record_step(reward=1.0, terminated=False, truncated=False, info={"success": True})
        acc.record_step(reward=1.0, terminated=True, truncated=False, info={})
        m = acc.finish()
        assert m.success is None

    def test_matrix_b_nonterminal_success_terminal_false(self) -> None:
        """TEST B: Non-terminal success=True, terminal success=False -> success=False."""
        acc = EpisodeMetricsAccumulator()
        acc.record_step(reward=1.0, terminated=False, truncated=False, info={"success": True})
        acc.record_step(reward=1.0, terminated=True, truncated=False, info={"success": False})
        m = acc.finish()
        assert m.success is False

    def test_matrix_c_nonterminal_success_terminal_true(self) -> None:
        """TEST C: Non-terminal success=True, terminal success=True -> success=True."""
        acc = EpisodeMetricsAccumulator()
        acc.record_step(reward=1.0, terminated=False, truncated=False, info={"success": True})
        acc.record_step(reward=1.0, terminated=True, truncated=False, info={"success": True})
        m = acc.finish()
        assert m.success is True

    def test_matrix_d_nonterminal_collision_terminal_absent(self) -> None:
        """TEST D: Non-terminal collision=True, terminal collision absent -> collision=True."""
        acc = EpisodeMetricsAccumulator()
        acc.record_step(reward=1.0, terminated=False, truncated=False, info={"collision": True})
        acc.record_step(reward=1.0, terminated=True, truncated=False, info={})
        m = acc.finish()
        assert m.collision is True

    def test_matrix_e_terminal_success_true_collision_true(self) -> None:
        """TEST E: Terminal success=True + collision=True -> success=False, collision=True."""
        acc = EpisodeMetricsAccumulator()
        acc.record_step(
            reward=1.0,
            terminated=True,
            truncated=False,
            info={"success": True, "collision": True},
        )
        m = acc.finish()
        assert m.success is False
        assert m.collision is True

    def test_matrix_f_terminal_collision_with_success_absent(self) -> None:
        """TEST F: Terminal collision=True with success absent -> success=None, collision=True."""
        acc = EpisodeMetricsAccumulator()
        acc.record_step(
            reward=1.0,
            terminated=True,
            truncated=False,
            info={"collision": True},
        )
        m = acc.finish()
        assert m.success is None
        assert m.collision is True

    def test_phase_19_manual_semantic_proof_episode(self) -> None:
        """PHASE 19: Step 1 success=True, Step 2 collision=True, Step 3 info={}.

        Canonical outcome must NOT become success=True. Monotonic collision=True must be preserved.
        """
        acc = EpisodeMetricsAccumulator()
        acc.record_step(reward=1.0, terminated=False, truncated=False, info={"success": True})
        acc.record_step(reward=1.0, terminated=False, truncated=False, info={"collision": True})
        acc.record_step(reward=1.0, terminated=True, truncated=False, info={})
        m = acc.finish()
        assert m.success is None
        assert m.collision is True


class TestTrafficTelemetrySB3:
    """Explicit regression coverage for TEST G & H (Traffic Telemetry Retention & Policy)."""

    def test_matrix_g_nonterminal_traffic_overflow_preserved_in_sb3(self) -> None:
        """TEST G: Non-terminal traffic overflow=True, terminal success=True.

        The SB3 path MUST preserve step-1 traffic telemetry so TrafficOutcomePolicy
        sees the overflow and applies overflow semantics (success=False).
        """
        logger = MetricLoggerCallback()
        adapter = SB3CallbackAdapter(
            callbacks=[logger],
            outcome_policy=TrafficOutcomePolicy(),
        )

        # Step 1: non-terminal with traffic overflow
        adapter.num_timesteps = 1
        adapter.locals = {
            "dones": [False],
            "rewards": [1.0],
            "infos": [{"step_overflow": True, "total_queue": 15}],
        }
        adapter._on_step()

        # Step 2: terminal with success=True
        adapter.num_timesteps = 2
        adapter.locals = {
            "dones": [True],
            "rewards": [2.0],
            "infos": [{"success": True, "total_queue": 5}],
        }
        adapter._on_step()

        assert len(logger.episode_metrics) == 1
        m = logger.episode_metrics[0]
        # Overflow overrides positive terminal success in TrafficOutcomePolicy
        assert m.success is False
        assert m.additional_metrics.get("had_overflow") is True
        assert m.reward == 3.0
        assert m.length == 2

    @pytest.mark.parametrize("trainer_cls", [PPOTrainer, SACTrainer])
    def test_matrix_g_traffic_overflow_in_real_training_path(
        self, trainer_cls: type[PPOTrainer] | type[SACTrainer], tmp_path: Path
    ) -> None:
        """TEST G (Real Trainer Path): Step 1 emits overflow=True, Step 2 emits success=True.

        Verifies that non-terminal telemetry survives all the way through SB3 PPO/SAC training
        to TrafficOutcomePolicy.
        """

        class TrafficOverflowEnv(gym.Env):
            def __init__(self) -> None:
                super().__init__()
                self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
                self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
                self._step = 0

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self._step = 0
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action):
                self._step += 1
                if self._step == 1:
                    return (
                        np.zeros(2, dtype=np.float32),
                        1.0,
                        False,
                        False,
                        {"step_overflow": True, "total_queue": 20},
                    )
                return (
                    np.zeros(2, dtype=np.float32),
                    1.0,
                    True,
                    False,
                    {"success": True, "total_queue": 5},
                )

        prefix = "ppo_traffic" if trainer_cls is PPOTrainer else "sac_traffic"
        total_ts = 2
        algo_name = "ppo" if trainer_cls is PPOTrainer else "sac"
        algo_params = (
            {"n_steps": total_ts, "n_epochs": 1}
            if trainer_cls is PPOTrainer
            else {"buffer_size": 50, "learning_starts": 1}
        )
        config = ExperimentConfig(
            name=f"{prefix}_overflow",
            seed=42,
            algorithm=AlgorithmConfig(name=algo_name, batch_size=2, parameters=algo_params),
            environment=EnvironmentConfig(name="traffic_intersection"),
            training=TrainingConfig(total_timesteps=total_ts),
            output_dir=tmp_path / f"{prefix}_overflow",
        )
        trainer = trainer_cls(config=config, env=TrafficOverflowEnv())
        try:
            result = trainer.fit()
        finally:
            trainer.close()

        records, metadata = _read_training_artifacts(result)
        assert len(records) >= 1
        # In TrafficOutcomePolicy, overflow anywhere makes success False
        for rec in records:
            assert rec.success is False
        assert result.success_rate == 0.0
        assert metadata["success_rate"] == 0.0

    def test_matrix_h_traffic_telemetry_only_nonterminal_autodetection(self) -> None:
        """TEST H: Traffic telemetry only on non-terminal step, terminal info empty.

        Verify traffic auto-detection upgrades to TrafficOutcomePolicy when identifying signal
        appears before termination, and that premature termination without horizon is failure.
        """
        logger = MetricLoggerCallback()
        # No explicit policy supplied -> DefaultOutcomePolicy initially
        adapter = SB3CallbackAdapter(callbacks=[logger])

        # Step 1: non-terminal with traffic key
        adapter.num_timesteps = 1
        adapter.locals = {
            "dones": [False],
            "rewards": [0.0],
            "infos": [{"total_queue": 10}],
        }
        adapter._on_step()

        # Step 2: terminal with empty info (premature termination: terminated=True, truncated=False)
        adapter.num_timesteps = 2
        adapter.locals = {
            "dones": [True],
            "rewards": [0.0],
            "infos": [{}],
        }
        adapter._on_step()

        assert len(logger.episode_metrics) == 1
        m = logger.episode_metrics[0]
        # Premature natural termination in TrafficOutcomePolicy -> success=False
        assert m.success is False


class TestDeduplicationAndNumPySafety:
    """Explicit regression coverage for TEST I & TEST J (Deduplication & NumPy Safety)."""

    def test_matrix_i_terminal_info_already_present_in_step_infos(self) -> None:
        """TEST I: Terminal info object is also already present as the last step info."""
        step_1 = {"collision": False, "step": 1}
        step_2 = {"collision": False, "success": True, "step": 2}
        step_infos = [step_1, step_2]

        m = extract_episode_metrics(
            reward=2.0,
            length=2,
            terminated=True,
            truncated=False,
            info=step_2,  # Same object as step_infos[-1]
            step_infos=step_infos,
        )
        assert m.length == 2
        assert m.success is True
        assert m.collision is False

    def test_matrix_j_distinct_dicts_with_multielement_numpy_arrays(self) -> None:
        """TEST J: Two DISTINCT dictionaries containing multi-element NumPy arrays.

        Must NOT compare arbitrary dictionaries using a == b which raises ValueError:
        The truth value of an array with more than one element is ambiguous.
        Explicit identity check deduplication must be safe.
        """
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])  # distinct array instance, identical values
        dict1 = {"array": arr1, "success": True}
        dict2 = {"array": arr2, "success": True}  # distinct dict instance

        # Comparing dicts directly raises ValueError
        with pytest.raises(ValueError, match="ambiguous"):
            dict1 == dict2  # noqa: B015

        # extract_episode_metrics safely handles distinct dicts with numpy arrays without crashing
        m = extract_episode_metrics(
            reward=2.0,
            length=2,
            terminated=True,
            truncated=False,
            info=dict2,
            step_infos=[dict1],
        )
        assert m.length == 2
        assert m.success is True

    def test_matrix_j_distinct_dicts_different_numpy_arrays_not_deduped(self) -> None:
        """TEST J variant: Two distinct dicts with DIFFERENT NumPy arrays are NOT deduplicated."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 4])
        dict1 = {"array": arr1}
        dict2 = {"array": arr2, "success": True}

        m = extract_episode_metrics(
            reward=2.0,
            length=2,
            terminated=True,
            truncated=False,
            info=dict2,
            step_infos=[dict1],
        )
        assert m.length == 2
        assert m.success is True

    def test_numpy_in_episode_metrics_equality_safe(self) -> None:
        """NumPy arrays in additional_metrics do not crash EpisodeMetrics.__eq__."""
        m1 = EpisodeMetrics(
            reward=1.0,
            length=5,
            success=True,
            collision=False,
            terminated=True,
            truncated=False,
            additional_metrics={"weights": np.array([0.1, 0.2, 0.3])},
        )
        m2 = EpisodeMetrics(
            reward=1.0,
            length=5,
            success=True,
            collision=False,
            terminated=True,
            truncated=False,
            additional_metrics={"weights": np.array([0.1, 0.2, 0.3])},
        )
        assert m1 == m2

        m3 = EpisodeMetrics(
            reward=1.0,
            length=5,
            success=True,
            collision=False,
            terminated=True,
            truncated=False,
            additional_metrics={"weights": np.array([0.1, 0.2, 0.4])},
        )
        assert m1 != m3


class TestMultiEnvironmentSB3Adapter:
    """Explicit regression coverage for TEST K through TEST N (Single & Vectorized Environments)."""

    def test_matrix_k_single_environment_lifecycle(self) -> None:
        """TEST K: Single-environment SB3 lifecycle."""
        logger = MetricLoggerCallback()
        adapter = SB3CallbackAdapter(callbacks=[logger])

        adapter.num_timesteps = 1
        adapter.locals = {"dones": [False], "rewards": [1.0], "infos": [{"step": 1}]}
        adapter._on_step()

        adapter.num_timesteps = 2
        adapter.locals = {
            "dones": [True],
            "rewards": [2.0],
            "infos": [{"step": 2, "success": True, "collision": False}],
        }
        adapter._on_step()

        assert len(logger.episode_metrics) == 1
        m = logger.episode_metrics[0]
        assert m.reward == 3.0
        assert m.length == 2
        assert m.success is True
        assert m.collision is False

    def test_matrix_l_m_n_vectorized_environments(self) -> None:
        """TEST L, M, N: Vectorized environments with asynchronous and synchronous completions.

        - TEST L: Multiple environments running concurrently without cross-talk.
        - TEST M: Two environments terminate at different timesteps (Env 0 at step 2, Env 1 at step 4).
                  Then reverse (Env 1 terminates first).
        - TEST N: Two environments terminate simultaneously (dones=[True, True]).
        """
        logger = MetricLoggerCallback()
        adapter = SB3CallbackAdapter(callbacks=[logger])

        # Step 1: env 0 and env 1 running
        adapter.num_timesteps = 1
        adapter.locals = {
            "dones": [False, False],
            "rewards": [1.0, 10.0],
            "infos": [{"env": 0, "collision": False}, {"env": 1, "collision": False}],
        }
        adapter._on_step()

        # Step 2: env 0 finishes (success=True), env 1 continues (TEST M)
        adapter.num_timesteps = 2
        adapter.locals = {
            "dones": [True, False],
            "rewards": [1.0, 10.0],
            "infos": [{"env": 0, "success": True}, {"env": 1}],
        }
        adapter._on_step()
        assert len(logger.episode_metrics) == 1
        m0 = logger.episode_metrics[0]
        assert m0.reward == 2.0
        assert m0.length == 2
        assert m0.success is True

        # Step 3: env 0 restarted new episode, env 1 continues
        adapter.num_timesteps = 3
        adapter.locals = {
            "dones": [False, False],
            "rewards": [5.0, 10.0],
            "infos": [{"env": 0}, {"env": 1}],
        }
        adapter._on_step()

        # Step 4: env 1 finishes (collision=True), env 0 continues
        adapter.num_timesteps = 4
        adapter.locals = {
            "dones": [False, True],
            "rewards": [5.0, 10.0],
            "infos": [{"env": 0}, {"env": 1, "collision": True, "success": True}],
        }
        adapter._on_step()
        assert len(logger.episode_metrics) == 2
        m1 = logger.episode_metrics[1]
        assert m1.reward == 40.0
        assert m1.length == 4
        # Collision overrides positive success
        assert m1.collision is True
        assert m1.success is False

        # Step 5: Both finish simultaneously (TEST N)
        adapter.num_timesteps = 5
        adapter.locals = {
            "dones": [True, True],
            "rewards": [5.0, 1.0],
            "infos": [
                {"env": 0, "success": True, "collision": False},
                {"env": 1, "success": False, "collision": False},
            ],
        }
        adapter._on_step()
        assert len(logger.episode_metrics) == 4
        # Env 0 had steps 3, 4, 5 -> length 3, reward 5 + 5 + 5 = 15
        m0_ep2 = logger.episode_metrics[2]
        assert m0_ep2.reward == 15.0
        assert m0_ep2.length == 3
        assert m0_ep2.success is True
        # Env 1 had step 5 only -> length 1, reward 1.0
        m1_ep2 = logger.episode_metrics[3]
        assert m1_ep2.reward == 1.0
        assert m1_ep2.length == 1
        assert m1_ep2.success is False


class TestDirectExtractionVsSB3Consistency:
    """Explicit regression coverage for PHASE 7 (Direct vs SB3 vs PPO vs SAC Consistency)."""

    def test_phase_7_consistency_across_all_evaluation_and_training_paths(
        self, tmp_path: Path
    ) -> None:
        """Process the same logical episode across 5 distinct execution paths:

        1. EpisodeMetricsAccumulator directly
        2. extract_episode_metrics
        3. SB3CallbackAdapter
        4. PPOTrainer training path
        5. SACTrainer training path

        Verify semantic equivalence for: reward, length, success, collision, terminated, truncated.
        """
        step_1_info = {"step": 1, "collision": False}
        step_2_info = {"step": 2, "collision": False, "success": True}

        # 1. EpisodeMetricsAccumulator
        acc = EpisodeMetricsAccumulator()
        acc.record_step(reward=1.0, terminated=False, truncated=False, info=step_1_info)
        acc.record_step(reward=2.0, terminated=True, truncated=False, info=step_2_info)
        m_acc = acc.finish()

        # 2. extract_episode_metrics
        m_extract = extract_episode_metrics(
            reward=3.0,
            length=2,
            terminated=True,
            truncated=False,
            info=step_2_info,
            step_infos=[step_1_info, step_2_info],
        )

        # 3. SB3CallbackAdapter
        logger = MetricLoggerCallback()
        adapter = SB3CallbackAdapter(callbacks=[logger])
        adapter.num_timesteps = 1
        adapter.locals = {"dones": [False], "rewards": [1.0], "infos": [step_1_info]}
        adapter._on_step()
        adapter.num_timesteps = 2
        adapter.locals = {"dones": [True], "rewards": [2.0], "infos": [step_2_info]}
        adapter._on_step()
        assert len(logger.episode_metrics) == 1
        m_sb3 = logger.episode_metrics[0]

        # 4 & 5. PPO and SAC training paths
        class ReplayEnv(gym.Env):
            def __init__(self) -> None:
                super().__init__()
                self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
                self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
                self.step_idx = 0

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.step_idx = 0
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action):
                self.step_idx += 1
                if self.step_idx == 1:
                    return np.zeros(2, dtype=np.float32), 1.0, False, False, dict(step_1_info)
                return np.zeros(2, dtype=np.float32), 2.0, True, False, dict(step_2_info)

        ppo_result = _run_trainer_e2e(
            PPOTrainer, ReplayEnv(), tmp_path, "ppo_consistency", total_timesteps=2
        )
        ppo_records, _ = _read_training_artifacts(ppo_result)
        m_ppo = ppo_records[0]

        sac_result = _run_trainer_e2e(
            SACTrainer, ReplayEnv(), tmp_path, "sac_consistency", total_timesteps=2
        )
        sac_records, _ = _read_training_artifacts(sac_result)
        m_sac = sac_records[0]

        # Verify semantic equivalence across all 5 paths
        for path_name, m_path in [
            ("accumulator", m_acc),
            ("extract", m_extract),
            ("sb3", m_sb3),
        ]:
            assert m_path.reward == 3.0, f"{path_name} reward mismatch"
            assert m_path.length == 2, f"{path_name} length mismatch"
            assert m_path.success is True, f"{path_name} success mismatch"
            assert m_path.collision is False, f"{path_name} collision mismatch"
            assert m_path.terminated is True, f"{path_name} terminated mismatch"
            assert m_path.truncated is False, f"{path_name} truncated mismatch"

        for trainer_name, rec in [("PPO", m_ppo), ("SAC", m_sac)]:
            assert rec.reward == 3.0, f"{trainer_name} reward mismatch"
            assert rec.length == 2, f"{trainer_name} length mismatch"
            assert rec.success is True, f"{trainer_name} success mismatch"
            assert rec.collision is False, f"{trainer_name} collision mismatch"


class TestFlagDistinctions:
    """Explicit regression coverage for TEST O, P, Q."""

    def test_matrix_o_terminated_vs_truncated_distinction(self) -> None:
        """TEST O: Terminated vs truncated distinction."""
        m_term = extract_episode_metrics(
            reward=1.0, length=5, terminated=True, truncated=False, info={}
        )
        assert m_term.terminated is True
        assert m_term.truncated is False

        m_trunc = extract_episode_metrics(
            reward=1.0, length=5, terminated=False, truncated=True, info={}
        )
        assert m_trunc.terminated is False
        assert m_trunc.truncated is True

    def test_matrix_p_terminal_info_absent(self) -> None:
        """TEST P: Terminal info absent (None / empty)."""
        m_none = extract_episode_metrics(
            reward=1.0, length=5, terminated=True, truncated=False, info=None
        )
        assert m_none.success is None
        assert m_none.collision is None

        m_empty = extract_episode_metrics(
            reward=1.0, length=5, terminated=True, truncated=False, info={}
        )
        assert m_empty.success is None
        assert m_empty.collision is None

    def test_matrix_q_terminal_info_present_outcome_fields_absent(self) -> None:
        """TEST Q: Terminal info present but outcome fields absent."""
        m = extract_episode_metrics(
            reward=1.0,
            length=5,
            terminated=True,
            truncated=False,
            info={"score": 42, "battery": 95.5},
        )
        assert m.success is None
        assert m.collision is None
        assert m.additional_metrics.get("score") == 42
        assert m.additional_metrics.get("battery") == 95.5
