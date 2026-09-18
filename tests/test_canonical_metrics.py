"""Regression and contract tests for the canonical EpisodeMetrics extraction and consumers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.evaluation.generalization import (
    GeneralizationDistribution,
    GeneralizationEvaluator,
)
from adaptive_rl.metrics import (
    EpisodeMetrics,
    extract_episode_metrics,
)
from adaptive_rl.training.callbacks import (
    MetricLoggerCallback,
    SB3CallbackAdapter,
)


class MockPolicyAlgo(BaseAlgorithm):
    """Minimal policy algorithm returning deterministic discrete or continuous zero actions."""

    def __init__(self, action_space: Optional[gym.Space] = None) -> None:
        self.action_space = action_space
        self.name = "mock_policy"

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, Any]:
        if self.action_space is not None and hasattr(self.action_space, "shape"):
            shape = getattr(self.action_space, "shape", ())
            return np.zeros(shape, dtype=np.float32), {}
        return 0, {}

    def train(self, total_timesteps: int, callback: Any = None) -> None:
        pass

    def save(self, path: Any) -> None:
        pass

    def load(self, path: Any, env: Optional[gym.Env] = None) -> None:
        pass


class MockStepEnv(gym.Env):
    """Scriptable environment returning programmed step outcomes."""

    def __init__(self, step_outcomes: list[tuple[float, bool, bool, Dict[str, Any]]]) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.step_outcomes = list(step_outcomes)
        self.step_idx = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.step_idx < len(self.step_outcomes):
            rew, term, trunc, info = self.step_outcomes[self.step_idx]
            self.step_idx += 1
        else:
            rew, term, trunc, info = 0.0, True, False, {}
        return np.zeros(2, dtype=np.float32), rew, term, trunc, dict(info)


# ---------------------------------------------------------------------------
# Core Dataclass & Precedence Unit Tests
# ---------------------------------------------------------------------------


def test_episode_metrics_is_immutable_dataclass() -> None:
    """EpisodeMetrics must be an immutable frozen dataclass."""
    assert is_dataclass(EpisodeMetrics)
    m = EpisodeMetrics(
        reward=10.0,
        length=5,
        success=True,
        collision=False,
        terminated=True,
        truncated=False,
        additional_metrics={"step": 5},
    )
    with pytest.raises(FrozenInstanceError):
        m.reward = 20.0  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        m.success = False  # type: ignore[misc]


def test_episode_metrics_mutation_isolation() -> None:
    """Mutating caller's dictionary after extraction must not alter EpisodeMetrics."""
    raw_info = {"battery": 80.0, "success": True}
    m = extract_episode_metrics(
        reward=5.0,
        length=10,
        terminated=True,
        truncated=False,
        info=raw_info,
    )
    assert m.additional_metrics["battery"] == 80.0
    raw_info["battery"] = 10.0
    assert m.additional_metrics["battery"] == 80.0


def test_episode_success_true() -> None:
    """episode_success=True extracts as success=True."""
    m = extract_episode_metrics(
        reward=1.0,
        length=1,
        terminated=True,
        truncated=False,
        info={"episode_success": True},
    )
    assert m.success is True


def test_success_false() -> None:
    """Explicit success=False must remain False."""
    m = extract_episode_metrics(
        reward=0.0,
        length=10,
        terminated=True,
        truncated=False,
        info={"success": False},
    )
    assert m.success is False


def test_missing_success_returns_none() -> None:
    """Missing success information must remain None, never coerced to False."""
    m1 = extract_episode_metrics(1.0, 5, True, False, info={})
    m2 = extract_episode_metrics(1.0, 5, True, False, info=None)
    m3 = extract_episode_metrics(1.0, 5, True, False, info={"battery": 50})
    assert m1.success is None
    assert m2.success is None
    assert m3.success is None


def test_episode_success_takes_precedence_over_success() -> None:
    """Precedence MUST be episode_success -> success -> is_success."""
    # episode_success=True overrides success=False
    m1 = extract_episode_metrics(
        1.0,
        5,
        True,
        False,
        info={"episode_success": True, "success": False, "is_success": False},
    )
    assert m1.success is True

    # episode_success=False overrides success=True (explicit False remains False)
    m2 = extract_episode_metrics(
        1.0,
        5,
        True,
        False,
        info={"episode_success": False, "success": True, "is_success": True},
    )
    assert m2.success is False

    # success overrides is_success when episode_success is absent
    m3 = extract_episode_metrics(
        1.0,
        5,
        True,
        False,
        info={"success": True, "is_success": False},
    )
    assert m3.success is True

    m4 = extract_episode_metrics(
        1.0,
        5,
        True,
        False,
        info={"success": False, "is_success": True},
    )
    assert m4.success is False

    # is_success used when others are absent
    m5 = extract_episode_metrics(1.0, 5, True, False, info={"is_success": True})
    assert m5.success is True

    m6 = extract_episode_metrics(1.0, 5, True, False, info={"is_success": False})
    assert m6.success is False


def test_collision_true() -> None:
    """collision=True extracts as collision=True."""
    m = extract_episode_metrics(
        -10.0,
        3,
        True,
        False,
        info={"collision": True},
    )
    assert m.collision is True


def test_missing_collision_returns_none() -> None:
    """Missing collision information must remain None, never coerced to False."""
    m1 = extract_episode_metrics(0.0, 5, True, False, info={})
    m2 = extract_episode_metrics(0.0, 5, True, False, info=None)
    m3 = extract_episode_metrics(0.0, 5, True, False, info={"other": 1})
    assert m1.collision is None
    assert m2.collision is None
    assert m3.collision is None


def test_collision_alias_precedence() -> None:
    """Collision precedence MUST be collision -> is_collision -> had_collision."""
    # collision=True overrides others
    m1 = extract_episode_metrics(
        0.0,
        5,
        True,
        False,
        info={"collision": True, "is_collision": False, "had_collision": False},
    )
    assert m1.collision is True

    # collision=False overrides others (explicit False remains False)
    m2 = extract_episode_metrics(
        0.0,
        5,
        True,
        False,
        info={"collision": False, "is_collision": True, "had_collision": True},
    )
    assert m2.collision is False

    # is_collision overrides had_collision
    m3 = extract_episode_metrics(
        0.0,
        5,
        True,
        False,
        info={"is_collision": True, "had_collision": False},
    )
    assert m3.collision is True

    m4 = extract_episode_metrics(
        0.0,
        5,
        True,
        False,
        info={"is_collision": False, "had_collision": True},
    )
    assert m4.collision is False

    # had_collision used when others are absent
    m5 = extract_episode_metrics(0.0, 5, True, False, info={"had_collision": True})
    assert m5.collision is True

    m6 = extract_episode_metrics(0.0, 5, True, False, info={"had_collision": False})
    assert m6.collision is False


def test_terminated_truncated_preserved_exactly() -> None:
    """terminated and truncated flags must be preserved exactly from arguments."""
    for term in (True, False):
        for trunc in (True, False):
            m = extract_episode_metrics(
                reward=10.0,
                length=5,
                terminated=term,
                truncated=trunc,
                info={"TimeLimit.truncated": not trunc},  # Ignored: info must not override
            )
            assert m.terminated is term
            assert m.truncated is trunc


def test_environment_specific_metrics_preserved_in_additional_metrics() -> None:
    """Environment-specific telemetry must be retained in additional_metrics."""
    info = {
        "battery_remaining": 45.2,
        "traffic_departures": 18,
        "wind_vector": (1.0, -0.5, 0.0),
        "success": True,
        "collision": False,
        "episode_success": True,
    }
    m = extract_episode_metrics(100.0, 20, True, False, info=info)
    assert m.additional_metrics["battery_remaining"] == 45.2
    assert m.additional_metrics["traffic_departures"] == 18
    assert m.additional_metrics["wind_vector"] == (1.0, -0.5, 0.0)
    # Core outcome keys should not clutter additional_metrics
    assert "success" not in m.additional_metrics
    assert "collision" not in m.additional_metrics
    assert "episode_success" not in m.additional_metrics


def test_explicit_false_distinguishable_from_missing_none() -> None:
    """Verify explicit False is distinct from missing None across success and collision."""
    m_false = extract_episode_metrics(
        1.0,
        5,
        True,
        False,
        info={"success": False, "collision": False},
    )
    m_none = extract_episode_metrics(1.0, 5, True, False, info={})

    assert m_false.success is False
    assert m_false.collision is False
    assert m_none.success is None
    assert m_none.collision is None

    assert m_false.success is not m_none.success
    assert m_false.collision is not m_none.collision


def test_no_heuristic_inference() -> None:
    """Do not infer success/collision from rewards, termination status, or episode length."""
    # Huge positive reward and natural termination without success info remains None
    m1 = extract_episode_metrics(1000.0, 5, terminated=True, truncated=False, info={})
    assert m1.success is None
    assert m1.collision is None

    # Huge negative penalty without collision info remains None
    m2 = extract_episode_metrics(-500.0, 1, terminated=True, truncated=False, info={})
    assert m2.success is None
    assert m2.collision is None

    # Max step truncation without success info remains None
    m3 = extract_episode_metrics(0.0, 1000, terminated=False, truncated=True, info={})
    assert m3.success is None
    assert m3.collision is None


# ---------------------------------------------------------------------------
# Training Integration Unit Tests (PPO & SAC)
# ---------------------------------------------------------------------------


def test_ppo_consumes_canonical_episode_metrics() -> None:
    """Verify SB3CallbackAdapter extracts EpisodeMetrics and MetricLoggerCallback consumes them."""
    logger = MetricLoggerCallback()
    adapter = SB3CallbackAdapter(callbacks=[logger])

    # Simulate an episode completion step from Stable-Baselines3
    adapter.locals = {
        "dones": [True],
        "rewards": [15.0],
        "infos": [{"success": True, "collision": False, "custom_metric": 42}],
    }
    adapter.num_timesteps = 10

    # Execute step hook
    adapter._on_step()

    assert logger.total_episodes == 1
    assert len(logger.episode_metrics) == 1
    m = logger.episode_metrics[0]
    assert isinstance(m, EpisodeMetrics)
    assert m.reward == 15.0
    assert m.length == 1
    assert m.success is True
    assert m.collision is False
    assert m.additional_metrics["custom_metric"] == 42
    assert logger.successes == 1
    assert logger.collisions == 0


def test_sac_consumes_canonical_episode_metrics() -> None:
    """Verify SACTrainer and SB3CallbackAdapter extract and consume canonical EpisodeMetrics."""
    logger = MetricLoggerCallback()
    adapter = SB3CallbackAdapter(callbacks=[logger])

    # Simulate multi-step SAC episode ending in collision
    adapter.locals = {
        "dones": [False],
        "rewards": [1.0],
        "infos": [{}],
    }
    adapter.num_timesteps = 1
    adapter._on_step()

    adapter.locals = {
        "dones": [True],
        "rewards": [-10.0],
        "infos": [{"collision": True, "success": False}],
    }
    adapter.num_timesteps = 2
    adapter._on_step()

    assert logger.total_episodes == 1
    assert len(logger.episode_metrics) == 1
    m = logger.episode_metrics[0]
    assert isinstance(m, EpisodeMetrics)
    assert m.reward == -9.0
    assert m.length == 2
    assert m.success is False
    assert m.collision is True
    assert logger.successes == 0
    assert logger.collisions == 1


# ---------------------------------------------------------------------------
# Evaluator & GeneralizationEvaluator Integration Tests
# ---------------------------------------------------------------------------


def test_evaluator_consumes_canonical_episode_metrics() -> None:
    """Verify Evaluator creates canonical EpisodeMetrics and aggregates them accurately."""
    # Episode 1: Success (reward 10, length 2)
    # Episode 2: Collision (reward -5, length 1)
    step_trace = [
        # Episode 1: step 0, then step 1 (terminal success)
        (5.0, False, False, {}),
        (5.0, True, False, {"success": True, "collision": False}),
        # Episode 2: step 0 (terminal collision)
        (-5.0, True, False, {"collision": True, "success": False}),
    ]
    env = MockStepEnv(step_trace)
    algo = MockPolicyAlgo()
    evaluator = Evaluator(algorithm=algo, env=env)

    metrics = evaluator.evaluate(num_episodes=2, deterministic=True)

    # Evaluator records EpisodeMetrics objects
    assert len(evaluator.last_episode_metrics) == 2
    assert all(isinstance(m, EpisodeMetrics) for m in evaluator.last_episode_metrics)

    m1, m2 = evaluator.last_episode_metrics
    assert m1.reward == 10.0
    assert m1.length == 2
    assert m1.success is True
    assert m1.collision is False

    assert m2.reward == -5.0
    assert m2.length == 1
    assert m2.success is False
    assert m2.collision is True

    # Aggregated EvaluationMetrics must reflect the canonical metrics
    assert metrics.episodes == 2
    assert metrics.mean_reward == 2.5  # (10 + -5) / 2
    assert metrics.success_rate == 0.5
    assert metrics.collision_rate == 0.5
    env.close()


def test_evaluator_missing_metrics_semantics() -> None:
    """Verify Evaluator preserves None when metrics are unavailable."""
    # Environment with no success or collision keys (e.g. classical continuous control)
    step_trace = [
        (1.0, True, False, {"arbitrary": "value"}),
    ]
    env = MockStepEnv(step_trace)
    algo = MockPolicyAlgo()
    evaluator = Evaluator(algorithm=algo, env=env)

    metrics = evaluator.evaluate(num_episodes=1, deterministic=True)
    m = evaluator.last_episode_metrics[0]
    assert m.success is None
    assert m.collision is None

    # Rates must be None, NOT collapsed to 0.0
    assert metrics.success_rate is None
    assert metrics.collision_rate is None
    env.close()


def test_generalization_evaluator_consumes_canonical_episode_metrics() -> None:
    """Verify GeneralizationEvaluator creates and aggregates canonical EpisodeMetrics."""
    # Distribution of 2 train seeds and 2 test seeds
    distribution = GeneralizationDistribution(
        train_seeds=[10, 11],
        test_seeds=[20, 21],
    )

    # Step trace: each episode is 1 step
    # Seed 10: success
    # Seed 11: collision
    # Seed 20: success
    # Seed 21: failure (explicit False, no collision)
    step_trace = [
        (10.0, True, False, {"success": True, "collision": False}),
        (-10.0, True, False, {"collision": True, "success": False}),
        (10.0, True, False, {"success": True, "collision": False}),
        (0.0, True, False, {"success": False, "collision": False}),
    ]
    env = MockStepEnv(step_trace)
    algo = MockPolicyAlgo()
    gen_eval = GeneralizationEvaluator(algorithm=algo, env=env)

    report = gen_eval.evaluate_generalization(
        distribution=distribution,
        experiment_name="gen_test",
        deterministic=True,
    )

    assert len(gen_eval.last_train_metrics) == 2
    assert len(gen_eval.last_test_metrics) == 2
    assert all(isinstance(m, EpisodeMetrics) for m in gen_eval.last_train_metrics)
    assert all(isinstance(m, EpisodeMetrics) for m in gen_eval.last_test_metrics)

    # Train: 1 success, 1 collision -> success_rate = 0.5, collision_rate = 0.5
    assert report.train_metrics.success_rate == 0.5
    assert report.train_metrics.collision_rate == 0.5

    # Test: 1 success, 1 failure -> success_rate = 0.5, collision_rate = 0.0
    assert report.test_metrics.success_rate == 0.5
    assert report.test_metrics.collision_rate == 0.0

    gen_eval.close()
