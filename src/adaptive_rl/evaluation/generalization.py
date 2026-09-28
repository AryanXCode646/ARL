"""Unseen-environment generalization benchmark protocol and evaluation engine.

Provides deterministic train/test seed partitioning, generalization gap computation,
and structured benchmarking across training and unseen evaluation environments.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.evaluation.metrics import EvaluationMetrics

# Single source of truth for benchmark seed partitioning.
# Half-open intervals: [start, end) where end is exclusive.
TRAIN_SEED_START: int = 0
TRAIN_SEED_END: int = 1000  # 1000 deterministic training seeds: 0..999

TEST_SEED_START: int = 1000
TEST_SEED_END: int = 1200   # 200 deterministic unseen test seeds: 1000..1199

VALID_SPLITS: Tuple[str, ...] = ("train", "test")


def is_seed_in_split(seed: int, split: str) -> bool:
    """Check whether a given integer seed belongs to the specified split.

    Args:
        seed: The integer seed to verify.
        split: The split name ('train' or 'test').

    Returns:
        True if seed is inside the split's half-open range, False otherwise.
    """
    clean_split = split.strip().lower()
    if clean_split == "train":
        return TRAIN_SEED_START <= seed < TRAIN_SEED_END
    if clean_split == "test":
        return TEST_SEED_START <= seed < TEST_SEED_END
    raise ValueError(f"Unknown split '{split}'. Expected one of: {VALID_SPLITS}")


def validate_split_seed(seed: int, split: str) -> None:
    """Validate that a seed falls strictly inside the partition for the specified split.

    Args:
        seed: The integer seed to check.
        split: The split name ('train' or 'test').

    Raises:
        ValueError: If split is unknown or seed falls outside the split partition.
    """
    clean_split = split.strip().lower()
    if not is_seed_in_split(seed, clean_split):
        if clean_split == "train":
            raise ValueError(
                f"Seed {seed} is out of bounds for 'train' split. "
                f"Allowed range is [{TRAIN_SEED_START}, {TRAIN_SEED_END}) (end exclusive)."
            )
        elif clean_split == "test":
            raise ValueError(
                f"Seed {seed} is out of bounds for 'test' split. "
                f"Allowed range is [{TEST_SEED_START}, {TEST_SEED_END}) (end exclusive)."
            )


def get_split_seeds(split: str, num_episodes: Optional[int] = None) -> List[int]:
    """Deterministically retrieve seeds for the specified benchmark split.

    Args:
        split: 'train' or 'test'.
        num_episodes: Optional number of seeds to return. If omitted, returns all
            seeds in the split.

    Returns:
        List of integer seeds strictly within the partition.

    Raises:
        ValueError: If split is invalid or num_episodes is <= 0 or exceeds partition capacity.
    """
    clean_split = split.strip().lower()
    if clean_split == "train":
        start, end = TRAIN_SEED_START, TRAIN_SEED_END
    elif clean_split == "test":
        start, end = TEST_SEED_START, TEST_SEED_END
    else:
        raise ValueError(f"Unknown split '{split}'. Expected one of: {VALID_SPLITS}")

    max_capacity = end - start
    if num_episodes is not None:
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")
        if num_episodes > max_capacity:
            raise ValueError(
                f"Requested {num_episodes} episodes, but '{clean_split}' split capacity "
                f"is {max_capacity} seeds."
            )
        return list(range(start, start + num_episodes))

    return list(range(start, end))


@dataclass(frozen=True)
class GeneralizationGap:
    """Holds directional generalization gaps between train and unseen test distributions.

    Defined as:
        success_gap = train_success_rate - test_success_rate
        reward_gap  = train_mean_reward - test_mean_reward
    """

    success_gap: Optional[float]
    reward_gap: float
    collision_gap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert gaps to dictionary."""
        return {
            "success": round(self.success_gap, 4) if self.success_gap is not None else None,
            "reward": round(self.reward_gap, 4),
            "collision": round(self.collision_gap, 4) if self.collision_gap is not None else None,
        }


def compute_generalization_gap(
    train_metrics: EvaluationMetrics,
    test_metrics: EvaluationMetrics,
) -> GeneralizationGap:
    r"""Compute the empirical generalization gap between train and test distributions.

    Handles edge cases safely (e.g. 0% test success, 100% train success, missing rates).

    $$\Delta_{\text{success}} = \text{train\_success\_rate} - \text{test\_success\_rate}$$
    $$\Delta_{\text{reward}} = \text{train\_mean\_reward} - \text{test\_mean\_reward}$$
    $$\Delta_{\text{collision}} = \text{train\_collision\_rate} - \text{test\_collision\_rate}$$

    Args:
        train_metrics: Metrics evaluated on the training distribution.
        test_metrics: Metrics evaluated on the unseen test distribution.

    Returns:
        GeneralizationGap containing calculated directional differences.

    Raises:
        ValueError: If episodes in either metric set is <= 0 or metrics are non-finite.
    """
    if train_metrics.episodes <= 0 or test_metrics.episodes <= 0:
        raise ValueError("Cannot compute generalization gap on empty evaluation episodes.")

    if not math.isfinite(train_metrics.mean_reward) or not math.isfinite(test_metrics.mean_reward):
        raise ValueError(
            f"Non-finite mean reward detected: train={train_metrics.mean_reward}, test={test_metrics.mean_reward}"
        )

    # Success gap
    success_gap: Optional[float] = None
    if train_metrics.success_rate is not None and test_metrics.success_rate is not None:
        t_succ = float(train_metrics.success_rate)
        e_succ = float(test_metrics.success_rate)
        if not (0.0 <= t_succ <= 1.0) or not (0.0 <= e_succ <= 1.0):
            raise ValueError(f"Success rates must be in [0, 1], got train={t_succ}, test={e_succ}")
        success_gap = float(t_succ - e_succ)

    # Collision gap
    collision_gap: Optional[float] = None
    if train_metrics.collision_rate is not None and test_metrics.collision_rate is not None:
        t_coll = float(train_metrics.collision_rate)
        e_coll = float(test_metrics.collision_rate)
        collision_gap = float(t_coll - e_coll)

    # Reward gap
    reward_gap = float(train_metrics.mean_reward - test_metrics.mean_reward)

    return GeneralizationGap(
        success_gap=success_gap,
        reward_gap=reward_gap,
        collision_gap=collision_gap,
    )


class GeneralizationBenchmarkResult(BaseModel):
    """Structured report schema for the unseen-environment generalization benchmark."""

    model_config = ConfigDict(extra="ignore")

    train: Dict[str, Any] = Field(..., description="Train split evaluation metrics and seeds")
    test: Dict[str, Any] = Field(..., description="Test split evaluation metrics and seeds")
    generalization_gap: Dict[str, Any] = Field(
        ..., description="Performance gaps between train and test splits"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Benchmark run metadata (model, timestamp, config)"
    )

    def save_report(self, output_path: str | Path) -> Path:
        """Serialize benchmark results to structured JSON.

        Converts NumPy scalar types to native Python floats and ints to guarantee clean serialization.
        """
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        def _json_sanitize(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _json_sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_json_sanitize(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return _json_sanitize(obj.tolist())
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        raw_data = self.model_dump()
        sanitized_data = _json_sanitize(raw_data)

        with open(target, "w", encoding="utf-8") as f:
            json.dump(sanitized_data, f, indent=2)

        return target


def evaluate_generalization(
    algorithm: BaseAlgorithm,
    env: Optional[gym.Env] = None,
    num_episodes: int = 20,
    deterministic: bool = True,
    output_path: Optional[str | Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> GeneralizationBenchmarkResult:
    """Execute complete unseen-environment generalization benchmark.

    Evaluates the policy across the disjoint training distribution and held-out test distribution,
    computes generalization gaps, and generates structured output.

    Args:
        algorithm: Policy algorithm instance.
        env: Optional Gymnasium environment instance (created if None).
        num_episodes: Number of evaluation episodes per split.
        deterministic: Whether to use deterministic actions.
        output_path: Optional path to write JSON benchmark artifact.
        metadata: Optional metadata dict to record in the report.

    Returns:
        GeneralizationBenchmarkResult containing train metrics, test metrics, and generalization gaps.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")

    close_env = False
    if env is None:
        env = DroneNavigation3DEnv()
        close_env = True

    try:
        train_seeds = get_split_seeds("train", num_episodes=num_episodes)
        test_seeds = get_split_seeds("test", num_episodes=num_episodes)

        evaluator = Evaluator(algorithm=algorithm, env=env)

        # 1. Evaluate on training distribution
        train_metrics = evaluator.evaluate(
            num_episodes=num_episodes,
            deterministic=deterministic,
            seeds=train_seeds,
            split="train",
        )

        # 2. Evaluate on unseen test distribution
        test_metrics = evaluator.evaluate(
            num_episodes=num_episodes,
            deterministic=deterministic,
            seeds=test_seeds,
            split="test",
        )

        # 3. Compute generalization gap
        gap = compute_generalization_gap(train_metrics, test_metrics)

        train_dict: Dict[str, Any] = {
            "seeds": train_seeds,
            "episodes": train_metrics.episodes,
            "success_rate": round(train_metrics.success_rate, 4)
            if train_metrics.success_rate is not None
            else None,
            "collision_rate": round(train_metrics.collision_rate, 4)
            if train_metrics.collision_rate is not None
            else None,
            "mean_reward": round(train_metrics.mean_reward, 2),
            "mean_episode_length": round(train_metrics.mean_episode_length, 2),
        }
        if train_metrics.obstacle_collision_rate is not None:
            train_dict["obstacle_collision_rate"] = round(train_metrics.obstacle_collision_rate, 4)
        if train_metrics.boundary_collision_rate is not None:
            train_dict["boundary_collision_rate"] = round(train_metrics.boundary_collision_rate, 4)
        if train_metrics.mean_path_length is not None:
            train_dict["mean_path_length"] = round(train_metrics.mean_path_length, 2)
        if train_metrics.mean_path_efficiency is not None:
            train_dict["mean_path_efficiency"] = round(train_metrics.mean_path_efficiency, 4)

        test_dict: Dict[str, Any] = {
            "seeds": test_seeds,
            "episodes": test_metrics.episodes,
            "success_rate": round(test_metrics.success_rate, 4)
            if test_metrics.success_rate is not None
            else None,
            "collision_rate": round(test_metrics.collision_rate, 4)
            if test_metrics.collision_rate is not None
            else None,
            "mean_reward": round(test_metrics.mean_reward, 2),
            "mean_episode_length": round(test_metrics.mean_episode_length, 2),
        }
        if test_metrics.obstacle_collision_rate is not None:
            test_dict["obstacle_collision_rate"] = round(test_metrics.obstacle_collision_rate, 4)
        if test_metrics.boundary_collision_rate is not None:
            test_dict["boundary_collision_rate"] = round(test_metrics.boundary_collision_rate, 4)
        if test_metrics.mean_path_length is not None:
            test_dict["mean_path_length"] = round(test_metrics.mean_path_length, 2)
        if test_metrics.mean_path_efficiency is not None:
            test_dict["mean_path_efficiency"] = round(test_metrics.mean_path_efficiency, 4)

        meta: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episodes_per_split": num_episodes,
            "deterministic": deterministic,
        }
        if metadata:
            meta.update(metadata)

        result = GeneralizationBenchmarkResult(
            train=train_dict,
            test=test_dict,
            generalization_gap=gap.to_dict(),
            metadata=meta,
        )

        if output_path is not None:
            result.save_report(output_path)

        return result
    finally:
        if close_env:
            env.close()


__all__ = [
    "GeneralizationBenchmarkResult",
    "GeneralizationGap",
    "TEST_SEED_END",
    "TEST_SEED_START",
    "TRAIN_SEED_END",
    "TRAIN_SEED_START",
    "VALID_SPLITS",
    "compute_generalization_gap",
    "evaluate_generalization",
    "get_split_seeds",
    "is_seed_in_split",
    "validate_split_seed",
]
