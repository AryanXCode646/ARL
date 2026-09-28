"""Evaluation engine for AdaptiveRL."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.algorithms.random_policy import RandomPolicy
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.metrics import EvaluationMetrics
from adaptive_rl.evaluation.statistics import MetricStatistics, summarize_seed_values


@dataclass(frozen=True)
class EpisodeEvaluationRecord:
    """Record for a single evaluation episode."""

    episode_index: int
    seed: Optional[int]
    return_value: float
    episode_length: int
    success: Optional[bool]
    collision: Optional[bool]
    truncated: bool
    path_length: Optional[float] = None
    episode_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "seed": self.seed,
            "return": self.return_value,
            "episode_length": self.episode_length,
            "success": self.success,
            "collision": self.collision,
            "truncated": self.truncated,
            "path_length": self.path_length,
            "episode_seed": self.episode_seed,
        }


@dataclass(frozen=True)
class SeedEvaluationSummary:
    """Episode-level summary for a single evaluation seed."""

    seed: int
    episodes: int
    success_rate: float | None
    collision_rate: float | None
    truncation_rate: float | None
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    std_episode_length: float
    path_length: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "seed": self.seed,
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "collision_rate": self.collision_rate,
            "truncation_rate": self.truncation_rate,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_episode_length": self.mean_episode_length,
            "std_episode_length": self.std_episode_length,
            "path_length": self.path_length,
        }


@dataclass(frozen=True)
class MultiSeedEvaluationResult:
    """Complete multi-seed result retaining episodes and seed-level identity."""

    seeds: list[int]
    episodes_per_seed: int
    deterministic: bool
    per_seed: list[SeedEvaluationSummary]
    episodes: list[EpisodeEvaluationRecord]
    aggregate: dict[str, MetricStatistics]
    environment: str

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "seeds": list(self.seeds),
                "seed_count": len(self.seeds),
                "episodes_per_seed": self.episodes_per_seed,
                "total_episodes": self.total_episodes,
                "deterministic": self.deterministic,
                "environment": self.environment,
                "confidence_interval": "two-sided 95% Student's t interval across seed summaries; "
                "sample standard deviation; unavailable when fewer than two values exist",
                "duplicate_seed_policy": "rejected",
            },
            "episodes": [record.to_dict() for record in self.episodes],
            "per_seed": [summary.to_dict() for summary in self.per_seed],
            "aggregate": {
                name: statistics.to_dict() for name, statistics in self.aggregate.items()
            },
        }


class Evaluator:
    """Standardized multi-episode evaluation engine for drone navigation."""

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: Optional[gym.Env] = None,
        env_name: str = "drone",
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.algorithm = algorithm
        self.env_kwargs = dict(env_kwargs or {})
        self.env_name = env_name

        if env is not None:
            self.env = env
        else:
            self.env = make_env(self.env_name, **self.env_kwargs)

        self.last_episode_records: List[EpisodeEvaluationRecord] = []

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        base_seed: Optional[int] = None,
    ) -> EvaluationMetrics:
        """Execute evaluation rollouts and compute aggregated metrics."""
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")

        self.last_episode_records = []
        rewards: List[float] = []
        lengths: List[int] = []
        successes: List[Optional[bool]] = []
        collisions: List[Optional[bool]] = []
        truncations: List[bool] = []

        for ep in range(num_episodes):
            seed = (base_seed + ep) if base_seed is not None else None
            obs, info = self.env.reset(seed=seed)
            ep_reward = 0.0
            ep_length = 0
            done = False
            last_info = dict(info or {})
            was_truncated = False
            previous_position = self._position_from_info(last_info)
            path_length = 0.0 if previous_position is not None else None

            while not done:
                action, _ = self.algorithm.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                last_info = step_info
                was_truncated = bool(truncated)
                current_position = self._position_from_info(step_info)
                if (
                    path_length is not None
                    and previous_position is not None
                    and current_position is not None
                    and current_position.shape == previous_position.shape
                ):
                    path_length += float(np.linalg.norm(current_position - previous_position))
                else:
                    path_length = None
                previous_position = current_position
                done = terminated or truncated

            success_value = last_info.get("success", last_info.get("is_success"))
            collision_value = last_info.get("collision")
            is_success = bool(success_value) if success_value is not None else None
            is_collision = bool(collision_value) if collision_value is not None else None

            rewards.append(ep_reward)
            lengths.append(ep_length)
            successes.append(is_success)
            collisions.append(is_collision)
            truncations.append(was_truncated)
            if not math.isfinite(ep_reward):
                raise ValueError(f"Episode {ep} produced a non-finite cumulative reward.")

            self.last_episode_records.append(
                EpisodeEvaluationRecord(
                    episode_index=ep,
                    seed=seed,
                    return_value=ep_reward,
                    episode_length=ep_length,
                    success=is_success,
                    collision=is_collision,
                    truncated=was_truncated,
                    path_length=path_length,
                    episode_seed=seed,
                )
            )

        mean_rew = float(np.mean(rewards))
        std_rew = float(np.std(rewards))
        min_rew = float(np.min(rewards))
        max_rew = float(np.max(rewards))
        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))

        observed_successes = [value for value in successes if value is not None]
        observed_collisions = [value for value in collisions if value is not None]
        succ_rate = (
            float(sum(observed_successes) / len(observed_successes)) if observed_successes else None
        )
        coll_rate = (
            float(sum(observed_collisions) / len(observed_collisions))
            if observed_collisions
            else None
        )
        trunc_rate = float(sum(truncations) / num_episodes)

        return EvaluationMetrics(
            episodes=num_episodes,
            mean_reward=mean_rew,
            std_reward=std_rew,
            min_reward=min_rew,
            max_reward=max_rew,
            success_rate=succ_rate,
            collision_rate=coll_rate,
            truncation_rate=trunc_rate,
            mean_episode_length=mean_len,
            std_episode_length=std_len,
            additional_metrics={
                "all_rewards": rewards,
                "all_lengths": lengths,
                "all_path_lengths": [record.path_length for record in self.last_episode_records],
                "deterministic": deterministic,
                "base_seed": base_seed,
                "truncation_count": int(sum(truncations)),
                "timeout_rate": trunc_rate,
            },
        )

    @staticmethod
    def _position_from_info(info: dict[str, Any]) -> np.ndarray | None:
        position = info.get("position")
        if position is None:
            return None
        try:
            coordinates = np.asarray(position, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if coordinates.ndim != 1 or coordinates.size == 0:
            return None
        if not np.all(np.isfinite(coordinates)):
            return None
        return coordinates

    def evaluate_seeds(
        self,
        seeds: Sequence[int],
        episodes_per_seed: int,
        deterministic: bool = True,
    ) -> MultiSeedEvaluationResult:
        """Evaluate a policy independently for each explicit seed.

        Duplicate seeds are rejected because repeated entries do not represent
        independent test conditions and would over-weight that environment.
        """
        seed_values = list(seeds)
        if not seed_values:
            raise ValueError("Evaluation seeds must not be empty.")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_values):
            raise ValueError("Evaluation seeds must be integers.")
        if any(seed < 0 for seed in seed_values):
            raise ValueError("Evaluation seeds must be non-negative.")
        if len(set(seed_values)) != len(seed_values):
            raise ValueError("Evaluation seeds must be unique; duplicate seeds are not allowed.")
        if isinstance(episodes_per_seed, bool) or not isinstance(episodes_per_seed, int):
            raise ValueError("episodes_per_seed must be an integer.")
        if episodes_per_seed <= 0:
            raise ValueError(f"episodes_per_seed must be positive, got {episodes_per_seed}.")
        if not isinstance(deterministic, bool):
            raise ValueError("deterministic must be a boolean.")

        all_records: list[EpisodeEvaluationRecord] = []
        seed_summaries: list[SeedEvaluationSummary] = []
        for seed in seed_values:
            episode_seed_base = seed * episodes_per_seed
            metrics = self.evaluate(
                num_episodes=episodes_per_seed,
                deterministic=deterministic,
                base_seed=episode_seed_base,
            )
            records = [
                replace(record, seed=seed, episode_index=index)
                for index, record in enumerate(self.last_episode_records)
            ]
            all_records.extend(records)
            path_lengths = [
                record.path_length for record in records if record.path_length is not None
            ]
            seed_summaries.append(
                SeedEvaluationSummary(
                    seed=seed,
                    episodes=metrics.episodes,
                    success_rate=metrics.success_rate,
                    collision_rate=metrics.collision_rate,
                    truncation_rate=metrics.truncation_rate,
                    mean_reward=metrics.mean_reward,
                    std_reward=metrics.std_reward,
                    mean_episode_length=metrics.mean_episode_length,
                    std_episode_length=metrics.std_episode_length,
                    path_length=(
                        float(math.fsum(path_lengths) / len(path_lengths)) if path_lengths else None
                    ),
                )
            )

        metric_names = (
            "mean_reward",
            "success_rate",
            "collision_rate",
            "truncation_rate",
            "mean_episode_length",
            "std_reward",
            "std_episode_length",
            "path_length",
        )
        aggregate = {
            name: summarize_seed_values([getattr(summary, name) for summary in seed_summaries])
            for name in metric_names
        }
        self.last_episode_records = all_records
        return MultiSeedEvaluationResult(
            seeds=seed_values,
            episodes_per_seed=episodes_per_seed,
            deterministic=deterministic,
            per_seed=seed_summaries,
            episodes=all_records,
            aggregate=aggregate,
            environment=self.env_name,
        )

    @staticmethod
    def save_multiseed_report(
        result: MultiSeedEvaluationResult,
        json_path: str | Path,
        csv_path: str | Path,
    ) -> tuple[Path, Path]:
        """Write the complete multi-seed result to stable JSON and aggregate CSV."""
        json_target = Path(json_path)
        csv_target = Path(csv_path)
        json_target.parent.mkdir(parents=True, exist_ok=True)
        csv_target.parent.mkdir(parents=True, exist_ok=True)

        with json_target.open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, allow_nan=False)

        columns = [
            "metric",
            "mean",
            "std",
            "ci95_lower",
            "ci95_upper",
            "sample_count",
            "seed_count",
            "episodes_per_seed",
            "total_episodes",
        ]
        with csv_target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for metric, statistics in result.aggregate.items():
                writer.writerow(
                    {
                        "metric": metric,
                        **statistics.to_dict(),
                        "seed_count": len(result.seeds),
                        "episodes_per_seed": result.episodes_per_seed,
                        "total_episodes": result.total_episodes,
                    }
                )
        return json_target, csv_target

    @staticmethod
    def save_report(
        metrics: EvaluationMetrics,
        output_path: str | Path,
    ) -> Path:
        """Serialize evaluation metrics to JSON."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "episodes": metrics.episodes,
            "mean_reward": round(metrics.mean_reward, 2),
            "std_reward": round(metrics.std_reward, 2),
            "min_reward": round(metrics.min_reward, 2),
            "max_reward": round(metrics.max_reward, 2),
            "success_rate": round(metrics.success_rate, 4)
            if metrics.success_rate is not None
            else None,
            "collision_rate": round(metrics.collision_rate, 4)
            if metrics.collision_rate is not None
            else None,
            "truncation_rate": round(metrics.truncation_rate, 4)
            if metrics.truncation_rate is not None
            else None,
            "mean_episode_length": round(metrics.mean_episode_length, 2),
            "std_episode_length": round(metrics.std_episode_length, 2),
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return target

    def close(self) -> None:
        """Close evaluation environment."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()


def evaluate_random_policy(
    env: Optional[gym.Env] = None,
    num_episodes: int = 20,
    base_seed: Optional[int] = 42,
) -> EvaluationMetrics:
    """Evaluate an untrained uniform-random action baseline policy under controlled seeds.

    Args:
        env: Optional Gymnasium environment instance (defaults to standard DroneNavigation3DEnv).
        num_episodes: Total evaluation episodes to run.
        base_seed: Deterministic base seed.

    Returns:
        EvaluationMetrics containing empirical benchmark metrics.
    """
    close_env = False
    if env is None:
        env = DroneNavigation3DEnv()
        close_env = True
    try:
        policy = RandomPolicy(action_space=env.action_space, seed=base_seed)
        evaluator = Evaluator(algorithm=policy, env=env)
        return evaluator.evaluate(
            num_episodes=num_episodes,
            deterministic=False,
            base_seed=base_seed,
        )
    finally:
        if close_env:
            env.close()


def evaluate_ppo_policy(
    algorithm: BaseAlgorithm,
    env: Optional[gym.Env] = None,
    num_episodes: int = 20,
    base_seed: Optional[int] = 42,
    deterministic: bool = True,
) -> EvaluationMetrics:
    """Evaluate a trained PPO policy under controlled benchmark seeds.

    Args:
        algorithm: Initialized or loaded PPOAlgorithm wrapper.
        env: Optional Gymnasium environment instance (defaults to standard DroneNavigation3DEnv).
        num_episodes: Total evaluation episodes to run.
        base_seed: Deterministic base seed.
        deterministic: Whether to use deterministic mode.

    Returns:
        EvaluationMetrics containing empirical benchmark metrics.
    """
    close_env = False
    if env is None:
        env = DroneNavigation3DEnv()
        close_env = True
    try:
        evaluator = Evaluator(algorithm=algorithm, env=env)
        return evaluator.evaluate(
            num_episodes=num_episodes,
            deterministic=deterministic,
            base_seed=base_seed,
        )
    finally:
        if close_env:
            env.close()


def compare_policies(
    ppo_algorithm: BaseAlgorithm,
    random_policy: Optional[BaseAlgorithm] = None,
    env: Optional[gym.Env] = None,
    num_episodes: int = 20,
    base_seed: Optional[int] = 42,
) -> Dict[str, EvaluationMetrics]:
    """Execute head-to-head evaluation between trained PPO and Random baseline under identical seeds."""
    close_env = False
    if env is None:
        env = DroneNavigation3DEnv()
        close_env = True
    try:
        if random_policy is None:
            random_policy = RandomPolicy(action_space=env.action_space, seed=base_seed)

        ppo_eval = Evaluator(algorithm=ppo_algorithm, env=env)
        ppo_metrics = ppo_eval.evaluate(
            num_episodes=num_episodes,
            deterministic=True,
            base_seed=base_seed,
        )

        rand_eval = Evaluator(algorithm=random_policy, env=env)
        rand_metrics = rand_eval.evaluate(
            num_episodes=num_episodes,
            deterministic=False,
            base_seed=base_seed,
        )

        return {
            "PPO": ppo_metrics,
            "Random Policy": rand_metrics,
        }
    finally:
        if close_env:
            env.close()


def run_obstacle_density_experiment(
    algorithm: BaseAlgorithm,
    obstacle_counts: Sequence[int] = (4, 6, 8),
    episodes_per_density: int = 10,
    base_seed: int = 42,
    bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0),
    output_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Evaluate a trained agent across varied obstacle densities (e.g. 4, 6, 8 obstacles)."""
    results: List[Dict[str, Any]] = []

    for count in obstacle_counts:
        env = DroneNavigation3DEnv(bounds=bounds, num_obstacles=count)
        try:
            evaluator = Evaluator(algorithm=algorithm, env=env)
            metrics = evaluator.evaluate(
                num_episodes=episodes_per_density,
                deterministic=True,
                base_seed=base_seed,
            )
            entry: Dict[str, Any] = {
                "obstacle_count": int(count),
                "episodes": int(episodes_per_density),
                "success_rate": round(float(metrics.success_rate or 0.0), 4),
                "collision_rate": round(float(metrics.collision_rate or 0.0), 4),
                "mean_reward": round(float(metrics.mean_reward), 2),
                "mean_episode_length": round(float(metrics.mean_episode_length), 2),
            }
            results.append(entry)
        finally:
            env.close()

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


__all__ = [
    "EpisodeEvaluationRecord",
    "Evaluator",
    "compare_policies",
    "evaluate_ppo_policy",
    "evaluate_random_policy",
    "run_obstacle_density_experiment",
]
