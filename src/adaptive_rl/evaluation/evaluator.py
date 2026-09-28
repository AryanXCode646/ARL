"""Evaluation engine for AdaptiveRL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.metrics import EvaluationMetrics


@dataclass(frozen=True)
class EpisodeEvaluationRecord:
    """Record for a single evaluation episode."""

    episode_index: int
    seed: Optional[int]
    return_value: float
    episode_length: int
    success: bool
    collision: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "seed": self.seed,
            "return": self.return_value,
            "episode_length": self.episode_length,
            "success": self.success,
            "collision": self.collision,
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
        successes: List[bool] = []
        collisions: List[bool] = []

        for ep in range(num_episodes):
            seed = (base_seed + ep) if base_seed is not None else None
            obs, info = self.env.reset(seed=seed)
            ep_reward = 0.0
            ep_length = 0
            done = False
            last_info = dict(info or {})

            while not done:
                action, _ = self.algorithm.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                last_info = step_info
                done = terminated or truncated

            is_success = bool(last_info.get("success", False))
            is_collision = bool(last_info.get("collision", False))

            rewards.append(ep_reward)
            lengths.append(ep_length)
            successes.append(is_success)
            collisions.append(is_collision)

            self.last_episode_records.append(
                EpisodeEvaluationRecord(
                    episode_index=ep,
                    seed=seed,
                    return_value=ep_reward,
                    episode_length=ep_length,
                    success=is_success,
                    collision=is_collision,
                )
            )

        mean_rew = float(np.mean(rewards))
        std_rew = float(np.std(rewards))
        min_rew = float(np.min(rewards))
        max_rew = float(np.max(rewards))
        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))

        succ_rate = float(sum(successes) / num_episodes)
        coll_rate = float(sum(collisions) / num_episodes)

        return EvaluationMetrics(
            episodes=num_episodes,
            mean_reward=mean_rew,
            std_reward=std_rew,
            min_reward=min_rew,
            max_reward=max_rew,
            success_rate=succ_rate,
            collision_rate=coll_rate,
            mean_episode_length=mean_len,
            std_episode_length=std_len,
            additional_metrics={
                "all_rewards": rewards,
                "all_lengths": lengths,
                "deterministic": deterministic,
                "base_seed": base_seed,
            },
        )

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


__all__ = [
    "EpisodeEvaluationRecord",
    "Evaluator",
]
