"""Evaluation engine for AdaptiveRL."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.algorithms.random_policy import RandomPolicy
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.metrics import (
    EvaluationMetrics,
    StandardizedExperimentMetrics,
    compute_trajectory_metrics,
)


@dataclass(frozen=True)
class EpisodeEvaluationRecord:
    """Record for a single evaluation episode."""

    episode_index: int
    seed: Optional[int]
    return_value: float
    episode_length: int
    success: bool
    collision: bool
    collision_type: Optional[str] = None
    path_length: Optional[float] = None
    straight_line_distance: Optional[float] = None
    path_efficiency: Optional[float] = None
    min_obstacle_clearance: Optional[float] = None
    max_velocity: Optional[float] = None
    max_acceleration: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "episode_index": self.episode_index,
            "seed": self.seed,
            "return": self.return_value,
            "episode_length": self.episode_length,
            "success": self.success,
            "collision": self.collision,
        }
        if self.collision_type is not None:
            data["collision_type"] = self.collision_type
        if self.path_length is not None:
            data["path_length"] = self.path_length
        if self.straight_line_distance is not None:
            data["straight_line_distance"] = self.straight_line_distance
        if self.path_efficiency is not None:
            data["path_efficiency"] = self.path_efficiency
        if self.min_obstacle_clearance is not None:
            data["min_obstacle_clearance"] = self.min_obstacle_clearance
        if self.max_velocity is not None:
            data["max_velocity"] = self.max_velocity
        if self.max_acceleration is not None:
            data["max_acceleration"] = self.max_acceleration
        return data


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
        obstacle_collisions: List[bool] = []
        boundary_collisions: List[bool] = []

        path_lengths: List[float] = []
        straight_line_dists: List[float] = []
        path_efficiencies: List[float] = []
        min_clearances: List[float] = []
        max_velocities: List[float] = []
        max_accelerations: List[float] = []

        for ep in range(num_episodes):
            seed = (base_seed + ep) if base_seed is not None else None
            obs, info = self.env.reset(seed=seed)
            ep_reward = 0.0
            ep_length = 0
            done = False
            last_info = dict(info or {})

            # Trajectory tracking for trajectory-quality and safety metrics
            positions: List[np.ndarray] = []
            velocities: List[np.ndarray] = []
            accelerations: List[np.ndarray] = []

            init_pos = last_info.get("position")
            if init_pos is not None:
                positions.append(np.asarray(init_pos, dtype=np.float64).copy())
            init_vel = last_info.get("velocity")
            if init_vel is not None:
                velocities.append(np.asarray(init_vel, dtype=np.float64).copy())
            init_acc = last_info.get("acceleration")
            if init_acc is not None:
                accelerations.append(np.asarray(init_acc, dtype=np.float64).copy())

            goal = last_info.get("goal")
            unwrapped_env = getattr(self.env, "unwrapped", self.env)
            obstacles = getattr(unwrapped_env, "_obstacles", None)
            if obstacles is None:
                obstacles = getattr(unwrapped_env, "obstacles", None)

            while not done:
                if self.algorithm is not None:
                    action, _ = self.algorithm.predict(obs, deterministic=deterministic)
                elif hasattr(self.env, "action_space") and self.env.action_space is not None:
                    action = self.env.action_space.sample()
                else:
                    action = np.zeros(3, dtype=np.float32)

                obs, reward, terminated, truncated, step_info = self.env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                last_info = dict(step_info or {})
                done = terminated or truncated

                # Track position, velocity, and acceleration
                step_pos = last_info.get("position")
                if step_pos is not None:
                    positions.append(np.asarray(step_pos, dtype=np.float64).copy())
                step_vel = last_info.get("velocity")
                if step_vel is not None:
                    velocities.append(np.asarray(step_vel, dtype=np.float64).copy())
                step_acc = last_info.get("acceleration")
                if step_acc is not None:
                    accelerations.append(np.asarray(step_acc, dtype=np.float64).copy())
                elif hasattr(unwrapped_env, "kinematics"):
                    accelerations.append(unwrapped_env.kinematics.state.acceleration.copy())

            is_success = bool(last_info.get("success", False))
            is_collision = bool(last_info.get("collision", False))
            collision_type = str(last_info.get("collision_type", "none"))

            is_obs_coll = is_collision and (collision_type == "obstacle")
            is_bound_coll = is_collision and collision_type.startswith("boundary")

            rewards.append(ep_reward)
            lengths.append(ep_length)
            successes.append(is_success)
            collisions.append(is_collision)
            obstacle_collisions.append(is_obs_coll)
            boundary_collisions.append(is_bound_coll)

            traj_metrics = compute_trajectory_metrics(
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
                goal=goal,
                obstacles=obstacles,
            )

            ep_path_len = traj_metrics["path_length"]
            ep_straight_dist = traj_metrics["straight_line_distance"]
            ep_path_eff = traj_metrics["path_efficiency"]
            ep_min_clear = traj_metrics["min_obstacle_clearance"]
            ep_max_vel = traj_metrics["max_velocity"]
            ep_max_acc = traj_metrics["max_acceleration"]

            if ep_path_len is not None:
                path_lengths.append(ep_path_len)
            if ep_straight_dist is not None:
                straight_line_dists.append(ep_straight_dist)
            if ep_path_eff is not None:
                path_efficiencies.append(ep_path_eff)
            if ep_min_clear is not None:
                min_clearances.append(ep_min_clear)
            if ep_max_vel is not None:
                max_velocities.append(ep_max_vel)
            if ep_max_acc is not None:
                max_accelerations.append(ep_max_acc)

            self.last_episode_records.append(
                EpisodeEvaluationRecord(
                    episode_index=ep,
                    seed=seed,
                    return_value=ep_reward,
                    episode_length=ep_length,
                    success=is_success,
                    collision=is_collision,
                    collision_type=collision_type if is_collision else None,
                    path_length=ep_path_len,
                    straight_line_distance=ep_straight_dist,
                    path_efficiency=ep_path_eff,
                    min_obstacle_clearance=ep_min_clear,
                    max_velocity=ep_max_vel,
                    max_acceleration=ep_max_acc,
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

        obs_coll_count = int(sum(obstacle_collisions))
        bound_coll_count = int(sum(boundary_collisions))
        obs_coll_rate = float(obs_coll_count / num_episodes)
        bound_coll_rate = float(bound_coll_count / num_episodes)

        mean_path_len = float(np.mean(path_lengths)) if path_lengths else None
        std_path_len = float(np.std(path_lengths)) if path_lengths else None
        mean_straight_dist = float(np.mean(straight_line_dists)) if straight_line_dists else None
        mean_path_eff = float(np.mean(path_efficiencies)) if path_efficiencies else None
        mean_min_clear = float(np.mean(min_clearances)) if min_clearances else None
        mean_max_vel = float(np.mean(max_velocities)) if max_velocities else None
        mean_max_acc = float(np.mean(max_accelerations)) if max_accelerations else None

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
            mean_path_length=mean_path_len,
            std_path_length=std_path_len,
            mean_straight_line_distance=mean_straight_dist,
            mean_path_efficiency=mean_path_eff,
            mean_min_obstacle_clearance=mean_min_clear,
            mean_max_velocity=mean_max_vel,
            mean_max_acceleration=mean_max_acc,
            obstacle_collision_rate=obs_coll_rate,
            boundary_collision_rate=bound_coll_rate,
            obstacle_collision_count=obs_coll_count,
            boundary_collision_count=bound_coll_count,
            additional_metrics={
                "all_rewards": rewards,
                "all_lengths": lengths,
                "deterministic": deterministic,
                "base_seed": base_seed,
                "mean_path_length": mean_path_len,
                "path_length": mean_path_len,
                "std_path_length": std_path_len,
                "straight_line_distance": mean_straight_dist,
                "mean_straight_line_distance": mean_straight_dist,
                "path_efficiency": mean_path_eff,
                "mean_path_efficiency": mean_path_eff,
                "min_obstacle_clearance": mean_min_clear,
                "mean_min_obstacle_clearance": mean_min_clear,
                "max_velocity": mean_max_vel,
                "mean_max_velocity": mean_max_vel,
                "max_acceleration": mean_max_acc,
                "mean_max_acceleration": mean_max_acc,
                "obstacle_collision_count": obs_coll_count,
                "boundary_collision_count": bound_coll_count,
                "obstacle_collision_rate": obs_coll_rate,
                "boundary_collision_rate": bound_coll_rate,
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
            "mean_path_length": round(metrics.mean_path_length, 2)
            if metrics.mean_path_length is not None
            else None,
            "std_path_length": round(metrics.std_path_length, 2)
            if metrics.std_path_length is not None
            else None,
            "mean_straight_line_distance": round(metrics.mean_straight_line_distance, 2)
            if metrics.mean_straight_line_distance is not None
            else None,
            "mean_path_efficiency": round(metrics.mean_path_efficiency, 4)
            if metrics.mean_path_efficiency is not None
            else None,
            "mean_min_obstacle_clearance": round(metrics.mean_min_obstacle_clearance, 2)
            if metrics.mean_min_obstacle_clearance is not None
            else None,
            "mean_max_velocity": round(metrics.mean_max_velocity, 2)
            if metrics.mean_max_velocity is not None
            else None,
            "mean_max_acceleration": round(metrics.mean_max_acceleration, 2)
            if metrics.mean_max_acceleration is not None
            else None,
            "obstacle_collision_rate": round(metrics.obstacle_collision_rate, 4)
            if metrics.obstacle_collision_rate is not None
            else None,
            "boundary_collision_rate": round(metrics.boundary_collision_rate, 4)
            if metrics.boundary_collision_rate is not None
            else None,
            "obstacle_collision_count": metrics.obstacle_collision_count,
            "boundary_collision_count": metrics.boundary_collision_count,
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return target

    @staticmethod
    def save_csv_report(
        metrics: EvaluationMetrics,
        output_path: str | Path,
    ) -> Path:
        """Serialize standardized evaluation metrics to CSV."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        std_metrics = StandardizedExperimentMetrics.from_rl_metrics(metrics)
        row = std_metrics.to_csv_dict()

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

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
