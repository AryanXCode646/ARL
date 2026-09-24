"""Training engine implementations for AdaptiveRL."""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import gymnasium as gym
import numpy as np
import torch

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.config import ExperimentConfig
from adaptive_rl.environments.registry import make_env
from adaptive_rl.experiments.metadata import (
    EpisodeRecord,
    ExperimentMetadata,
    save_episodes_csv,
)
from adaptive_rl.training.callbacks import (
    BaseCallback,
    CheckpointCallback,
    MetricLoggerCallback,
    SB3CallbackAdapter,
)
from adaptive_rl.training.checkpointing import CheckpointManager


@dataclass
class TrainingResult:
    """Structured summary and artifacts resulting from a training run."""

    experiment_name: str
    total_timesteps: int
    episodes_completed: int
    mean_reward: float
    final_model_path: Path
    checkpoints: List[dict[str, Any]] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    success_rate: float = 0.0
    collision_rate: float = 0.0
    metadata_path: Optional[Path] = None
    episodes_csv_path: Optional[Path] = None


class BaseTrainer(ABC):
    """Abstract interface for RL training workflows in AdaptiveRL.

    Handles the shared lifecycle: seeding, env creation, checkpoint manager,
    callbacks, algorithm construction, training, model saving, metadata
    generation, and evaluation.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        env: Optional[gym.Env] = None,
        callbacks: Optional[List[BaseCallback]] = None,
    ) -> None:
        """Common initialization for all trainers.

        Args:
            config: Validated ExperimentConfig instance.
            env: Optional pre‑instantiated Gymnasium environment.
            callbacks: Optional additional callbacks supplied by the caller.
        """
        self.config = config
        self._set_deterministic_seed(self.config.seed)

        # 1. Environment initialization
        if env is not None:
            self.env = env
        else:
            self.env = make_env(
                self.config.environment.name,
                **self.config.environment.parameters,
            )

        # 2. Checkpoint management
        checkpoint_dir = self.config.output_dir / "checkpoints" / self.config.name
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # 3. Callbacks – metric logger always present
        self.metric_logger = MetricLoggerCallback()
        self._callbacks: List[BaseCallback] = [self.metric_logger]

        if self.config.training.checkpoint_freq > 0:
            checkpoint_cb = CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                save_freq=self.config.training.checkpoint_freq,
            )
            self._callbacks.append(checkpoint_cb)

        if callbacks:
            self._callbacks.extend(callbacks)

        # 4. Algorithm construction – delegated to subclass
        self.algorithm = self._create_algorithm()

    @staticmethod
    def _set_deterministic_seed(seed: int) -> None:
        """Enforce deterministic random seeds across libraries."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def _create_algorithm(self) -> Any:
        """Instantiate the concrete RL algorithm.

        Sub‑classes must return an instance of ``PPOAlgorithm`` or ``SACAlgorithm``
        configured with the current ``config`` and ``env``.
        """
        raise NotImplementedError

    def fit(self) -> TrainingResult:
        """Execute the full training lifecycle.

        This mirrors the previous per‑algorithm implementations but is now
        centralised. The concrete algorithm is supplied by ``_create_algorithm``.
        """
        started_at = time.time()
        import adaptive_rl

        adapter = SB3CallbackAdapter(
            callbacks=self._callbacks,
            algorithm=self.algorithm,
        )

        # Run optimisation
        self.algorithm.train(
            total_timesteps=self.config.training.total_timesteps,
            callback=adapter,
        )

        # Save final model
        models_dir = self.config.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        final_model_path = models_dir / f"{self.config.name}_final.zip"
        self.algorithm.save(final_model_path)

        finished_at = time.time()
        duration = finished_at - started_at

        # Build per‑episode records for CSV export
        episode_records: List[EpisodeRecord] = []
        cumulative_ts = 0
        for i, (rew, length) in enumerate(
            zip(self.metric_logger.episode_rewards, self.metric_logger.episode_lengths)
        ):
            cumulative_ts += length
            episode_records.append(
                EpisodeRecord(
                    episode=i + 1,
                    reward=float(rew),
                    length=int(length),
                    success=False,
                    collision=False,
                    timestep=cumulative_ts,
                )
            )

        # Save metadata.json and episodes.csv
        metadata_dir = self.config.output_dir / "metadata"
        metadata = ExperimentMetadata(
            experiment_name=self.config.name,
            algorithm=self.config.algorithm.name,
            environment=self.config.environment.name,
            seed=self.config.seed,
            total_timesteps=self.config.training.total_timesteps,
            actual_timesteps=self.algorithm.num_timesteps,
            episodes_completed=self.metric_logger.total_episodes,
            mean_reward=self.metric_logger.mean_reward,
            success_rate=self.metric_logger.success_rate,
            collision_rate=self.metric_logger.collision_rate,
            final_model_path=str(final_model_path),
            checkpoint_paths=[cp["path"] for cp in self.checkpoint_manager.list_checkpoints()],
            config_snapshot=self.config.model_dump(mode="python"),
            adaptive_rl_version=adaptive_rl.__version__,
            finished_at=__import__("datetime").datetime.fromtimestamp(
                finished_at, tz=__import__("datetime").timezone.utc
            ).isoformat(),
            duration_seconds=round(duration, 3),
        )
        metadata_path = metadata.save(metadata_dir, name=self.config.name)
        episodes_csv_path = save_episodes_csv(
            records=episode_records, output_dir=metadata_dir, name=self.config.name
        )

        result = TrainingResult(
            experiment_name=self.config.name,
            total_timesteps=self.config.training.total_timesteps,
            episodes_completed=self.metric_logger.total_episodes,
            mean_reward=self.metric_logger.mean_reward,
            final_model_path=final_model_path,
            checkpoints=self.checkpoint_manager.list_checkpoints(),
            episode_rewards=list(self.metric_logger.episode_rewards),
            episode_lengths=list(self.metric_logger.episode_lengths),
            success_rate=self.metric_logger.success_rate,
            collision_rate=self.metric_logger.collision_rate,
            metadata_path=metadata_path,
            episodes_csv_path=episodes_csv_path,
        )
        return result

    def evaluate(
        self,
        episodes: int = 10,
        deterministic: bool = True,
    ) -> tuple[float, float]:
        """Run evaluation episodes on the current policy.

        The loop is identical for PPO and SAC, therefore resides in the base
        class.
        """
        rewards: List[float] = []
        for ep in range(episodes):
            obs, _ = self.env.reset(seed=self.config.seed + ep if self.config.seed else None)
            ep_reward = 0.0
            done = False
            while not done:
                action, _ = self.algorithm.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += float(reward)
                done = terminated or truncated
            rewards.append(ep_reward)
        return float(np.mean(rewards)), float(np.std(rewards))

    def close(self) -> None:
        """Clean up resources such as the environment."""
        if hasattr(self, "env"):
            try:
                self.env.close()
            except Exception:
                pass


class PPOTrainer(BaseTrainer):
    """Concrete PPO trainer delegating algorithm creation to the base class."""

    def _create_algorithm(self) -> PPOAlgorithm:
        algo_params = dict(self.config.algorithm.parameters)
        return PPOAlgorithm(
            env=self.env,
            learning_rate=self.config.algorithm.learning_rate,
            gamma=self.config.algorithm.gamma,
            batch_size=self.config.algorithm.batch_size,
            seed=self.config.seed,
            **algo_params,
        )


class SACTrainer(BaseTrainer):
    """Concrete SAC trainer delegating algorithm creation to the base class."""

    def _create_algorithm(self) -> SACAlgorithm:
        algo_params = dict(self.config.algorithm.parameters)
        return SACAlgorithm(
            env=self.env,
            learning_rate=self.config.algorithm.learning_rate,
            gamma=self.config.algorithm.gamma,
            batch_size=self.config.algorithm.batch_size,
            seed=self.config.seed,
            **algo_params,
        )


def get_trainer(
    config: ExperimentConfig,
    env: Optional[gym.Env] = None,
    callbacks: Optional[List[BaseCallback]] = None,
) -> BaseTrainer:
    """Factory returning the appropriate trainer based on configuration.

    Supports curriculum training via ``CurriculumTrainer`` when enabled.
    """
    if config.curriculum is not None and config.curriculum.enabled:
        from adaptive_rl.curriculum.trainer import CurriculumTrainer

        return CurriculumTrainer(config=config, env=env, callbacks=callbacks)

    algo_name = config.algorithm.name.lower()
    if algo_name == "ppo":
        return PPOTrainer(config=config, env=env, callbacks=callbacks)
    elif algo_name == "sac":
        return SACTrainer(config=config, env=env, callbacks=callbacks)
    else:
        raise ValueError(
            f"Unsupported algorithm '{config.algorithm.name}'. Supported algorithms: 'ppo', 'sac'"
        )
