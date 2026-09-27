"""Evaluation engine implementations for AdaptiveRL."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import gymnasium as gym
import numpy as np

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.environments.registry import list_environments, make_env
from adaptive_rl.evaluation.metrics import EvaluationMetrics
from adaptive_rl.evaluation.scenarios import EvaluationScenario
from adaptive_rl.evaluation.seeding import derive_evaluation_seed
from adaptive_rl.metrics import (
    DefaultOutcomePolicy,
    EpisodeMetrics,
    EpisodeMetricsAccumulator,
    OutcomePolicy,
    TrafficOutcomePolicy,
    compute_rate,
)

__all__ = [
    "BaseEvaluator",
    "Evaluator",
    "EpisodeEvaluationRecord",
    "EPISODE_RECORD_FIELDS",
]

EPISODE_RECORD_FIELDS: tuple[str, ...] = (
    "episode_index",
    "seed",
    "environment",
    "scenario",
    "return",
    "episode_length",
    "success",
    "collision",
)


@dataclass(frozen=True)
class EpisodeEvaluationRecord:
    """Transport record for a single evaluation episode.

    Fields are deliberately minimal and mirror the canonical EpisodeMetrics
    together with contextual identifiers required for downstream analysis.
    """

    episode_index: int
    seed: Optional[int]
    environment: str
    scenario: Optional[str]
    return_value: float
    episode_length: int
    success: Optional[bool]
    collision: Optional[bool]

    @classmethod
    def from_episode_metrics(
        cls,
        metrics: EpisodeMetrics,
        *,
        episode_index: int,
        seed: Optional[int],
        environment: str,
        scenario: Optional[str] = None,
    ) -> EpisodeEvaluationRecord:
        """Create an export record directly from canonical EpisodeMetrics."""
        return cls(
            episode_index=episode_index,
            seed=seed,
            environment=environment,
            scenario=scenario,
            return_value=metrics.reward,
            episode_length=metrics.length,
            success=metrics.success,
            collision=metrics.collision,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict suitable for JSON/CSV output."""
        return {
            "episode_index": self.episode_index,
            "seed": self.seed,
            "environment": self.environment,
            "scenario": self.scenario,
            "return": self.return_value,
            "episode_length": self.episode_length,
            "success": self.success,
            "collision": self.collision,
        }


class BaseEvaluator(ABC):
    """Abstract interface for agent/environment evaluation routines."""

    @abstractmethod
    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        base_seed: Optional[int] = None,
    ) -> EvaluationMetrics:
        """Run evaluation benchmark over the specified number of episodes."""
        pass


class Evaluator(BaseEvaluator):
    """Standardized multi-episode evaluation benchmark engine."""

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: Optional[gym.Env] = None,
        env_name: Optional[str] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        outcome_policy: Optional[OutcomePolicy] = None,
        scenario: Optional[str] = None,
    ) -> None:
        """Initialize evaluator with algorithm and evaluation environment.

        Args:
            algorithm: Trained agent algorithm instance implementing BaseAlgorithm.
            env: Pre-instantiated Gymnasium environment (optional).
            env_name: Registered environment name to instantiate via factory (optional).
            env_kwargs: Additional parameters forwarded to make_env when env_name is used.
            outcome_policy: Optional explicit domain OutcomePolicy instance.
            scenario: Optional scenario name context.
        """
        self.algorithm = algorithm
        self.env_kwargs = dict(env_kwargs or {})
        self.scenario = scenario

        if env_name is not None:
            self.env_name = env_name
            self.env = env if env is not None else make_env(env_name, **self.env_kwargs)
        elif env is not None:
            self.env = env
            spec = getattr(env, "spec", None)
            if spec is not None and getattr(spec, "id", None):
                self.env_name = str(spec.id)
            else:
                registered = list_environments()
                cls_name = type(env).__name__
                simplified = cls_name.lower().replace("env", "")
                if simplified in registered:
                    self.env_name = simplified
                elif cls_name in registered:
                    self.env_name = cls_name
                else:
                    self.env_name = cls_name
        else:
            raise ValueError("Evaluator requires either 'env' or 'env_name'.")

        if outcome_policy is not None:
            self.outcome_policy: OutcomePolicy = outcome_policy
            self._explicit_policy = True
        elif "traffic" in self.env_name.lower():
            self.outcome_policy = TrafficOutcomePolicy()
            self._explicit_policy = False
        else:
            self.outcome_policy = DefaultOutcomePolicy()
            self._explicit_policy = False

        self.last_episode_metrics: List[EpisodeMetrics] = []
        # Records for optional episode‑level export. Cleared at the start of each evaluate call.
        self.last_episode_records: List[EpisodeEvaluationRecord] = []

    def _make_episode_accumulator(self) -> EpisodeMetricsAccumulator:
        """Bind a policy for one episode without treating inferred defaults as explicit.

        Explicit caller-supplied policies are locked on the accumulator. Inferred
        TrafficOutcomePolicy (from env name) is selected via is_traffic so it is
        immutable for the episode. Inferred DefaultOutcomePolicy leaves fallback
        enabled for the first record_step only.
        """
        if self._explicit_policy:
            return EpisodeMetricsAccumulator(outcome_policy=self.outcome_policy)
        if isinstance(self.outcome_policy, TrafficOutcomePolicy):
            return EpisodeMetricsAccumulator(is_traffic=True)
        return EpisodeMetricsAccumulator()

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        base_seed: Optional[int] = None,
        scenario: Optional[str] = None,
    ) -> EvaluationMetrics:
        """Execute evaluation rollouts and compute aggregated metrics.

        Args:
            num_episodes: Number of evaluation episodes to execute.
            deterministic: Whether to use deterministic action selection.
            base_seed: Base seed for reproducible evaluation episode initializations.
            scenario: Optional scenario name overriding the evaluator-level scenario.

        Returns:
            EvaluationMetrics: Standardized aggregated performance metrics.
                Rate denominator semantics: success_rate, collision_rate, and overflow_rate
                are calculated among episodes where the metric is defined (non-None). If a
                metric is unavailable across all episodes (all None), the rate evaluates to None.
        """
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")
        # Ensure per‑evaluation records are cleared to avoid leakage when reusing the same Evaluator instance
        self.last_episode_records = []
        active_scenario = scenario if scenario is not None else self.scenario

        episode_metrics: List[EpisodeMetrics] = []

        is_traffic_env = isinstance(self.outcome_policy, TrafficOutcomePolicy)

        # Traffic telemetry accumulators
        all_step_queues: List[int] = []
        all_step_mean_waits: List[float] = []
        ep_max_waits: List[int] = []
        ep_final_departures: List[int] = []
        ep_final_delays: List[float] = []

        for ep in range(num_episodes):
            seed = derive_evaluation_seed(base_seed, ep) if base_seed is not None else None
            obs, info = self.env.reset(seed=seed)
            acc = self._make_episode_accumulator()
            done = False

            ep_step_max_waits: List[int] = []
            last_step_info: Dict[str, Any] = dict(info or {})

            # Reset policy if policy provides episode-level reset (e.g. PlannerPolicy)
            if hasattr(self.algorithm, "reset_policy"):
                self.algorithm.reset_policy()

            while not done:
                action, _ = self.algorithm.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                acc.record_step(
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    info=step_info,
                )
                last_step_info = step_info

                # Track domain-specific traffic telemetry
                if "total_queue" in step_info or "queue_lengths" in step_info:
                    is_traffic_env = True
                    if not self._explicit_policy and not isinstance(
                        self.outcome_policy, TrafficOutcomePolicy
                    ):
                        # Bind subsequent episodes only; this episode's policy is already locked.
                        self.outcome_policy = TrafficOutcomePolicy()
                    if "total_queue" in step_info:
                        all_step_queues.append(int(step_info["total_queue"]))
                    if "max_wait" in step_info:
                        ep_step_max_waits.append(int(step_info["max_wait"]))
                    if "mean_wait" in step_info:
                        all_step_mean_waits.append(float(step_info["mean_wait"]))

                done = terminated or truncated

            m = acc.finish()
            episode_metrics.append(m)

            # Build episode export record from canonical EpisodeMetrics
            record = EpisodeEvaluationRecord.from_episode_metrics(
                m,
                episode_index=ep,
                seed=seed,
                environment=self.env_name,
                scenario=active_scenario,
            )
            self.last_episode_records.append(record)

            if is_traffic_env:
                ep_max_waits.append(max(ep_step_max_waits) if ep_step_max_waits else 0)
                ep_final_departures.append(int(last_step_info.get("cumulative_departures", 0)))
                ep_final_delays.append(float(last_step_info.get("cumulative_delay", 0.0)))

        self.last_episode_metrics = episode_metrics

        rewards = [m.reward for m in episode_metrics]
        lengths = [m.length for m in episode_metrics]

        mean_rew = float(np.mean(rewards))
        std_rew = float(np.std(rewards))
        min_rew = float(np.min(rewards))
        max_rew = float(np.max(rewards))
        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))

        # Rate aggregation denominator semantics:
        # Rates (success_rate, collision_rate, overflow_rate) are calculated among defined episodes
        # (non-None). If a metric is unavailable across all episodes (all None), the rate evaluates to None.
        has_overflow_info = any("had_overflow" in m.additional_metrics for m in episode_metrics)

        success_rate = compute_rate([m.success for m in episode_metrics])
        collision_rate = compute_rate([m.collision for m in episode_metrics])
        overflow_rate = (
            compute_rate(
                [
                    m.additional_metrics["had_overflow"]
                    for m in episode_metrics
                    if "had_overflow" in m.additional_metrics
                ]
            )
            if has_overflow_info
            else None
        )
        truncation_rate: Optional[float] = float(
            sum(1 for m in episode_metrics if m.truncated) / num_episodes
        )

        additional: Dict[str, Any] = {
            "all_rewards": rewards,
            "all_lengths": lengths,
            "deterministic": deterministic,
            "base_seed": base_seed,
            "truncation_rate": truncation_rate,
        }

        # Preserve centralized traffic telemetry if environment is traffic
        if is_traffic_env:
            # Documented aggregation rules:
            # - mean_queue_length: Mean of per-timestep total queue lengths averaged across all evaluation timesteps.
            # - mean_max_wait_time: Mean of per-episode maximum vehicle waiting times across all evaluation episodes.
            # - mean_wait_time: Mean of per-timestep mean vehicle waiting times averaged across all evaluation timesteps.
            # - mean_total_departures: Mean cumulative vehicle departures per episode across all evaluation episodes.
            # - total_departures: Total vehicle departures summed across all evaluation episodes.
            # - cumulative_delay: Mean cumulative vehicle delay per episode across all evaluation episodes.
            # - overflow_rate: Fraction of evaluation episodes in which at least one queue overflow occurred.
            additional["mean_queue_length"] = (
                float(np.mean(all_step_queues)) if all_step_queues else 0.0
            )
            additional["mean_max_wait_time"] = float(np.mean(ep_max_waits)) if ep_max_waits else 0.0
            additional["mean_wait_time"] = (
                float(np.mean(all_step_mean_waits)) if all_step_mean_waits else 0.0
            )
            additional["mean_total_departures"] = (
                float(np.mean(ep_final_departures)) if ep_final_departures else 0.0
            )
            additional["total_departures"] = (
                int(np.sum(ep_final_departures)) if ep_final_departures else 0
            )
            additional["cumulative_delay"] = (
                float(np.mean(ep_final_delays)) if ep_final_delays else 0.0
            )
            additional["overflow_rate"] = overflow_rate

        return EvaluationMetrics(
            episodes=num_episodes,
            mean_reward=mean_rew,
            std_reward=std_rew,
            min_reward=min_rew,
            max_reward=max_rew,
            success_rate=success_rate,
            collision_rate=collision_rate,
            overflow_rate=overflow_rate,
            truncation_rate=truncation_rate,
            mean_episode_length=mean_len,
            std_episode_length=std_len,
            additional_metrics=additional,
        )

    def evaluate_scenarios(
        self,
        scenarios: List[EvaluationScenario],
        deterministic: bool = True,
        num_episodes_per_scenario: int = 1,
    ) -> Dict[str, EvaluationMetrics]:
        """Benchmark the agent across a curated collection of evaluation scenarios.

        Args:
            scenarios: List of EvaluationScenario specifications.
            deterministic: Whether to evaluate deterministically.
            num_episodes_per_scenario: Number of evaluation episodes per scenario (defaults to 1).

        Returns:
            Dict[str, EvaluationMetrics]: Mapping from scenario name to evaluation metrics.
        """
        # Ensure parent records are cleared before evaluating scenarios
        self.last_episode_records = []
        results: Dict[str, EvaluationMetrics] = {}

        for sc in scenarios:
            is_new_env = False
            if not sc.environment_overrides and self.env is not None:
                sc_env = self.env
            else:
                scenario_kwargs = dict(self.env_kwargs)
                scenario_kwargs.update(sc.environment_overrides)
                sc_env = make_env(self.env_name, **scenario_kwargs)
                is_new_env = True

            try:
                sc_evaluator = Evaluator(
                    algorithm=self.algorithm,
                    env=sc_env,
                    outcome_policy=self.outcome_policy if self._explicit_policy else None,
                    scenario=sc.name,
                )
                metrics = sc_evaluator.evaluate(
                    num_episodes=num_episodes_per_scenario,
                    deterministic=deterministic,
                    base_seed=sc.seed,
                    scenario=sc.name,
                )
                results[sc.name] = metrics
                # Aggregate per‑scenario episode records into parent evaluator
                self.last_episode_records.extend(sc_evaluator.last_episode_records)
            finally:
                if is_new_env:
                    sc_env.close()

        return results

    @staticmethod
    def save_report(
        metrics: EvaluationMetrics | Dict[str, EvaluationMetrics],
        output_path: str | Path,
    ) -> Path:
        """Serialize evaluation metrics to a formatted JSON report.

        Args:
            metrics: EvaluationMetrics instance or dictionary of named metrics.
            output_path: Destination filepath.

        Returns:
            Path: Written report path.
        """
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(metrics, EvaluationMetrics):
            data = metrics.model_dump()
        else:
            data = {name: m.model_dump() for name, m in metrics.items()}

        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return target

    @staticmethod
    def save_episode_report(
        records: Sequence[Union[EpisodeEvaluationRecord, Dict[str, Any]]],
        output_path: Union[str, Path],
    ) -> Path:
        """Serialize per‑episode evaluation records to JSON or CSV.

        The output format is inferred from the file extension. Supported extensions:
        - ``.json``: writes a JSON array of record dictionaries.
        - ``.csv``: writes a CSV with columns
          ``episode_index,seed,environment,scenario,return,episode_length,success,collision``.

        Args:
            records: Sequence of EpisodeEvaluationRecord instances or dicts containing
                required episode fields.
            output_path: Destination filepath (.json or .csv).

        Returns:
            Path: Written report path.

        Raises:
            ValueError: If the file extension is unsupported or a record is malformed.
            TypeError: If a record is not an EpisodeEvaluationRecord or dict.
        """
        target = Path(output_path)
        ext = target.suffix.lower()
        if ext not in (".json", ".csv"):
            raise ValueError(
                f"Unsupported file extension for episode report: '{ext}'. Supported formats: .json, .csv"
            )

        target.parent.mkdir(parents=True, exist_ok=True)

        data: List[Dict[str, Any]] = []
        for idx, rec in enumerate(records):
            if isinstance(rec, EpisodeEvaluationRecord):
                data.append(rec.to_dict())
            elif isinstance(rec, dict):
                missing = [field for field in EPISODE_RECORD_FIELDS if field not in rec]
                if missing:
                    raise ValueError(
                        f"Malformed record at index {idx}: missing required field(s): {', '.join(missing)}"
                    )
                data.append({field: rec[field] for field in EPISODE_RECORD_FIELDS})
            else:
                raise TypeError(
                    f"Record at index {idx} must be EpisodeEvaluationRecord or dict, got {type(rec).__name__}"
                )

        if ext == ".json":
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            with open(target, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(EPISODE_RECORD_FIELDS))
                writer.writeheader()
                for row in data:
                    writer.writerow(row)

        return target

    def close(self) -> None:
        """Close the evaluation environment."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
