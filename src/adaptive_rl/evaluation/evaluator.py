"""Evaluation engine implementations for AdaptiveRL."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.metrics import EvaluationMetrics
from adaptive_rl.evaluation.scenarios import EvaluationScenario
from adaptive_rl.metrics import (
    DefaultOutcomePolicy,
    EpisodeMetrics,
    EpisodeMetricsAccumulator,
    OutcomePolicy,
    TrafficOutcomePolicy,
    compute_rate,
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
    ) -> None:
        """Initialize evaluator with algorithm and evaluation environment.

        Args:
            algorithm: Trained agent algorithm instance implementing BaseAlgorithm.
            env: Pre-instantiated Gymnasium environment (optional).
            env_name: Registered environment name to instantiate via factory (optional).
            env_kwargs: Additional parameters forwarded to make_env when env_name is used.
            outcome_policy: Optional explicit domain OutcomePolicy instance.
        """
        self.algorithm = algorithm
        self.env_kwargs = dict(env_kwargs or {})

        if env_name is not None:
            self.env_name = env_name
            self.env = env if env is not None else make_env(env_name, **self.env_kwargs)
        elif env is not None:
            self.env = env
            spec = getattr(env, "spec", None)
            if spec is not None and getattr(spec, "id", None):
                self.env_name = str(spec.id)
            else:
                from adaptive_rl.environments.registry import list_environments

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
        self.last_episode_records: List["EpisodeEvaluationRecord"] = []
        # Holds the name of the scenario being evaluated (if any).
        self._current_scenario_name: Optional[str] = None

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
    ) -> EvaluationMetrics:
        """Execute evaluation rollouts and compute aggregated metrics.

        Args:
            num_episodes: Number of evaluation episodes to execute.
            deterministic: Whether to use deterministic action selection.
            base_seed: Base seed for reproducible evaluation episode initializations.

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

        episode_metrics: List[EpisodeMetrics] = []

        is_traffic_env = isinstance(self.outcome_policy, TrafficOutcomePolicy)

        # Traffic telemetry accumulators
        all_step_queues: List[int] = []
        all_step_mean_waits: List[float] = []
        ep_max_waits: List[int] = []
        ep_final_departures: List[int] = []
        ep_final_delays: List[float] = []

        from adaptive_rl.evaluation.seeding import derive_evaluation_seed

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

            # Build episode export record
            record = EpisodeEvaluationRecord(
                episode_index=ep,
                seed=seed,
                environment=self.env_name,
                scenario=self._current_scenario_name,
                return_value=m.reward,
                episode_length=m.length,
                success=m.success,
                collision=m.collision,
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
    ) -> Dict[str, EvaluationMetrics]:
        """Benchmark the agent across a curated collection of evaluation scenarios.

        Args:
            scenarios: List of EvaluationScenario specifications.
            deterministic: Whether to evaluate deterministically.

        Returns:
            Dict[str, EvaluationMetrics]: Mapping from scenario name to evaluation metrics.
        """
        # Ensure parent records are cleared before evaluating scenarios
        self.last_episode_records = []
        results: Dict[str, EvaluationMetrics] = {}

        for sc in scenarios:
            scenario_kwargs = dict(self.env_kwargs)
            scenario_kwargs.update(sc.environment_overrides)

            # Create environment for this specific scenario
            try:
                sc_env = make_env(self.env_name, **scenario_kwargs)
            except Exception:
                # Fallback to the existing environment when the name is not registered
                sc_env = self.env
            sc_evaluator = Evaluator(algorithm=self.algorithm, env=sc_env)
            # Propagate scenario identifier for per‑episode export
            sc_evaluator._current_scenario_name = sc.name
            metrics = sc_evaluator.evaluate(
                num_episodes=1,
                deterministic=deterministic,
                base_seed=sc.seed,
            )
            results[sc.name] = metrics
            # Aggregate per‑scenario episode records into parent evaluator
            self.last_episode_records.extend(sc_evaluator.last_episode_records)
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

        # Return after writing aggregate report
        # New: Serialize per‑episode records to JSON or CSV.
        return target

    @staticmethod
    def save_episode_report(
        records: List[EpisodeEvaluationRecord] | List[Dict[str, Any]],
        output_path: str | Path,
    ) -> Path:
        """Serialize per‑episode evaluation records to JSON or CSV.

        The output format is inferred from the file extension. Supported extensions:
        - ``.json``: writes a JSON array of record dictionaries.
        - ``.csv``: writes a CSV with columns
          ``episode_index,seed,environment,scenario,return,episode_length,success,collision``.
        """
        import csv
        from pathlib import Path as _Path

        target = _Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Normalise records to list of dicts
        data = [r.to_dict() if isinstance(r, EpisodeEvaluationRecord) else r for r in records]

        ext = target.suffix.lower()
        if ext == ".json":
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        elif ext == ".csv":
            fieldnames = [
                "episode_index",
                "seed",
                "environment",
                "scenario",
                "return",
                "episode_length",
                "success",
                "collision",
            ]
            with open(target, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    writer.writerow({k: row.get(k) for k in fieldnames})
        else:
            raise ValueError(f"Unsupported file extension for episode report: {ext}")

        return target

    def close(self) -> None:
        """Close the evaluation environment."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
