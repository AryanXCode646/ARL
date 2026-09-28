"""Benchmark orchestration for PPO learning curves across training budgets."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.config import BenchmarkConfig, ExperimentConfig
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.training.trainer import PPOTrainer


def validate_budgets(
    raw_budgets: Sequence[int] | str | None, *, allow_empty: bool = False
) -> list[int]:
    """Validate and normalize benchmark budgets.

    Rules:
      - reject empty, zero, negative, non-integer, malformed comma-separated values
      - reject duplicates after normalization
      - sort ascending for reproducibility
    """
    if raw_budgets is None:
        if allow_empty:
            return []
        raise ValueError("Benchmark budgets cannot be empty.")

    if isinstance(raw_budgets, str):
        values = [part.strip() for part in raw_budgets.split(",")]
        if not values or any(part == "" for part in values):
            raise ValueError("Malformed budget list: expected comma-separated integers.")
        normalized: list[int] = []
        for token in values:
            try:
                normalized.append(int(token))
            except ValueError as exc:  # pragma: no cover - explicit validation path
                raise ValueError(f"Malformed budget value: {token!r}") from exc
    else:
        normalized = list(raw_budgets)

    if not normalized:
        if allow_empty:
            return []
        raise ValueError("Benchmark budgets cannot be empty.")

    cleaned: list[int] = []
    seen: set[int] = set()
    for value in normalized:
        if isinstance(value, bool):
            raise ValueError(f"Budget values must be integers, got boolean {value!r}.")
        if not isinstance(value, int):
            raise ValueError(
                f"Budget values must be integers, got {type(value).__name__}: {value!r}"
            )
        if value <= 0:
            raise ValueError(f"Budget values must be positive, got {value!r}.")
        if value in seen:
            raise ValueError(f"Duplicate budget value detected: {value}.")
        seen.add(value)
        cleaned.append(value)

    cleaned.sort()
    return cleaned


@dataclass(frozen=True)
class LearningCurvePoint:
    """Performance record for a single training-budget evaluation point."""

    budget_timesteps: int
    trained_timesteps: int
    success_rate: float | None
    collision_rate: float | None
    timeout_rate: float | None
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    training_time_seconds: float
    model_path: str
    training_seed: int
    evaluation_seeds: list[int]
    evaluation_episodes: int
    deterministic: bool
    algorithm: str
    environment: str


@dataclass
class LearningCurveBenchmarkResult:
    """Top-level container for the ordered learning-curve benchmark output."""

    benchmark_name: str
    algorithm: str
    environment: str
    budgets: list[int]
    training_seed: int
    evaluation_seeds: list[int]
    evaluation_episodes: int
    deterministic: bool
    points: list[LearningCurvePoint] = field(default_factory=list)
    plot_data: dict[str, list[float | int | None]] = field(default_factory=dict)
    output_dir: Path | None = None
    json_path: Path | None = None
    csv_path: Path | None = None
    plot_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the benchmark result to a JSON-friendly dictionary."""
        payload: dict[str, Any] = {
            "benchmark": {
                "name": self.benchmark_name,
                "algorithm": self.algorithm,
                "environment": self.environment,
                "training_seed": self.training_seed,
                "evaluation_seeds": self.evaluation_seeds,
                "evaluation_episodes": self.evaluation_episodes,
                "deterministic": self.deterministic,
                "budgets": self.budgets,
            },
            "results": [
                {
                    "budget_timesteps": point.budget_timesteps,
                    "trained_timesteps": point.trained_timesteps,
                    "success_rate": point.success_rate,
                    "collision_rate": point.collision_rate,
                    "timeout_rate": point.timeout_rate,
                    "mean_reward": point.mean_reward,
                    "std_reward": point.std_reward,
                    "mean_episode_length": point.mean_episode_length,
                    "training_time_seconds": point.training_time_seconds,
                    "model_path": point.model_path,
                    "training_seed": point.training_seed,
                    "evaluation_seeds": point.evaluation_seeds,
                    "evaluation_episodes": point.evaluation_episodes,
                    "deterministic": point.deterministic,
                    "algorithm": point.algorithm,
                    "environment": point.environment,
                }
                for point in self.points
            ],
            "plot_data": self.plot_data,
        }
        return payload


def _resolve_benchmark_config(
    base_config: ExperimentConfig, overrides: dict[str, Any] | None
) -> BenchmarkConfig:
    """Merge benchmark settings into a config object without mutating the original."""
    if base_config.benchmark is not None:
        benchmark_cfg = BenchmarkConfig.model_validate(
            base_config.benchmark.model_dump(mode="python")
        )
    else:
        benchmark_cfg = BenchmarkConfig()

    if overrides:
        benchmark_cfg = benchmark_cfg.model_copy(update=overrides)
    return benchmark_cfg


def _budget_dir(base_output_dir: Path, budget: int) -> Path:
    return base_output_dir / "learning_curve" / f"budget_{budget}"


def _evaluate_model(
    model_path: Path,
    *,
    env_name: str,
    env_kwargs: dict[str, Any],
    evaluation_seeds: Sequence[int],
    evaluation_episodes: int,
    deterministic: bool,
) -> tuple[float | None, float | None, float | None, float, float, float]:
    env = make_env(env_name, **env_kwargs)
    try:
        algo = PPOAlgorithm.from_pretrained(model_path, env=env)
        evaluator = Evaluator(algorithm=algo, env=env)

        all_rewards: list[float] = []
        all_lengths: list[int] = []
        successes: list[bool] = []
        collisions: list[bool] = []
        timeout_count = 0
        total_episodes = 0

        for seed in evaluation_seeds:
            metrics = evaluator.evaluate(
                num_episodes=evaluation_episodes,
                deterministic=deterministic,
                base_seed=seed,
            )
            all_rewards.extend(metrics.additional_metrics.get("all_rewards", []))
            all_lengths.extend(metrics.additional_metrics.get("all_lengths", []))
            records = evaluator.last_episode_records
            total_episodes += len(records)
            successes.extend(record.success for record in records if record.success is not None)
            collisions.extend(
                record.collision for record in records if record.collision is not None
            )
            timeout_count += sum(record.truncated for record in records)

        mean_reward = float(sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
        std_reward = (
            float(
                (
                    sum((reward - mean_reward) ** 2 for reward in all_rewards)
                    / max(1, len(all_rewards) - 1)
                )
                ** 0.5
            )
            if len(all_rewards) > 1
            else 0.0
        )
        mean_episode_length = float(sum(all_lengths) / len(all_lengths)) if all_lengths else 0.0

        success_rate = float(sum(successes) / len(successes)) if successes else None
        collision_rate = float(sum(collisions) / len(collisions)) if collisions else None
        timeout_rate = float(timeout_count / total_episodes) if total_episodes else None

        return (
            success_rate,
            collision_rate,
            timeout_rate,
            mean_reward,
            std_reward,
            mean_episode_length,
        )
    finally:
        env.close()


def _run_single_budget(
    config: ExperimentConfig,
    budget: int,
    *,
    training_seed: int,
    evaluation_seeds: Sequence[int],
    evaluation_episodes: int,
    deterministic: bool,
    output_base_dir: Path,
) -> LearningCurvePoint:
    """Run a single budget as a fresh training process from the same base configuration."""
    benchmark_dir = _budget_dir(output_base_dir, budget)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    config_copy = config.model_copy(deep=True)
    config_copy.name = f"ppo_budget_{budget}"
    config_copy.seed = training_seed
    config_copy.algorithm.parameters.pop("seed", None)
    if config_copy.training is None:
        raise ValueError("A training section is required to run the learning-curve benchmark.")
    config_copy.training.total_timesteps = budget
    config_copy.output_dir = benchmark_dir
    config_copy.log_dir = benchmark_dir / "logs"

    env = make_env(config_copy.environment.name, **config_copy.environment.parameters)
    trainer: PPOTrainer | None = None
    start = time.perf_counter()
    try:
        trainer = PPOTrainer(config=config_copy, env=env)
        result = trainer.fit()
        training_time_seconds = time.perf_counter() - start
    finally:
        if trainer is not None:
            trainer.close()
        else:
            env.close()

    model_path = result.final_model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Training budget {budget} did not create model artifact: {model_path}"
        )
    if trainer is None:
        raise RuntimeError("Training did not create a PPO trainer.")
    trained_timesteps = trainer.algorithm.num_timesteps
    if not math.isfinite(training_time_seconds) or training_time_seconds < 0:
        raise RuntimeError(
            f"Invalid training duration for budget {budget}: {training_time_seconds}"
        )

    success_rate, collision_rate, timeout_rate, mean_reward, std_reward, mean_episode_length = (
        _evaluate_model(
            model_path,
            env_name=config_copy.environment.name,
            env_kwargs=config_copy.environment.parameters,
            evaluation_seeds=evaluation_seeds,
            evaluation_episodes=evaluation_episodes,
            deterministic=deterministic,
        )
    )

    return LearningCurvePoint(
        budget_timesteps=budget,
        trained_timesteps=trained_timesteps,
        success_rate=success_rate,
        collision_rate=collision_rate,
        timeout_rate=timeout_rate,
        mean_reward=mean_reward,
        std_reward=std_reward,
        mean_episode_length=mean_episode_length,
        training_time_seconds=float(training_time_seconds),
        model_path=str(model_path),
        training_seed=training_seed,
        evaluation_seeds=list(evaluation_seeds),
        evaluation_episodes=evaluation_episodes,
        deterministic=deterministic,
        algorithm=config_copy.algorithm.name,
        environment=config_copy.environment.name,
    )


def run_learning_curve_benchmark(
    config: ExperimentConfig,
    budgets: Sequence[int] | str | None = None,
    *,
    training_seed: int | None = None,
    evaluation_seeds: Sequence[int] | None = None,
    evaluation_episodes: int | None = None,
    deterministic: bool | None = None,
    output_dir: str | Path | None = None,
    plot: bool = False,
) -> LearningCurveBenchmarkResult:
    """Train a model for each budget, evaluate it, and serialize benchmark artifacts."""
    if config.algorithm.name.lower() != "ppo":
        raise ValueError("The learning-curve benchmark currently supports only PPO.")
    if config.training is None:
        raise ValueError("A training section is required to run the learning-curve benchmark.")

    if budgets is None:
        benchmark_cfg = _resolve_benchmark_config(config, None)
        normalized = validate_budgets(benchmark_cfg.budgets)
    else:
        normalized = validate_budgets(budgets)

    if training_seed is not None:
        final_training_seed = training_seed
    elif config.benchmark is not None:
        final_training_seed = config.benchmark.training_seed
    else:
        final_training_seed = config.seed

    if evaluation_seeds is None:
        if config.benchmark is not None:
            final_eval_seeds = list(config.benchmark.evaluation_seeds)
        else:
            final_eval_seeds = [config.seed + i for i in range(config.evaluation.eval_episodes)]
    else:
        final_eval_seeds = list(evaluation_seeds)

    if isinstance(final_training_seed, bool) or not isinstance(final_training_seed, int):
        raise ValueError("Training seed must be an integer.")
    if final_training_seed < 0:
        raise ValueError("Training seed must be non-negative.")
    if not final_eval_seeds:
        raise ValueError("Evaluation seeds must not be empty.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in final_eval_seeds):
        raise ValueError("Evaluation seeds must be integers.")
    if any(seed < 0 for seed in final_eval_seeds):
        raise ValueError("Evaluation seeds must be non-negative.")
    if len(set(final_eval_seeds)) != len(final_eval_seeds):
        raise ValueError("Evaluation seeds must not contain duplicates.")

    if evaluation_episodes is None:
        if config.benchmark is not None:
            final_eval_episodes = config.benchmark.evaluation_episodes
        else:
            final_eval_episodes = config.evaluation.eval_episodes
    else:
        final_eval_episodes = evaluation_episodes
    if isinstance(final_eval_episodes, bool) or not isinstance(final_eval_episodes, int):
        raise ValueError("Evaluation episodes per seed must be an integer.")
    if final_eval_episodes <= 0:
        raise ValueError("Evaluation episodes per seed must be positive.")

    if deterministic is None:
        if config.benchmark is not None:
            final_deterministic = config.benchmark.deterministic
        else:
            final_deterministic = config.evaluation.deterministic
    else:
        if not isinstance(deterministic, bool):
            raise ValueError("Deterministic evaluation setting must be a boolean.")
        final_deterministic = deterministic

    if output_dir is None:
        target_dir = Path(config.output_dir) / "benchmarks"
    else:
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "learning_curve_budget.json"
    csv_path = target_dir / "learning_curve_budget.csv"
    plot_path = target_dir / "learning_curve_budget.png" if plot else None

    bench_points: list[LearningCurvePoint] = []
    for budget in normalized:
        point = _run_single_budget(
            config,
            budget,
            training_seed=final_training_seed,
            evaluation_seeds=final_eval_seeds,
            evaluation_episodes=final_eval_episodes,
            deterministic=final_deterministic,
            output_base_dir=target_dir,
        )
        bench_points.append(point)

    result = LearningCurveBenchmarkResult(
        benchmark_name="ppo_learning_curve",
        algorithm=config.algorithm.name,
        environment=config.environment.name,
        budgets=normalized,
        training_seed=final_training_seed,
        evaluation_seeds=list(final_eval_seeds),
        evaluation_episodes=final_eval_episodes,
        deterministic=final_deterministic,
        points=bench_points,
        plot_data={
            "budgets": [int(point.budget_timesteps) for point in bench_points],
            "success_rate": [point.success_rate for point in bench_points],
            "mean_reward": [point.mean_reward for point in bench_points],
        },
        output_dir=target_dir,
        json_path=json_path,
        csv_path=csv_path,
        plot_path=plot_path,
    )

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, allow_nan=False)

    fieldnames = [
        "budget_timesteps",
        "trained_timesteps",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_reward",
        "std_reward",
        "mean_episode_length",
        "training_time_seconds",
        "model_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in bench_points:
            row = {
                "budget_timesteps": point.budget_timesteps,
                "trained_timesteps": point.trained_timesteps,
                "success_rate": point.success_rate,
                "collision_rate": point.collision_rate,
                "timeout_rate": point.timeout_rate,
                "mean_reward": point.mean_reward,
                "std_reward": point.std_reward,
                "mean_episode_length": point.mean_episode_length,
                "training_time_seconds": point.training_time_seconds,
                "model_path": point.model_path,
            }
            writer.writerow(row)

    if plot:
        plot_learning_curve(result, plot_path)

    return result


def plot_learning_curve(
    result: LearningCurveBenchmarkResult, output_path: str | Path | None = None
) -> Path:
    """Render a lightweight learning curve plot for success rate and mean reward."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Plotting requires matplotlib. Install optional visualization dependencies to enable --plot."
        ) from exc

    plot_target = Path(output_path) if output_path is not None else result.plot_path
    if plot_target is None:
        raise ValueError("An output path is required for plotting.")
    plot_target.parent.mkdir(parents=True, exist_ok=True)

    budgets = [int(point.budget_timesteps) for point in result.points]
    success_rates = [point.success_rate for point in result.points]
    rewards = [point.mean_reward for point in result.points]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(budgets, success_rates, marker="o", linewidth=2)
    axes[0].set_title("Success rate vs training budget")
    axes[0].set_xlabel("Training budget (timesteps)")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].plot(budgets, rewards, marker="s", linewidth=2, color="tab:orange")
    axes[1].set_title("Mean reward vs training budget")
    axes[1].set_xlabel("Training budget (timesteps)")
    axes[1].set_ylabel("Mean reward")

    fig.savefig(plot_target, dpi=160)
    plt.close(fig)
    return plot_target
