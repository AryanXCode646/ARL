"""Focused tests for PPO learning-curve benchmarking."""

import csv
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from adaptive_rl.benchmarking import (
    LearningCurveBenchmarkResult,
    run_learning_curve_benchmark,
    validate_budgets,
)
from adaptive_rl.config import (
    AlgorithmConfig,
    BenchmarkConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
    load_config,
    save_config,
)
from adaptive_rl.evaluation.evaluator import Evaluator


def _make_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="benchmark_test",
        seed=31,
        algorithm=AlgorithmConfig(
            name="ppo",
            batch_size=32,
            parameters={"n_steps": 64, "n_epochs": 1, "seed": 999},
        ),
        environment=EnvironmentConfig(
            name="drone",
            max_steps=8,
            parameters={"bounds": [20.0, 20.0, 10.0], "num_obstacles": 1},
        ),
        training=TrainingConfig(total_timesteps=256, checkpoint_freq=0, log_interval=1),
        evaluation=EvaluationConfig(eval_episodes=1, deterministic=True),
        output_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        benchmark=BenchmarkConfig(
            budgets=[128, 64],
            training_seed=17,
            evaluation_seeds=[11, 12],
            evaluation_episodes=1,
            deterministic=True,
        ),
    )


@pytest.mark.parametrize(
    ("raw_budgets", "message"),
    [
        ([], "empty"),
        ([0], "positive"),
        ([-1], "positive"),
        ("foo,128", "Malformed budget value"),
        ("64,,128", "Malformed budget list"),
        ([64, 64], "Duplicate"),
    ],
)
def test_budget_validation_rejects_invalid_values(raw_budgets: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_budgets(raw_budgets)


def test_budget_validation_normalizes_valid_values() -> None:
    assert validate_budgets("128,64") == [64, 128]
    assert validate_budgets([128, 64]) == [64, 128]


@pytest.mark.parametrize("budgets", [[], [0], [-1], [64, 64], ["64"]])
def test_benchmark_config_rejects_invalid_budgets(budgets: Any) -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig(budgets=budgets)


def test_benchmark_config_defaults_and_old_config_compatibility(tmp_path: Path) -> None:
    benchmark = BenchmarkConfig()
    assert benchmark.budgets == [5000, 10000, 25000, 50000]
    assert benchmark.training_seed == 42
    assert benchmark.evaluation_seeds
    assert benchmark.evaluation_episodes > 0

    old_config_path = tmp_path / "old_config.yaml"
    old_config_path.write_text(
        """
name: old_config
seed: 4
algorithm:
  name: ppo
environment:
  name: drone
training:
  total_timesteps: 128
evaluation:
  eval_episodes: 2
""",
        encoding="utf-8",
    )
    loaded = load_config(old_config_path)
    assert loaded.name == "old_config"
    assert loaded.benchmark is None

    benchmark_config_path = tmp_path / "benchmark_config.yaml"
    benchmark_config_path.write_text(
        """
name: configured_benchmark
algorithm:
  name: ppo
environment:
  name: drone
training:
  total_timesteps: 128
evaluation:
  eval_episodes: 2
benchmark:
  budgets: [128, 64]
  training_seed: 7
  evaluation_seeds: [20, 21]
  evaluation_episodes: 3
  deterministic: false
""",
        encoding="utf-8",
    )
    configured = load_config(benchmark_config_path)
    assert configured.benchmark is not None
    assert configured.benchmark.budgets == [64, 128]
    assert configured.benchmark.training_seed == 7
    assert configured.benchmark.evaluation_seeds == [20, 21]
    assert configured.benchmark.evaluation_episodes == 3
    assert configured.benchmark.deterministic is False


def test_core_and_benchmark_imports_do_not_load_optional_rl_stack() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    script = textwrap.dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder

        class BlockOptionalRLImports(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "gymnasium" or fullname.startswith(
                    ("stable_baselines3", "torch")
                ):
                    raise AssertionError(f"Optional RL dependency imported eagerly: {fullname}")
                return None

        sys.meta_path.insert(0, BlockOptionalRLImports())
        import adaptive_rl
        from adaptive_rl.benchmarking import run_learning_curve_benchmark

        assert callable(run_learning_curve_benchmark)
        assert callable(adaptive_rl.run_learning_curve_benchmark)
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository_root / "src"), environment.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_learning_curve_benchmark_execution_and_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl.benchmarking import learning_curve
    from adaptive_rl.training.trainer import PPOTrainer

    config = _make_config(tmp_path)
    original = config.model_dump(mode="python")
    real_make_env = learning_curve._make_env
    real_evaluate_model = learning_curve._evaluate_model
    trainers: list[Any] = []
    envs: list[gym.Env] = []
    evaluation_calls: list[dict[str, Any]] = []

    class TrackingTrainer(PPOTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.fit_calls = 0
            trainers.append(self)

        def fit(self) -> Any:
            self.fit_calls += 1
            return super().fit()

    def track_env(env_name: str, **kwargs: Any) -> gym.Env:
        env = real_make_env(env_name, **kwargs)
        envs.append(env)
        return env

    def track_trainer(config: ExperimentConfig, env: gym.Env) -> PPOTrainer:
        return TrackingTrainer(config=config, env=env)

    def track_evaluation(*args: Any, **kwargs: Any) -> Any:
        evaluation_calls.append(kwargs.copy())
        return real_evaluate_model(*args, **kwargs)

    monkeypatch.setattr(learning_curve, "_make_trainer", track_trainer)
    monkeypatch.setattr(learning_curve, "_make_env", track_env)
    monkeypatch.setattr(learning_curve, "_evaluate_model", track_evaluation)

    result = run_learning_curve_benchmark(config, budgets=[128, 64])

    assert result.budgets == [64, 128]
    assert [point.budget_timesteps for point in result.points] == [64, 128]
    assert len(result.points) == len(trainers) == 2
    assert [trainer.fit_calls for trainer in trainers] == [1, 1]
    assert [trainer.config.training.total_timesteps for trainer in trainers] == [64, 128]
    assert [trainer.config.seed for trainer in trainers] == [17, 17]
    assert [trainer.config.name for trainer in trainers] == ["ppo_budget_64", "ppo_budget_128"]
    assert trainers[0].config.output_dir != trainers[1].config.output_dir
    assert trainers[0].config.log_dir != trainers[1].config.log_dir
    assert [
        (trainer.config.training.checkpoint_freq, trainer.config.training.log_interval)
        for trainer in trainers
    ] == [(0, 1), (0, 1)]
    assert [trainer.algorithm.hyperparameters["seed"] for trainer in trainers] == [17, 17]
    expected_algorithm_config = config.algorithm.model_dump()
    expected_algorithm_config["parameters"].pop("seed")
    assert [trainer.config.algorithm.model_dump() for trainer in trainers] == [
        expected_algorithm_config,
        expected_algorithm_config,
    ]
    assert [trainer.config.environment.model_dump() for trainer in trainers] == [
        config.environment.model_dump(),
        config.environment.model_dump(),
    ]
    assert [trainer.config.evaluation.model_dump() for trainer in trainers] == [
        config.evaluation.model_dump(),
        config.evaluation.model_dump(),
    ]
    assert len({id(trainer.env) for trainer in trainers}) == 2
    assert len({id(trainer.algorithm.model) for trainer in trainers}) == 2
    assert [trainer.algorithm.num_timesteps for trainer in trainers] == [64, 128]
    assert all(env is trainer.env for env, trainer in zip((envs[0], envs[2]), trainers))
    assert len({id(env) for env in envs}) == 4

    for call in evaluation_calls:
        assert call["evaluation_seeds"] == [11, 12]
        assert call["evaluation_episodes"] == 1
        assert call["deterministic"] is True
        assert call["env_name"] == config.environment.name
        assert call["env_kwargs"] == config.environment.parameters
    assert len(evaluation_calls) == 2

    model_paths = [Path(point.model_path) for point in result.points]
    assert len(set(model_paths)) == 2
    assert all(path.is_file() for path in model_paths)
    assert all(path.parent.parent.parent.name == "learning_curve" for path in model_paths)
    assert result.json_path == config.output_dir / "benchmarks" / "learning_curve_budget.json"
    assert result.csv_path == config.output_dir / "benchmarks" / "learning_curve_budget.csv"

    for point in result.points:
        assert point.trained_timesteps == point.budget_timesteps
        assert point.success_rate is None or 0.0 <= point.success_rate <= 1.0
        assert point.collision_rate is None or 0.0 <= point.collision_rate <= 1.0
        assert point.timeout_rate is None or 0.0 <= point.timeout_rate <= 1.0
        assert math.isfinite(point.mean_reward)
        assert point.std_reward is None or point.std_reward >= 0.0
        assert point.mean_episode_length >= 0.0
        assert math.isfinite(point.training_time_seconds)
        assert point.training_time_seconds >= 0.0

    assert config.name == original["name"]
    assert config.training.total_timesteps == original["training"]["total_timesteps"]
    assert config.output_dir == original["output_dir"]
    assert config.log_dir == original["log_dir"]
    assert config.evaluation.model_dump() == original["evaluation"]
    assert config.algorithm.parameters == original["algorithm"]["parameters"]
    assert config.model_dump(mode="python") == original

    assert result.json_path is not None and result.json_path.is_file()
    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    json.dumps(data, allow_nan=False)
    assert data["benchmark"]["training_seed"] == 17
    assert data["benchmark"]["evaluation_seeds"] == [11, 12]
    assert data["benchmark"]["evaluation_episodes"] == 1
    assert data["benchmark"]["deterministic"] is True
    assert data["benchmark"]["budgets"] == [64, 128]
    assert len(data["results"]) == 2
    required_metrics = {
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
    }
    for row in data["results"]:
        assert required_metrics <= row.keys()
        assert Path(row["model_path"]).is_file()
    assert set(data["plot_data"]) == {"budgets", "success_rate", "mean_reward"}
    assert data["benchmark"]["metric_semantics"]["aggregation"] == (
        "pooled episode-level descriptive statistics"
    )
    assert data["benchmark"]["training_time_semantics"].startswith("monotonic elapsed time")
    assert all(len(row["per_seed_summaries"]) == 2 for row in data["results"])
    assert all("success_rate" in row["cross_seed_statistics"] for row in data["results"])

    assert result.csv_path is not None and result.csv_path.is_file()
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == [
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
    assert len(rows) == len(result.points)
    assert not any(field.startswith("Unnamed:") for field in reader.fieldnames or [])
    assert [int(row["budget_timesteps"]) for row in rows] == [64, 128]
    assert all(math.isfinite(float(row["mean_reward"])) for row in rows)
    assert all(Path(row["model_path"]).is_file() for row in rows)


def test_learning_curve_benchmark_repeats_deterministically(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    first = run_learning_curve_benchmark(
        config,
        budgets=[65],
        output_dir=tmp_path / "first",
    )
    second = run_learning_curve_benchmark(
        config,
        budgets=[65],
        output_dir=tmp_path / "second",
    )

    first_point = first.points[0]
    second_point = second.points[0]
    assert first.budgets == second.budgets == [65]
    assert first_point.trained_timesteps == second_point.trained_timesteps == 128
    assert first.training_seed == second.training_seed == 17
    assert first.evaluation_seeds == second.evaluation_seeds == [11, 12]
    assert first.evaluation_episodes == second.evaluation_episodes == 1
    assert first.deterministic is second.deterministic is True
    for field in (
        "budget_timesteps",
        "trained_timesteps",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "mean_reward",
        "std_reward",
        "mean_episode_length",
        "training_seed",
        "evaluation_seeds",
        "evaluation_episodes",
        "deterministic",
    ):
        assert getattr(first_point, field) == getattr(second_point, field)


def test_benchmark_keeps_single_episode_standard_deviation_unavailable(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    assert config.benchmark is not None
    config.benchmark.evaluation_seeds = [11]
    config.benchmark.evaluation_episodes = 1

    result = run_learning_curve_benchmark(
        config,
        budgets=[64],
        output_dir=tmp_path / "single_episode",
    )

    point = result.points[0]
    assert point.std_reward is None
    assert result.to_dict()["results"][0]["std_reward"] is None
    assert result.csv_path is not None
    with result.csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["std_reward"] == ""


def test_default_benchmark_evaluation_does_not_derive_seed_count_from_episode_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl.benchmarking import learning_curve

    config = _make_config(tmp_path)
    config.benchmark = None
    config.evaluation.eval_episodes = 2
    calls: list[dict[str, Any]] = []

    def fake_run(
        config: ExperimentConfig,
        budget: int,
        **kwargs: Any,
    ) -> learning_curve.LearningCurvePoint:
        calls.append({"budget": budget, **kwargs})
        return learning_curve.LearningCurvePoint(
            budget_timesteps=budget,
            trained_timesteps=budget,
            success_rate=None,
            collision_rate=None,
            timeout_rate=0.0,
            mean_reward=1.0,
            std_reward=0.0,
            mean_episode_length=1.0,
            training_time_seconds=0.0,
            model_path=str(tmp_path / "model.zip"),
            training_seed=kwargs["training_seed"],
            evaluation_seeds=list(kwargs["evaluation_seeds"]),
            evaluation_episodes=kwargs["evaluation_episodes"],
            deterministic=kwargs["deterministic"],
            algorithm="ppo",
            environment="drone",
        )

    monkeypatch.setattr(learning_curve, "_run_single_budget", fake_run)
    result = run_learning_curve_benchmark(config, budgets=[64], output_dir=tmp_path / "out")

    assert result.evaluation_seeds == [42, 43, 44, 45, 46]
    assert result.evaluation_episodes == 20
    assert calls[0]["evaluation_seeds"] == [42, 43, 44, 45, 46]
    assert calls[0]["evaluation_episodes"] == 20
    assert len(calls[0]["evaluation_seeds"]) * calls[0]["evaluation_episodes"] == 100


def test_plotting_is_lazy_and_closes_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl.benchmarking.learning_curve import (
        LearningCurveBenchmarkResult,
        plot_learning_curve,
    )

    figure = MagicMock()
    figure.savefig.side_effect = lambda path, dpi, metadata: Path(path).touch()
    axes = [MagicMock(), MagicMock()]
    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.subplots = MagicMock(return_value=(figure, axes))  # type: ignore[attr-defined]
    pyplot.close = MagicMock()  # type: ignore[attr-defined]
    matplotlib = ModuleType("matplotlib")
    matplotlib.use = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    output_path = tmp_path / "curve.png"
    result = LearningCurveBenchmarkResult(
        benchmark_name="ppo_learning_curve",
        algorithm="ppo",
        environment="drone",
        budgets=[],
        training_seed=0,
        evaluation_seeds=[],
        evaluation_episodes=1,
        deterministic=True,
    )
    path = plot_learning_curve(result, output_path)
    assert path == output_path
    assert output_path.is_file()
    assert figure.savefig.call_args.kwargs["metadata"] == {"Software": "AdaptiveRL"}
    pyplot.close.assert_called_once_with(figure)  # type: ignore[attr-defined]


def test_plotting_reports_missing_matplotlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl.benchmarking.learning_curve import (
        LearningCurveBenchmarkResult,
        plot_learning_curve,
    )

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    result = LearningCurveBenchmarkResult(
        benchmark_name="ppo_learning_curve",
        algorithm="ppo",
        environment="drone",
        budgets=[],
        training_seed=0,
        evaluation_seeds=[],
        evaluation_episodes=1,
        deterministic=True,
    )
    with pytest.raises(RuntimeError, match="requires matplotlib"):
        plot_learning_curve(result, tmp_path / "curve.png")


class _OutcomeEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def __init__(self, outcome: str) -> None:
        super().__init__()
        self.outcome = outcome

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.outcome == "collision":
            return (
                np.zeros(1, dtype=np.float32),
                1.0,
                True,
                False,
                {"success": False, "collision": True},
            )
        if self.outcome == "truncated":
            return (
                np.zeros(1, dtype=np.float32),
                1.0,
                False,
                True,
                {
                    "success": False,
                    "collision": False,
                },
            )
        if self.outcome == "success":
            return (
                np.zeros(1, dtype=np.float32),
                1.0,
                True,
                False,
                {
                    "success": True,
                    "collision": False,
                },
            )
        if self.outcome == "termination":
            return (
                np.zeros(1, dtype=np.float32),
                1.0,
                True,
                False,
                {
                    "success": False,
                    "collision": False,
                },
            )
        return np.zeros(1, dtype=np.float32), 1.0, True, False, {}


class _ConstantPolicy:
    def predict(self, observation: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.zeros(1, dtype=np.float32), None


@pytest.mark.parametrize(
    ("outcome", "expected_success", "expected_collision", "expected_timeout"),
    [
        ("termination", 0.0, 0.0, 0.0),
        ("collision", 0.0, 1.0, 0.0),
        ("truncated", 0.0, 0.0, 1.0),
        ("success", 1.0, 0.0, 0.0),
    ],
)
def test_evaluator_uses_gymnasium_truncation_signal(
    outcome: str,
    expected_success: float,
    expected_collision: float,
    expected_timeout: float,
) -> None:
    env = _OutcomeEnv(outcome)
    evaluator = Evaluator(algorithm=_ConstantPolicy(), env=env)  # type: ignore[arg-type]
    metrics = evaluator.evaluate(num_episodes=1, deterministic=True, base_seed=2)

    assert metrics.success_rate == expected_success
    assert metrics.collision_rate == expected_collision
    assert metrics.truncation_rate == expected_timeout
    assert evaluator.last_episode_records[0].truncated is (expected_timeout == 1.0)
    evaluator.close()


def test_evaluator_preserves_unavailable_outcome_metrics() -> None:
    env = _OutcomeEnv("unavailable")
    evaluator = Evaluator(algorithm=_ConstantPolicy(), env=env)  # type: ignore[arg-type]
    metrics = evaluator.evaluate(num_episodes=1)
    assert metrics.success_rate is None
    assert metrics.collision_rate is None
    evaluator.close()


def test_benchmark_cli_dispatch_and_budget_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adaptive_rl import benchmarking
    from adaptive_rl.cli import app

    config_path = tmp_path / "benchmark.yaml"
    save_config(_make_config(tmp_path), config_path)
    runner = CliRunner()
    benchmark_help = runner.invoke(app, ["benchmark", "--help"])
    assert benchmark_help.exit_code == 0
    assert "budgets" in benchmark_help.output
    budget_help = runner.invoke(app, ["benchmark", "budgets", "--help"])
    assert budget_help.exit_code == 0
    for option in (
        "--config",
        "--budgets",
        "--training-seed",
        "--eval-seeds",
        "--episodes",
        "--deterministic",
        "--stochastic",
        "--output-dir",
        "--plot",
        "--no-plot",
    ):
        assert option in budget_help.output
    dispatch: list[dict[str, Any]] = []
    fake_result = LearningCurveBenchmarkResult(
        benchmark_name="ppo_learning_curve",
        algorithm="ppo",
        environment="drone",
        budgets=[64, 128],
        training_seed=17,
        evaluation_seeds=[11, 12],
        evaluation_episodes=1,
        deterministic=True,
        json_path=tmp_path / "learning_curve_budget.json",
        csv_path=tmp_path / "learning_curve_budget.csv",
    )

    def fake_benchmark(config: ExperimentConfig, **kwargs: Any) -> LearningCurveBenchmarkResult:
        dispatch.append({"config": config, **kwargs})
        return fake_result

    monkeypatch.setattr(benchmarking, "run_learning_curve_benchmark", fake_benchmark)
    success = runner.invoke(
        app,
        [
            "benchmark",
            "budgets",
            "--config",
            str(config_path),
            "--budgets",
            "64,128",
            "--training-seed",
            "17",
            "--eval-seeds",
            "11,12",
            "--episodes",
            "1",
        ],
    )
    assert success.exit_code == 0, success.output
    assert "learning-curve benchmark complete" in success.output
    assert "64, 128" in success.output
    assert len(dispatch) == 1
    assert dispatch[0]["budgets"] == [64, 128]
    assert dispatch[0]["training_seed"] == 17
    assert dispatch[0]["evaluation_seeds"] == [11, 12]
    assert dispatch[0]["evaluation_episodes"] == 1

    for invalid in ("64,-1", "foo,128"):
        result = runner.invoke(
            app,
            [
                "benchmark",
                "budgets",
                "--config",
                str(config_path),
                "--budgets",
                invalid,
            ],
        )
        assert result.exit_code == 1
        assert "Benchmark failed with error" in result.output
        assert "Traceback" not in result.output
    assert len(dispatch) == 1

    malformed_seeds = runner.invoke(
        app,
        [
            "benchmark",
            "budgets",
            "--config",
            str(config_path),
            "--budgets",
            "64,128",
            "--eval-seeds",
            "11,,12",
        ],
    )
    assert malformed_seeds.exit_code == 1
    assert "Malformed evaluation seed list" in malformed_seeds.output
    assert "Traceback" not in malformed_seeds.output
    assert len(dispatch) == 1

    invalid_seed_value = runner.invoke(
        app,
        [
            "benchmark",
            "budgets",
            "--config",
            str(config_path),
            "--budgets",
            "64,128",
            "--eval-seeds",
            "foo,12",
        ],
    )
    assert invalid_seed_value.exit_code == 1
    assert "Malformed evaluation seed value: 'foo'" in invalid_seed_value.output
    assert len(dispatch) == 1
