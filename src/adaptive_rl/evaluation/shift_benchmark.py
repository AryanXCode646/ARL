"""Controlled distribution-shift benchmark schemas, gap metrics, and reporting.

Implements Issue #105: a reusable benchmark protocol in which a policy is trained
exclusively under one TRAIN environment distribution and then evaluated, without
adaptation, on several TEST distributions with shifted environment parameters
(obstacle density, wind, dynamic obstacles, injected disturbance).

Design notes:
- `ShiftScenario` generalizes `EvaluationScenario` (single ``seed`` becomes a
  ``seeds`` list plus a ``role``); the ``environment_overrides`` field name and
  semantics are kept identical.
- Train/test seed integrity reuses `GeneralizationDistribution` (strict disjoint
  validation) instead of reimplementing overlap checks.
- Per-scenario evaluation reuses `GeneralizationEvaluator.evaluate_seeds` and the
  shared `aggregate_seed_evaluation` aggregation, so success/collision/reward/
  recovery semantics match the existing generalization pipeline exactly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from adaptive_rl.config import ConfigError, ExperimentConfig
from adaptive_rl.environments.registry import RegistryError, make_env
from adaptive_rl.evaluation.generalization import GeneralizationDistribution
from adaptive_rl.evaluation.metrics import EvaluationMetrics

#: Machine-readable version of the shift-benchmark schema and report format.
SHIFT_BENCHMARK_SCHEMA_VERSION = "1.0"

#: Canonical statement of the recovery-time definition embedded in every report.
RECOVERY_DEFINITION: Dict[str, Any] = {
    "metric": "recovery_time",
    "unit": "environment_steps",
    "disturbance_event": "transient wind magnitude ||gust + injected_disturbance|| "
    "(m/s, steady wind excluded) reaching the per-scenario disturbance_event_threshold",
    "disturbance_onset": "first step at or above threshold while no event is active "
    "(events never overlap)",
    "recovery": "magnitude strictly below threshold AND drone speed within "
    "recovery_speed_tolerance (m/s) of the onset reference speed, sustained for "
    "recovery_hold_steps consecutive steps",
    "recovery_time": "recovery_step - onset_step (>= 1 by construction; never wall-clock time)",
    "censored": "an event still open at episode end contributes to recovery_events "
    "but never to a recovery-time mean and is never reported as 0",
    "unavailable": "episodes with zero disturbance events have undefined recovery "
    "(reported as null, never 0.0)",
    "aggregation": "scenario recovery_time is the mean over measured episodes only; "
    "per-episode class counts are stored as recovery_episodes_measured / "
    "recovery_episodes_unavailable / recovery_episodes_censored",
    "telemetry_contract": [
        "disturbance_magnitude",
        "disturbance_threshold",
        "disturbance_active",
        "disturbance_onset_step",
        "recovered",
        "recovery_time",
        "recovery_events",
        "recovery_completed",
        "recovery_censored",
        "recovery_times",
        "episode_recovery_time",
    ],
}


class ShiftScenario(BaseModel):
    """One controlled distribution in a shift benchmark (TRAIN or TEST).

    Generalizes `EvaluationScenario`: the single ``seed`` becomes an explicit
    ``seeds`` list and a ``role`` distinguishes the training distribution from
    test distributions. ``environment_overrides`` keeps the exact name and
    semantics of `EvaluationScenario.environment_overrides`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Unique scenario name (e.g. TRAIN, TEST-A)")
    role: Literal["train", "test"] = Field(
        ..., description="Scenario role: exactly one 'train', one or more 'test'"
    )
    description: str = Field(
        default="", description="Human-readable description of the distribution shift"
    )
    seeds: List[int] = Field(..., description="Deterministic evaluation seeds (one episode each)")
    environment_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment parameters overridden for this scenario",
    )


class DistributionShiftBenchmarkSpec(BaseModel):
    """Validated specification of a controlled distribution-shift benchmark."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Benchmark name")
    version: str = Field(
        default=SHIFT_BENCHMARK_SCHEMA_VERSION,
        description="Benchmark schema version",
    )
    environment_name: str = Field(
        ..., min_length=1, description="Registered environment used by all scenarios"
    )
    base_environment_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment parameters shared by every scenario",
    )
    scenarios: List[ShiftScenario] = Field(
        ..., description="Exactly one TRAIN scenario and one or more TEST scenarios"
    )
    evaluation_deterministic: bool = Field(
        default=True, description="Whether scenario evaluation uses deterministic actions"
    )

    @model_validator(mode="after")
    def _validate_benchmark_structure(self) -> DistributionShiftBenchmarkSpec:
        names = [s.name for s in self.scenarios]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Scenario names must be unique, found duplicates: {duplicates}")
        train = [s for s in self.scenarios if s.role == "train"]
        if len(train) != 1:
            raise ValueError(f"Benchmark requires exactly one TRAIN scenario, found {len(train)}")
        test = [s for s in self.scenarios if s.role == "test"]
        if not test:
            raise ValueError("Benchmark requires at least one TEST scenario")
        for s in self.scenarios:
            if not s.seeds:
                raise ValueError(f"Scenario '{s.name}' must define a non-empty seed list")
        # Reuse the canonical disjoint-seed validation (raises on any overlap).
        GeneralizationDistribution(
            train_seeds=list(train[0].seeds),
            test_seeds=[seed for s in test for seed in s.seeds],
            description=f"Disjoint train/test seed partition for {self.name}",
        )
        return self

    @property
    def train_scenario(self) -> ShiftScenario:
        """Return the single TRAIN scenario (validated to exist exactly once)."""
        for s in self.scenarios:
            if s.role == "train":
                return s
        raise ValueError("Benchmark specification has no TRAIN scenario")

    @property
    def test_scenarios(self) -> List[ShiftScenario]:
        """Return TEST scenarios in declaration order."""
        return [s for s in self.scenarios if s.role == "test"]

    @property
    def train_seeds(self) -> List[int]:
        """Seeds of the TRAIN scenario (also the training distribution)."""
        return list(self.train_scenario.seeds)

    @property
    def all_test_seeds(self) -> List[int]:
        """Union of all TEST scenario seeds (order-preserving, deduplicated)."""
        seen: List[int] = []
        for s in self.test_scenarios:
            for seed in s.seeds:
                if seed not in seen:
                    seen.append(seed)
        return seen

    def scenario_environment_parameters(self, scenario: ShiftScenario) -> Dict[str, Any]:
        """Merge shared base parameters with scenario overrides (overrides win)."""
        params = dict(self.base_environment_parameters)
        params.update(dict(scenario.environment_overrides))
        return params

    def validate_environments(self) -> Dict[str, Dict[str, Any]]:
        """Trial-construct every scenario environment to fail fast on bad overrides.

        Instantiates each scenario with its effective parameters, performs one
        reset, and closes it. Unsupported parameters raise a clear error naming
        the offending scenario instead of failing mid-benchmark.

        Returns:
            Mapping of scenario name to the effective parameter dict used.
        """
        effective: Dict[str, Dict[str, Any]] = {}
        for scenario in self.scenarios:
            params = self.scenario_environment_parameters(scenario)
            try:
                env = make_env(self.environment_name, **params)
            except RegistryError as err:
                raise ValueError(
                    f"Scenario '{scenario.name}': cannot construct environment "
                    f"'{self.environment_name}' with parameters {params}: {err}"
                ) from err
            try:
                env.reset(seed=int(scenario.seeds[0]))
            except Exception as err:
                env.close()
                raise ValueError(
                    f"Scenario '{scenario.name}': environment reset failed with "
                    f"parameters {params}: {err}"
                ) from err
            env.close()
            effective[scenario.name] = params
        return effective


class ScenarioGaps(BaseModel):
    """Generalization gaps of one TEST scenario relative to TRAIN.

    Directionality convention (documented, uniform): a *positive* gap always means
    the TEST scenario performed *worse* than TRAIN.

    - success_gap = train_success_rate - test_success_rate (higher success better)
    - reward_gap = train_mean_reward - test_mean_reward (higher reward better)
    - collision_gap = test_collision_rate - train_collision_rate (lower collision better)
    - recovery_gap = test_recovery_time - train_recovery_time (lower recovery better)
    """

    model_config = ConfigDict(extra="forbid")

    success_gap: Optional[float] = Field(
        default=None, description="train_success_rate - test_success_rate (None if undefined)"
    )
    reward_gap: float = Field(..., description="train_mean_reward - test_mean_reward")
    collision_gap: Optional[float] = Field(
        default=None, description="test_collision_rate - train_collision_rate (None if undefined)"
    )
    recovery_gap: Optional[float] = Field(
        default=None, description="test_recovery_time - train_recovery_time (None if undefined)"
    )


def compute_scenario_gaps(
    train_metrics: EvaluationMetrics,
    test_metrics: EvaluationMetrics,
) -> ScenarioGaps:
    """Compute TEST-vs-TRAIN generalization gaps with explicit directionality.

    Positive gap = degradation on TEST for every metric. Optional metrics
    propagate None (unavailable) instead of collapsing into 0.0.
    """
    if train_metrics.success_rate is not None and test_metrics.success_rate is not None:
        success_gap: Optional[float] = float(train_metrics.success_rate - test_metrics.success_rate)
    else:
        success_gap = None
    if train_metrics.collision_rate is not None and test_metrics.collision_rate is not None:
        collision_gap: Optional[float] = float(
            test_metrics.collision_rate - train_metrics.collision_rate
        )
    else:
        collision_gap = None
    if train_metrics.recovery_time is not None and test_metrics.recovery_time is not None:
        recovery_gap: Optional[float] = float(
            test_metrics.recovery_time - train_metrics.recovery_time
        )
    else:
        recovery_gap = None
    return ScenarioGaps(
        success_gap=success_gap,
        reward_gap=float(train_metrics.mean_reward - test_metrics.mean_reward),
        collision_gap=collision_gap,
        recovery_gap=recovery_gap,
    )


class ScenarioResult(BaseModel):
    """Evaluation outcome of a single benchmark scenario."""

    model_config = ConfigDict(extra="forbid")

    scenario_name: str = Field(..., description="Scenario name")
    role: Literal["train", "test"] = Field(..., description="Scenario role")
    description: str = Field(default="", description="Scenario description")
    seeds: List[int] = Field(..., description="Evaluation seeds used")
    environment_overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Scenario-specific environment overrides"
    )
    effective_environment_parameters: Dict[str, Any] = Field(
        ..., description="Concrete parameters the scenario environment was built with"
    )
    metrics: EvaluationMetrics = Field(..., description="Measured scenario metrics")
    gaps: Optional[ScenarioGaps] = Field(
        default=None, description="TEST-vs-TRAIN gaps (None for the TRAIN scenario)"
    )
    policy_fingerprint: str = Field(
        ..., description="Hash of the evaluated policy weights at scenario evaluation time"
    )


class ShiftBenchmarkReport(BaseModel):
    """Machine-readable artifact of a full distribution-shift benchmark run."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(..., description="Shift-benchmark schema version")
    experiment_name: str = Field(..., description="Experiment identifier")
    environment_name: str = Field(..., description="Registered environment name")
    algorithm_name: str = Field(..., description="RL algorithm used")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC execution timestamp",
    )
    deterministic: bool = Field(..., description="Whether evaluation used deterministic actions")
    total_training_timesteps: int = Field(..., ge=0, description="Training timesteps on TRAIN")
    train_seeds: List[int] = Field(..., description="TRAIN scenario seeds")
    test_seeds: List[int] = Field(..., description="Union of TEST scenario seeds")
    scenarios: List[ScenarioResult] = Field(..., description="Per-scenario results in run order")
    recovery_definition: Dict[str, Any] = Field(
        default_factory=lambda: dict(RECOVERY_DEFINITION),
        description="Formal recovery-time definition used for the reported values",
    )
    training_provenance: Dict[str, Any] = Field(
        default_factory=dict, description="Training configuration and seed-containment audit"
    )
    environment_provenance: Dict[str, Any] = Field(
        default_factory=dict, description="Execution environment provenance (git, packages)"
    )
    config_sha256: Optional[str] = Field(
        default=None, description="SHA-256 of the canonical experiment configuration"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    def save_json(self, path: str | Path) -> Path:
        """Serialize the benchmark report to a formatted JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
        return target


def load_shift_benchmark_config(
    path: str | Path,
) -> Tuple[ExperimentConfig, DistributionShiftBenchmarkSpec]:
    """Load a shift-benchmark YAML file into an experiment config plus benchmark spec.

    The file contains a standard experiment configuration (validated exactly like
    every other AdaptiveRL config) with an additional top-level ``benchmark:``
    block holding the `DistributionShiftBenchmarkSpec` fields.

    Raises:
        ConfigError: If the file is missing, unparsable, lacks the ``benchmark``
            block, or fails either schema validation.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML file at {file_path}: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigError(
            f"Configuration file {file_path} must contain a YAML mapping/dictionary, "
            f"got {type(raw_data).__name__}"
        )

    if "benchmark" not in raw_data or not isinstance(raw_data["benchmark"], dict):
        raise ConfigError(
            f"Configuration file {file_path} must contain a 'benchmark' mapping with the "
            "distribution-shift specification (scenarios, seeds, environment overrides)"
        )

    benchmark_data = raw_data.pop("benchmark")
    try:
        exp_config = ExperimentConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigError(
            f"Experiment configuration validation failed for {file_path}:\n{exc}"
        ) from exc
    try:
        spec = DistributionShiftBenchmarkSpec.model_validate(benchmark_data)
    except ValidationError as exc:
        raise ConfigError(
            f"Shift-benchmark specification validation failed for {file_path}:\n{exc}"
        ) from exc
    return exp_config, spec


__all__ = [
    "SHIFT_BENCHMARK_SCHEMA_VERSION",
    "RECOVERY_DEFINITION",
    "ShiftScenario",
    "DistributionShiftBenchmarkSpec",
    "ScenarioGaps",
    "ScenarioResult",
    "ShiftBenchmarkReport",
    "compute_scenario_gaps",
    "load_shift_benchmark_config",
]
