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
SHIFT_BENCHMARK_SCHEMA_VERSION = "1.1"

#: Canonical statement of the recovery-time definition embedded in every report.
RECOVERY_DEFINITION: Dict[str, Any] = {
    "metric": "recovery_time",
    "unit": "environment_steps",
    "disturbance_event": "transient wind magnitude ||gust + injected_disturbance|| "
    "(m/s, steady wind excluded) reaching the disturbance_event_threshold shared by "
    "every benchmark scenario (fixed event definition; only the disturbance "
    "distribution shifts)",
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
    "aggregation": "scenario recovery_time is the event-weighted mean over completed "
    "recovery events pooled across episodes (episodes with many events weigh "
    "proportionally more; per-episode means are NOT averaged). Event counts and "
    "completion/censoring rates are reported alongside; per-episode class counts "
    "are stored as recovery_episodes_measured / recovery_episodes_unavailable / "
    "recovery_episodes_censored",
    "conditionality_warning": "recovery_time is explicitly conditional on recovery. "
    "It must never be read as an unconditional robustness score: a scenario with "
    "heavy censoring can show a low conditional mean precisely because its hardest "
    "events never recovered. Always interpret it jointly with completion_rate and "
    "censoring_rate; recovery_time_gap is suppressed (null) whenever either side "
    "has censored events",
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
    recovery_event_threshold: Optional[float] = Field(
        default=None,
        description="Fixed disturbance-event threshold (m/s) shared by every scenario; "
        "a scenario's explicit 'disturbance_event_threshold_override' wins if set",
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
            if len(set(s.seeds)) != len(s.seeds):
                duplicate_seeds = sorted({seed for seed in s.seeds if s.seeds.count(seed) > 1})
                raise ValueError(
                    f"Scenario '{s.name}' seeds must be unique, found duplicates: {duplicate_seeds}"
                )
        if self.recovery_event_threshold is not None and self.recovery_event_threshold <= 0.0:
            raise ValueError(
                "Benchmark recovery_event_threshold must be positive, "
                f"got {self.recovery_event_threshold}"
            )
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

    def effective_scenario_parameters(self, scenario: ShiftScenario) -> Dict[str, Any]:
        """Merge base parameters, the benchmark fixed event threshold, and overrides.

        The benchmark-level ``recovery_event_threshold`` is injected as
        ``disturbance_event_threshold_override`` so every scenario shares one event
        definition; a scenario's own explicit
        ``disturbance_event_threshold_override`` wins if set (documented override
        precedence: scenario-specific beats benchmark-wide).
        """
        params = self.scenario_environment_parameters(scenario)
        if (
            self.recovery_event_threshold is not None
            and "disturbance_event_threshold_override" not in scenario.environment_overrides
        ):
            params["disturbance_event_threshold_override"] = float(self.recovery_event_threshold)
        return params

    def validate_environments(self, max_steps: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Preflight every scenario environment, including one real dynamics step.

        Instantiates each scenario with its effective parameters, performs one
        reset, executes one action-space-compatible step, validates the 5-element
        Gymnasium result (observation inside the observation space, scalar reward,
        boolean flags, dict info), and closes it. Any failure raises a clear error
        naming the offending scenario instead of failing after PPO training starts.

        Args:
            max_steps: Optional episode step limit injected when the scenario
                parameters do not define it (mirrors the benchmark runner).

        Returns:
            Mapping of scenario name to the effective parameter dict used.
        """
        effective: Dict[str, Dict[str, Any]] = {}
        for scenario in self.scenarios:
            params = self.effective_scenario_parameters(scenario)
            if max_steps is not None and "max_steps" not in params:
                params["max_steps"] = max_steps
            try:
                env = make_env(self.environment_name, **params)
            except RegistryError as err:
                raise ValueError(
                    f"Scenario '{scenario.name}': cannot construct environment "
                    f"'{self.environment_name}' with parameters {params}: {err}"
                ) from err
            try:
                try:
                    obs, _ = env.reset(seed=int(scenario.seeds[0]))
                    action = env.action_space.sample()
                    step_result = env.step(action)
                except Exception as err:
                    raise ValueError(
                        f"Scenario '{scenario.name}': environment reset/step failed with "
                        f"parameters {params}: {err}"
                    ) from err
                if not isinstance(step_result, tuple) or len(step_result) != 5:
                    raise ValueError(
                        f"Scenario '{scenario.name}': step() must return a 5-element "
                        f"Gymnasium tuple, got {type(step_result).__name__}"
                    )
                step_obs, reward, terminated, truncated, info = step_result
                try:
                    float(reward)
                    bool(terminated)
                    bool(truncated)
                except (TypeError, ValueError) as err:
                    raise ValueError(
                        f"Scenario '{scenario.name}': invalid reward/flag types "
                        f"(reward={reward!r}, terminated={terminated!r}, "
                        f"truncated={truncated!r}): {err}"
                    ) from err
                if not isinstance(info, dict):
                    raise ValueError(
                        f"Scenario '{scenario.name}': step info must be a dict, "
                        f"got {type(info).__name__}"
                    )
                try:
                    obs_inside = bool(env.observation_space.contains(step_obs))
                except Exception as err:
                    raise ValueError(
                        f"Scenario '{scenario.name}': observation-space check failed: {err}"
                    ) from err
                if not obs_inside:
                    raise ValueError(
                        f"Scenario '{scenario.name}': step observation outside observation_space"
                    )
            finally:
                env.close()
            effective[scenario.name] = params
        return effective


class ScenarioGaps(BaseModel):
    """Generalization gaps of one TEST scenario relative to TRAIN.

    Directionality convention: a *positive* gap means the TEST scenario performed
    *worse* than TRAIN — except recovery time, where censoring makes a bare mean
    comparison unsafe (see below).

    - success_gap = train_success_rate - test_success_rate (higher success better)
    - reward_gap = train_mean_reward - test_mean_reward (higher reward better)
    - collision_gap = test_collision_rate - train_collision_rate (lower collision better)
    - recovery_time_gap = test_mean_completed_recovery_time - train_... (lower better),
      defined ONLY when both sides recorded at least one event and both recovery
      completion rates are exactly 1.0 (no censoring anywhere); otherwise None, so
      a heavily censored TEST can never look spuriously faster.
    - recovery_completion_gap = train_completion_rate - test_completion_rate
      (positive = TEST recovers a smaller share of its events).
    - recovery_censoring_gap = test_censoring_rate - train_censoring_rate
      (positive = TEST leaves a larger share of events unrecovered).
    """

    model_config = ConfigDict(extra="forbid")

    success_gap: Optional[float] = Field(
        default=None, description="train_success_rate - test_success_rate (None if undefined)"
    )
    reward_gap: float = Field(..., description="train_mean_reward - test_mean_reward")
    collision_gap: Optional[float] = Field(
        default=None, description="test_collision_rate - train_collision_rate (None if undefined)"
    )
    recovery_time_gap: Optional[float] = Field(
        default=None,
        description="test_mean_completed_recovery_time - train_mean_completed_recovery_time; "
        "None unless both sides have events and 100% completion (censoring-safe)",
    )
    recovery_completion_gap: Optional[float] = Field(
        default=None,
        description="train_completion_rate - test_completion_rate; positive = TEST worse "
        "(None if either rate undefined)",
    )
    recovery_censoring_gap: Optional[float] = Field(
        default=None,
        description="test_censoring_rate - train_censoring_rate; positive = TEST worse "
        "(None if either rate undefined)",
    )


def compute_scenario_gaps(
    train_metrics: EvaluationMetrics,
    test_metrics: EvaluationMetrics,
) -> ScenarioGaps:
    """Compute TEST-vs-TRAIN generalization gaps with explicit directionality.

    Success/reward/collision gaps keep the positive-means-TEST-worse convention.
    The recovery time gap is deliberately suppressed (None) whenever either side
    has censored events or no events at all; the completion/censoring gaps carry
    the robustness signal instead. Optional metrics propagate None (unavailable)
    instead of collapsing into 0.0.
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

    train_completion = train_metrics.recovery_completion_rate
    test_completion = test_metrics.recovery_completion_rate
    if train_completion is not None and test_completion is not None:
        completion_gap: Optional[float] = float(train_completion - test_completion)
    else:
        completion_gap = None

    train_censoring = train_metrics.recovery_censoring_rate
    test_censoring = test_metrics.recovery_censoring_rate
    if train_censoring is not None and test_censoring is not None:
        censoring_gap: Optional[float] = float(test_censoring - train_censoring)
    else:
        censoring_gap = None

    if (
        train_metrics.recovery_time is not None
        and test_metrics.recovery_time is not None
        and train_completion == 1.0
        and test_completion == 1.0
    ):
        time_gap: Optional[float] = float(test_metrics.recovery_time - train_metrics.recovery_time)
    else:
        time_gap = None

    return ScenarioGaps(
        success_gap=success_gap,
        reward_gap=float(train_metrics.mean_reward - test_metrics.mean_reward),
        collision_gap=collision_gap,
        recovery_time_gap=time_gap,
        recovery_completion_gap=completion_gap,
        recovery_censoring_gap=censoring_gap,
    )


class RecoverySummary(BaseModel):
    """Censoring-aware recovery summary of one benchmark scenario.

    `mean_completed_recovery_time` is event-weighted and explicitly conditional on
    recovery: it must be interpreted jointly with `completion_rate` /
    `censoring_rate`, never as an unconditional robustness score.
    """

    model_config = ConfigDict(extra="forbid")

    total_events: int = Field(..., ge=0, description="Total disturbance events observed")
    completed_events: int = Field(
        ..., ge=0, description="Events that completed with a valid recovery"
    )
    censored_events: int = Field(
        ..., ge=0, description="Events still open at episode end (never averaged, never 0-valued)"
    )
    completion_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="completed_events / total_events (None when total is 0)",
    )
    censoring_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="censored_events / total_events (None when total is 0)",
    )
    mean_completed_recovery_time: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Event-weighted mean over completed recovery times in environment "
        "steps (None when no event completed)",
    )

    @classmethod
    def from_metrics(cls, metrics: EvaluationMetrics) -> RecoverySummary:
        """Build a summary from aggregated scenario metrics.

        Counts are exact when the environment exposes the disturbance telemetry
        contract; environments without disturbance dynamics report zero events
        with undefined rates. Rates recomputed here must match the metrics rates.
        """
        total = metrics.recovery_events or 0
        completed = metrics.recovery_completed_events or 0
        censored = metrics.recovery_censored_events or 0
        return cls(
            total_events=total,
            completed_events=completed,
            censored_events=censored,
            completion_rate=metrics.recovery_completion_rate,
            censoring_rate=metrics.recovery_censoring_rate,
            mean_completed_recovery_time=metrics.recovery_time,
        )


class EpisodeBenchmarkRecord(BaseModel):
    """Raw per-episode observation of one evaluated seed.

    Stored for every scenario seed so the reported aggregates (success and
    collision rates, mean reward, event-weighted recovery statistics) can be
    independently recomputed from the artifact.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(..., description="Evaluation seed for this episode")
    reward: float = Field(..., description="Total cumulative episodic reward")
    length: int = Field(..., ge=0, description="Episode step count")
    success: Optional[bool] = Field(
        default=None, description="Episode success (None when undefined)"
    )
    collision: Optional[bool] = Field(
        default=None, description="Episode collision (None when undefined)"
    )
    terminated: bool = Field(..., description="Natural termination flag")
    truncated: bool = Field(..., description="Truncation/timeout flag")
    recovery_times: List[int] = Field(
        default_factory=list,
        description="Completed recovery times in environment steps for this episode",
    )
    recovery_events: int = Field(
        default=0, ge=0, description="Disturbance events observed in this episode"
    )
    recovery_completed: int = Field(
        default=0, ge=0, description="Events in this episode that completed recovery"
    )
    recovery_censored: int = Field(
        default=0, ge=0, description="Events in this episode still open at episode end"
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
    recovery: RecoverySummary = Field(
        ..., description="Censoring-aware recovery summary for this scenario"
    )
    episodes: List[EpisodeBenchmarkRecord] = Field(
        default_factory=list,
        description="Raw per-seed episode records (one per evaluated seed, in seed order)",
    )
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
    "EpisodeBenchmarkRecord",
    "RecoverySummary",
    "ScenarioGaps",
    "ScenarioResult",
    "ShiftBenchmarkReport",
    "compute_scenario_gaps",
    "load_shift_benchmark_config",
]
