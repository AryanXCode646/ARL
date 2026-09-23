"""Evaluation benchmarks, metrics, and scenario interfaces for AdaptiveRL."""

from adaptive_rl.evaluation.evaluator import BaseEvaluator, Evaluator
from adaptive_rl.evaluation.generalization import (
    GeneralizationDistribution,
    GeneralizationEvaluator,
    GeneralizationReport,
)
from adaptive_rl.evaluation.metrics import (
    EvaluationMetrics,
    StandardizedExperimentMetrics,
)
from adaptive_rl.evaluation.scenarios import EvaluationScenario
from adaptive_rl.evaluation.seeding import (
    derive_evaluation_seed,
    derive_planner_seed,
    generate_evaluation_seeds,
)
from adaptive_rl.evaluation.shift_benchmark import (
    RECOVERY_DEFINITION,
    SHIFT_BENCHMARK_SCHEMA_VERSION,
    DistributionShiftBenchmarkSpec,
    EpisodeBenchmarkRecord,
    RecoverySummary,
    ScenarioGaps,
    ScenarioResult,
    ShiftBenchmarkReport,
    ShiftScenario,
    compute_scenario_gaps,
    load_shift_benchmark_config,
)
from adaptive_rl.metrics import (
    EpisodeMetrics as EpisodeMetrics,
)
from adaptive_rl.metrics import (
    EpisodeMetricsAccumulator as EpisodeMetricsAccumulator,
)
from adaptive_rl.metrics import (
    compute_rate as compute_rate,
)
from adaptive_rl.metrics import (
    extract_episode_metrics as extract_episode_metrics,
)

__all__ = [
    "BaseEvaluator",
    "Evaluator",
    "EvaluationMetrics",
    "EvaluationScenario",
    "DistributionShiftBenchmarkSpec",
    "EpisodeBenchmarkRecord",
    "GeneralizationDistribution",
    "GeneralizationEvaluator",
    "GeneralizationReport",
    "RECOVERY_DEFINITION",
    "RecoverySummary",
    "SHIFT_BENCHMARK_SCHEMA_VERSION",
    "ScenarioGaps",
    "ScenarioResult",
    "ShiftBenchmarkReport",
    "ShiftScenario",
    "StandardizedExperimentMetrics",
    "compute_scenario_gaps",
    "derive_evaluation_seed",
    "derive_planner_seed",
    "generate_evaluation_seeds",
    "load_shift_benchmark_config",
]
