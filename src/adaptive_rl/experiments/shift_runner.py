"""Runner for controlled distribution-shift benchmarks (Issue #105).

Protocol enforced by `DistributionShiftBenchmarkRunner`:

1. The TRAIN environment is constructed from the shared base parameters merged
   with TRAIN-scenario overrides only.
2. The policy is trained exclusively on TRAIN conditions (training seeds drawn
   only from the TRAIN seed set via `TrainingDistributionWrapper`).
3. Training completes fully before any test evaluation begins.
4. The frozen trained policy (never fine-tuned or adapted) is evaluated on
   independent TRAIN/TEST scenario environments with explicitly recorded,
   mutually disjoint train/test seed sets.
5. Effective environment parameters, a policy fingerprint, a censoring-aware
   recovery summary, and raw per-seed episode records are stored per scenario,
   so a reader can verify exactly what was evaluated and recompute every
   aggregate from the artifact.
6. Recovery time gaps are reported only when both sides fully recovered
   (100% completion); otherwise the completion/censoring gaps carry the signal.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.config import ExperimentConfig, compute_config_sha256
from adaptive_rl.environments.registry import make_env
from adaptive_rl.environments.seeded_wrapper import TrainingDistributionWrapper
from adaptive_rl.evaluation.generalization import GeneralizationEvaluator
from adaptive_rl.evaluation.metrics import EvaluationMetrics
from adaptive_rl.evaluation.shift_benchmark import (
    SHIFT_BENCHMARK_SCHEMA_VERSION,
    DistributionShiftBenchmarkSpec,
    EpisodeBenchmarkRecord,
    RecoverySummary,
    ScenarioResult,
    ShiftBenchmarkReport,
    ShiftScenario,
    compute_scenario_gaps,
)
from adaptive_rl.experiments.provenance import collect_environment_provenance
from adaptive_rl.experiments.runner import BaseExperimentRunner
from adaptive_rl.metrics import EpisodeMetrics
from adaptive_rl.training.trainer import PPOTrainer, SACTrainer


def policy_fingerprint(algorithm: BaseAlgorithm) -> str:
    """Hash the policy weights of a trained SB3-backed algorithm.

    The fingerprint is recorded per scenario; identical fingerprints prove the
    same frozen policy was evaluated across all scenarios.
    """
    model = getattr(algorithm, "model", None)
    if model is None:
        raise RuntimeError("Cannot fingerprint policy: algorithm has no trained 'model' attribute")
    policy = getattr(model, "policy", None)
    if policy is None or not hasattr(policy, "state_dict"):
        raise RuntimeError(
            "Cannot fingerprint policy: model has no 'policy.state_dict()' interface"
        )
    digest = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        digest.update(name.encode("utf-8"))
        try:
            digest.update(bytes(tensor.cpu().numpy().tobytes()))
        except Exception as err:
            raise RuntimeError(f"Cannot fingerprint policy parameter '{name}': {err}") from err
    return digest.hexdigest()


class DistributionShiftBenchmarkRunner(BaseExperimentRunner):
    """Train once on TRAIN conditions, evaluate frozen on every shift scenario."""

    def __init__(self, spec: DistributionShiftBenchmarkSpec) -> None:
        """Initialize the runner with a validated benchmark specification.

        Args:
            spec: Validated `DistributionShiftBenchmarkSpec` (exactly one TRAIN
                scenario, at least one TEST scenario, disjoint seeds).
        """
        self.spec = spec
        self.last_report: Optional[ShiftBenchmarkReport] = None

    def _scenario_env_params(
        self, config: ExperimentConfig, scenario: ShiftScenario
    ) -> Dict[str, Any]:
        """Merge base parameters, the benchmark fixed threshold, overrides, max_steps."""
        params = self.spec.effective_scenario_parameters(scenario)
        if "max_steps" not in params:
            params["max_steps"] = config.environment.max_steps
        return params

    @staticmethod
    def _episode_records(
        scenario: ShiftScenario, episode_metrics: List[EpisodeMetrics]
    ) -> List[EpisodeBenchmarkRecord]:
        """Build raw per-seed records aligned with evaluation order (no re-evaluation)."""
        records: List[EpisodeBenchmarkRecord] = []
        for seed, m in zip([int(s) for s in scenario.seeds], episode_metrics):
            extra = m.additional_metrics if isinstance(m.additional_metrics, dict) else {}
            records.append(
                EpisodeBenchmarkRecord(
                    seed=seed,
                    reward=float(m.reward),
                    length=int(m.length),
                    success=m.success,
                    collision=m.collision,
                    terminated=bool(m.terminated),
                    truncated=bool(m.truncated),
                    recovery_times=[int(t) for t in (extra.get("recovery_times") or [])],
                    recovery_events=int(extra.get("recovery_events") or 0),
                    recovery_completed=int(extra.get("recovery_completed") or 0),
                    recovery_censored=int(extra.get("recovery_censored") or 0),
                )
            )
        return records

    def run(self, config: ExperimentConfig) -> ShiftBenchmarkReport:
        """Execute the full benchmark: train on TRAIN, evaluate on all scenarios.

        Args:
            config: Validated experiment configuration (algorithm, training,
                evaluation determinism, output directory).

        Returns:
            ShiftBenchmarkReport: Complete machine-readable benchmark artifact.

        Raises:
            ValueError: On unsupported algorithms or invalid scenario environments.
            RuntimeError: If test seeds leak into training or the policy changes
                during evaluation.
        """
        # 0. Fail fast on invalid scenario environments before any training.
        # Preflight constructs, resets, and steps every scenario environment.
        self.spec.validate_environments(max_steps=config.environment.max_steps)

        train_scenario = self.spec.train_scenario
        train_params = self._scenario_env_params(config, train_scenario)

        # 1. TRAIN environment only: base parameters + TRAIN overrides.
        raw_train_env = make_env(config.environment.name, **train_params)
        train_env = TrainingDistributionWrapper(
            env=raw_train_env,
            seeds=train_scenario.seeds,
            shuffle=True,
            rng_seed=config.seed,
        )

        # 2. Train exclusively on TRAIN conditions.
        algo_name = config.algorithm.name.lower()
        trainer: PPOTrainer | SACTrainer
        if algo_name == "ppo":
            trainer = PPOTrainer(config=config, env=train_env)
        elif algo_name == "sac":
            trainer = SACTrainer(config=config, env=train_env)
        else:
            train_env.close()
            raise ValueError(f"Unsupported algorithm for distribution-shift runner: {algo_name}")
        trainer.fit()

        # 3. Seed-containment audit: training must never observe test seeds.
        sampled = list(train_env.sampled_seeds_history)
        sampled_set = set(sampled)
        train_seed_set = set(train_scenario.seeds)
        test_seed_set = set(self.spec.all_test_seeds)
        if not sampled_set.issubset(train_seed_set):
            train_env.close()
            raise RuntimeError(
                "Training seed leak: sampled seeds outside the TRAIN seed set: "
                f"{sorted(sampled_set - train_seed_set)[:5]}"
            )
        if sampled_set.intersection(test_seed_set):
            train_env.close()
            raise RuntimeError(
                "Training/test contamination: training observed test seeds: "
                f"{sorted(sampled_set.intersection(test_seed_set))[:5]}"
            )

        # 4. Freeze: fingerprint the trained policy before any evaluation.
        frozen_fingerprint = policy_fingerprint(trainer.algorithm)

        # 5. Evaluate the same frozen policy on every scenario independently.
        scenario_results: List[ScenarioResult] = []
        train_metrics: Optional[EvaluationMetrics] = None
        eval_envs: List[Any] = []
        try:
            for scenario in self.spec.scenarios:
                params = self._scenario_env_params(config, scenario)
                eval_env = make_env(config.environment.name, **params)
                eval_envs.append(eval_env)
                evaluator = GeneralizationEvaluator(
                    algorithm=trainer.algorithm,
                    env=eval_env,
                    env_name=config.environment.name,
                    env_kwargs=params,
                )
                metrics = evaluator.evaluate_seeds(
                    scenario.seeds,
                    deterministic=config.evaluation.deterministic,
                )
                records = self._episode_records(scenario, evaluator.last_episode_metrics)

                current_fingerprint = policy_fingerprint(trainer.algorithm)
                if current_fingerprint != frozen_fingerprint:
                    raise RuntimeError(
                        f"Policy changed during evaluation of scenario '{scenario.name}': "
                        "the benchmark requires a single frozen policy"
                    )

                effective = params
                get_effective = getattr(eval_env, "get_effective_parameters", None)
                if callable(get_effective):
                    try:
                        effective = dict(get_effective())
                    except Exception:
                        effective = params

                if scenario.role == "train":
                    train_metrics = metrics
                    gaps = None
                else:
                    assert train_metrics is not None
                    gaps = compute_scenario_gaps(train_metrics, metrics)

                scenario_results.append(
                    ScenarioResult(
                        scenario_name=scenario.name,
                        role=scenario.role,
                        description=scenario.description,
                        seeds=[int(s) for s in scenario.seeds],
                        environment_overrides=dict(scenario.environment_overrides),
                        effective_environment_parameters={str(k): v for k, v in effective.items()},
                        metrics=metrics,
                        recovery=RecoverySummary.from_metrics(metrics),
                        episodes=records,
                        gaps=gaps,
                        policy_fingerprint=current_fingerprint,
                    )
                )
        finally:
            train_env.close()
            for eval_env in eval_envs:
                eval_env.close()

        assert train_metrics is not None
        total_timesteps = config.training.total_timesteps if config.training is not None else 0
        report = ShiftBenchmarkReport(
            schema_version=SHIFT_BENCHMARK_SCHEMA_VERSION,
            experiment_name=config.name,
            environment_name=config.environment.name,
            algorithm_name=config.algorithm.name,
            deterministic=config.evaluation.deterministic,
            total_training_timesteps=total_timesteps,
            train_seeds=self.spec.train_seeds,
            test_seeds=self.spec.all_test_seeds,
            scenarios=scenario_results,
            training_provenance={
                "total_training_timesteps": total_timesteps,
                "algorithm": config.algorithm.name,
                "algorithm_parameters": dict(config.algorithm.parameters),
                "seed": config.seed,
                "training_seed_set": self.spec.train_seeds,
                "unique_training_seeds_sampled": len(sampled_set),
                "sampled_training_seeds": sampled,
                "test_seed_contamination": False,
                "policy_frozen": True,
                "policy_fingerprint": frozen_fingerprint,
                "training_environment_parameters": {str(k): v for k, v in train_params.items()},
            },
            environment_provenance=collect_environment_provenance(),
            config_sha256=compute_config_sha256(config),
            metadata={
                "benchmark_name": self.spec.name,
                "benchmark_version": self.spec.version,
            },
        )

        report_path = config.output_dir / "distribution_shift" / f"{config.name}_shift_report.json"
        report.save_json(report_path)

        self.last_report = report
        return report
