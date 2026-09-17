"""Reproducible experiment orchestration manager for AdaptiveRL.

Provides ExperimentManager: a structured orchestration layer that records
provenance metadata, manages output directories, runs training/planning,
evaluates performance, and serializes results to a machine-readable manifest.
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import yaml
from pydantic import BaseModel

from adaptive_rl.algorithms import (
    AlgorithmKind,
    AlgorithmRegistryError,
    algorithm_registry,
)
from adaptive_rl.config import ExperimentConfig, load_config
from adaptive_rl.experiments.artifacts import (
    _make_experiment_id,
    _make_run_id,
    _sanitize_path_component,
    make_experiment_id,
    make_run_id,
    resolve_run_directory,
    sanitize_path_component,
    save_config_yaml,
    save_metrics_csv,
    save_metrics_json,
)
from adaptive_rl.experiments.manifest import ExperimentManifest, ExperimentResult
from adaptive_rl.experiments.provenance import (
    _get_git_commit,
    _get_git_provenance,
    _get_package_version,
    _get_platform_info,
    collect_environment_provenance,
    get_git_commit,
    get_git_provenance,
    get_package_version,
    get_platform_info,
)

# ---------------------------------------------------------------------------
# Re-exported symbols for backward compatibility
# ---------------------------------------------------------------------------
__all__ = [
    "ExperimentManager",
    "ExperimentManifest",
    "ExperimentResult",
    "_get_git_commit",
    "_get_git_provenance",
    "_get_package_version",
    "_get_platform_info",
    "get_git_commit",
    "get_git_provenance",
    "get_package_version",
    "get_platform_info",
    "collect_environment_provenance",
    "_sanitize_path_component",
    "_make_experiment_id",
    "_make_run_id",
    "sanitize_path_component",
    "make_experiment_id",
    "make_run_id",
    "resolve_run_directory",
    "save_config_yaml",
    "save_metrics_csv",
    "save_metrics_json",
    "_apply_config_override",
    "_apply_overrides",
]


# Re-exported from adaptive_rl.experiments.artifacts for backward compatibility
# Single authoritative ownership resides in artifacts.py


def _apply_config_override(config: ExperimentConfig, key: str, value: Any) -> None:
    """Apply an individual configuration override to an ExperimentConfig instance.

    Supports dot-notated paths (e.g. 'training.total_timesteps', 'algorithm.learning_rate',
    'environment.parameters.width', 'evaluation.eval_episodes') as well as top-level and
    convenience field names (e.g. 'seed', 'total_timesteps', 'learning_rate').
    """
    if "." in key:
        parts = key.split(".")
        target: Any = config
        for part in parts[:-1]:
            if isinstance(target, BaseModel):
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    raise ValueError(f"Unknown config section '{part}' in override '{key}'")
            elif isinstance(target, dict):
                if part not in target:
                    target[part] = {}
                target = target[part]
            else:
                raise ValueError(
                    f"Cannot traverse into '{type(target).__name__}' for override '{key}'"
                )

        last_part = parts[-1]
        if isinstance(target, BaseModel):
            if hasattr(target, last_part):
                field_info = type(target).model_fields.get(last_part)
                if field_info is not None and field_info.annotation is not None:
                    target_type = field_info.annotation
                    if target_type is bool and isinstance(value, str):
                        val_lower = value.strip().lower()
                        if val_lower in ("true", "1", "yes", "on"):
                            value = True
                        elif val_lower in ("false", "0", "no", "off"):
                            value = False
                    elif target_type in (int, float, str) and not isinstance(value, target_type):
                        try:
                            value = target_type(value)
                        except (ValueError, TypeError):
                            pass
                setattr(target, last_part, value)
            else:
                raise ValueError(f"Unknown field '{last_part}' in override '{key}'")
        elif isinstance(target, dict):
            target[last_part] = value
        else:
            raise ValueError(f"Cannot set field on '{type(target).__name__}' for override '{key}'")
    else:
        # Top-level or convenience aliases
        if hasattr(config, key):
            field_info = type(config).model_fields.get(key)
            if field_info is not None and field_info.annotation is not None:
                target_type = field_info.annotation
                if target_type is bool and isinstance(value, str):
                    val_lower = value.strip().lower()
                    if val_lower in ("true", "1", "yes", "on"):
                        value = True
                    elif val_lower in ("false", "0", "no", "off"):
                        value = False
                elif target_type in (int, float, str) and not isinstance(value, target_type):
                    try:
                        value = target_type(value)
                    except (ValueError, TypeError):
                        pass
            setattr(config, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.algorithm, key):
            setattr(config.algorithm, key, value)
        elif hasattr(config.environment, key):
            setattr(config.environment, key, value)
        elif hasattr(config.evaluation, key):
            setattr(config.evaluation, key, value)
        else:
            raise ValueError(f"Unknown configuration override key: '{key}'")


def _apply_overrides(config: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
    """Apply all overrides to a deep copy of config, returning the modified effective config."""
    effective = config.model_copy(deep=True)
    for k, v in overrides.items():
        _apply_config_override(effective, k, v)
    return ExperimentConfig.model_validate(effective.model_dump())


def _resolve_planner_policy(factory: Any, env: Any, kwargs: Optional[Dict[str, Any]] = None) -> Any:
    """Resolve a PlannerPolicy adapter for the given planner factory and environment."""
    from adaptive_rl.planning.base import PlannerPolicy

    params = dict(kwargs or {})
    unwrapped_env = getattr(env, "unwrapped", env)

    # 1. If factory itself is a PlannerPolicy subclass
    if isinstance(factory, type) and issubclass(factory, PlannerPolicy):
        try:
            return cast(Any, factory)(env=env, **params)
        except TypeError:
            return cast(Any, factory)(**params)

    # 2. If factory is an instance that already implements PlannerPolicy or predict()
    if isinstance(factory, PlannerPolicy) or (
        not isinstance(factory, type) and hasattr(factory, "predict")
    ):
        return factory

    factory_name = getattr(factory, "__name__", str(factory))

    # 3. If factory or instance provides a policy conversion method
    if hasattr(factory, "as_policy"):
        return factory.as_policy(env=env, **params)
    if hasattr(factory, "to_policy"):
        return factory.to_policy(env=env, **params)

    # 4. A* Planner policy adapter
    if hasattr(factory, "from_gridworld") or factory_name == "AStarPlanner":
        from adaptive_rl.planning.astar import AStarPlanner, AStarPlannerPolicy

        planner = None
        if isinstance(factory, type) and issubclass(factory, AStarPlanner):
            if hasattr(factory, "from_gridworld"):
                # from_gridworld uses getattr with defaults for missing env attributes
                planner = factory.from_gridworld(unwrapped_env)
        if planner is None and params:
            try:
                planner = factory(**params)
            except TypeError:
                # Factory requires more args than params provides; fall back to policy wrapper
                pass
        return AStarPlannerPolicy(planner=planner, env=env)

    # 5. RRT / RRT* Planner policy adapter
    if hasattr(factory, "from_navigation_env") or "RRT" in factory_name:
        from adaptive_rl.planning.rrt import RRTPlanner, RRTPlannerPolicy, RRTStarPlanner

        planner = None
        if isinstance(factory, type) and issubclass(factory, (RRTPlanner, RRTStarPlanner)):
            if hasattr(factory, "from_navigation_env"):
                planner = factory.from_navigation_env(unwrapped_env, **params)
        if planner is None and params:
            try:
                planner = factory(**params)
            except TypeError:
                pass
        return RRTPlannerPolicy(planner=planner, env=env)

    # 6. Generic planner instance/factory
    planner = factory(**params) if isinstance(factory, type) else factory
    if hasattr(planner, "predict"):
        return planner
    if hasattr(planner, "as_policy"):
        return planner.as_policy(env=env)

    raise ValueError(f"No policy adapter found for planner '{factory_name}'")


# ---------------------------------------------------------------------------
# ExperimentManager
# ---------------------------------------------------------------------------


class ExperimentManager:
    """Orchestrates reproducible end-to-end AdaptiveRL experiments.

    Responsibilities:
    - Load and validate configuration.
    - Track source_config, runtime overrides, and effective_config.
    - Generate deterministic experiment ID and unique run ID.
    - Record honest provenance metadata (full git SHA, branch, dirty status, error).
    - Create safe, collision-resistant run output directories.
    - Resolve algorithms strictly via AlgorithmRegistry (PR #82).
    - Classify execution mechanisms via registry metadata (RL policy vs planner).
    - Launch training (RL) or planning (classical motion planners).
    - Evaluate performance and save machine-readable artifacts.
    - Return a structured :class:`ExperimentResult`.

    Output directory structure::

        experiments/results/<experiment_id>/<run_id>/
        ├── config.yaml          — copy of effective config used
        ├── source_config.yaml   — copy of original source config
        ├── manifest.json        — provenance, configs, and artifact paths
        ├── metrics.json         — evaluation metrics
        ├── metrics.csv          — tabular metrics
        ├── evaluation.json      — full evaluation report
        ├── model/               — saved model weights (RL only)
        └── logs/                — training logs (RL only)
    """

    def __init__(
        self,
        base_output_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the experiment manager.

        Args:
            base_output_dir: Root directory for all experiment artifacts.
                Defaults to ``experiments/results`` relative to cwd.
        """
        self.base_output_dir = base_output_dir or Path("experiments/results")

    def run_from_config(
        self,
        config_path: Union[str, Path],
        timesteps_override: Optional[int] = None,
        seed_override: Optional[int] = None,
        **extra_overrides: Any,
    ) -> ExperimentResult:
        """Run a complete experiment from a YAML configuration file.

        Preserves source configuration and tracks explicit runtime overrides.

        Args:
            config_path: Path to the experiment YAML configuration.
            timesteps_override: Override training timesteps from config.
            seed_override: Override random seed from config.
            **extra_overrides: Additional runtime configuration overrides.

        Returns:
            ExperimentResult containing all metadata, metrics, and artifacts.
        """
        config_p = Path(config_path)
        try:
            source_config = load_config(config_p)
        except Exception as exc:
            exp_id = f"failed_config_{uuid.uuid4().hex[:8]}"
            run_id = _make_run_id()
            err_dir = self.base_output_dir / exp_id / run_id
            err_dir.mkdir(parents=True, exist_ok=True)
            manifest = ExperimentManifest(
                experiment_id=exp_id,
                run_id=run_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                algorithm="unknown",
                environment="unknown",
                seed=-1,
                config_path=str(config_path),
                source_config={},
                overrides=extra_overrides,
                effective_config={},
                evaluation_status="failed",
                failure_type=type(exc).__name__,
                failure_message=str(exc),
                failure_traceback=traceback.format_exc(),
                notes=f"Config loading failed: {exc}",
            )
            manifest.save(err_dir / "manifest.json")
            return ExperimentResult(
                experiment_id=exp_id,
                run_id=run_id,
                output_dir=err_dir,
                manifest=manifest,
                success=False,
                error_message=f"Config loading failed: {exc}",
            )

        overrides: Dict[str, Any] = {}
        if "overrides" in extra_overrides and isinstance(extra_overrides["overrides"], dict):
            overrides.update(extra_overrides.pop("overrides"))
        overrides.update(extra_overrides)
        if timesteps_override is not None:
            overrides["training.total_timesteps"] = timesteps_override
        if seed_override is not None:
            overrides["seed"] = seed_override

        return self.run(
            config=source_config,
            config_path=config_p,
            overrides=overrides,
            source_config=source_config,
        )

    def run(
        self,
        config: ExperimentConfig,
        config_path: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
        source_config: Optional[Union[ExperimentConfig, Dict[str, Any]]] = None,
    ) -> ExperimentResult:
        """Run a complete experiment from an ExperimentConfig.

        Args:
            config: Initial experiment configuration.
            config_path: Optional original config file path (for manifest).
            overrides: Optional dictionary of runtime overrides applied.
            source_config: Optional source configuration before overrides.

        Returns:
            ExperimentResult containing all metadata, metrics, and artifacts.
        """
        # Ensure working with an independent copy for source_config
        if source_config is None:
            source_config = config.model_copy(deep=True)

        overrides_dict = dict(overrides or {})

        # Apply all overrides to produce the authoritative effective_config
        try:
            effective_config = _apply_overrides(config, overrides_dict)
        except Exception as exc:
            exp_id = f"failed_override_{uuid.uuid4().hex[:8]}"
            run_id = _make_run_id()
            err_dir = self.base_output_dir / exp_id / run_id
            err_dir.mkdir(parents=True, exist_ok=True)
            manifest = ExperimentManifest(
                experiment_id=exp_id,
                run_id=run_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                algorithm=getattr(getattr(config, "algorithm", None), "name", "unknown"),
                environment=getattr(getattr(config, "environment", None), "name", "unknown"),
                seed=getattr(config, "seed", -1),
                config_path=str(config_path or ""),
                source_config=source_config.model_dump(mode="json")
                if isinstance(source_config, ExperimentConfig)
                else dict(source_config),
                overrides=overrides_dict,
                effective_config={},
                training_timesteps=None,
                **collect_environment_provenance(),
                evaluation_status="failed",
                notes=f"Configuration override failed: {exc}",
            )
            manifest.save(err_dir / "manifest.json")
            return ExperimentResult(
                experiment_id=exp_id,
                run_id=run_id,
                output_dir=err_dir,
                manifest=manifest,
                success=False,
                error_message=f"Configuration override failed: {exc}",
            )

        experiment_id = _make_experiment_id(effective_config)
        base_resolved = self.base_output_dir.resolve()

        # Atomic and race-safe run directory creation with path manipulation guards
        max_attempts = 10
        output_dir: Optional[Path] = None
        run_id = ""
        for attempt in range(max_attempts):
            run_id = _make_run_id()
            candidate_dir = (self.base_output_dir / experiment_id / run_id).resolve()
            if not candidate_dir.is_relative_to(base_resolved):
                raise ValueError(
                    f"Path traversal detected: '{candidate_dir}' escapes base directory '{base_resolved}'."
                )
            try:
                candidate_dir.mkdir(parents=True, exist_ok=False)
                output_dir = candidate_dir
                break
            except FileExistsError:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Failed to create unique run directory after {max_attempts} attempts."
                    )
                time.sleep(0.01)

        assert output_dir is not None

        # Save config copies
        effective_config_path = output_dir / "config.yaml"
        self._save_config_copy(effective_config, effective_config_path)

        source_config_path = output_dir / "source_config.yaml"
        if isinstance(source_config, ExperimentConfig):
            self._save_config_copy(source_config, source_config_path)
        else:
            with open(source_config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(source_config, f, sort_keys=False, default_flow_style=False)

        # Build base manifest
        source_dump = (
            source_config.model_dump(mode="json")
            if isinstance(source_config, ExperimentConfig)
            else dict(source_config)
        )
        manifest = ExperimentManifest(
            experiment_id=experiment_id,
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            algorithm=effective_config.algorithm.name,
            environment=effective_config.environment.name,
            seed=effective_config.seed,
            config_path=str(config_path or effective_config_path),
            source_config=source_dump,
            overrides=overrides_dict,
            effective_config=effective_config.model_dump(mode="json"),
            training_timesteps=(
                effective_config.training.total_timesteps
                if effective_config.training is not None
                else None
            ),
            **collect_environment_provenance(),
        )
        # Store relative artifact paths for complete filesystem portability
        manifest.artifact_paths["config"] = "config.yaml"
        manifest.artifact_paths["source_config"] = "source_config.yaml"

        # Algorithm resolution strictly via AlgorithmRegistry (PR #82)
        algo_name = effective_config.algorithm.name.strip().lower()
        try:
            metadata = algorithm_registry.get_metadata(algo_name)
            factory = algorithm_registry.get_factory(algo_name)
        except AlgorithmRegistryError as err:
            manifest.evaluation_status = "failed"
            manifest.notes = f"Algorithm resolution failed: {err}"
            manifest_path = output_dir / "manifest.json"
            manifest.save(manifest_path)
            return ExperimentResult(
                experiment_id=experiment_id,
                run_id=run_id,
                output_dir=output_dir,
                manifest=manifest,
                success=False,
                error_message=f"Algorithm resolution failed: {err}",
            )

        # Classification driven entirely by registry metadata (no hard-coded name lists)
        if metadata.kind == AlgorithmKind.PLANNER:
            result = self._run_planner_experiment(effective_config, factory, output_dir, manifest)
        elif metadata.kind == AlgorithmKind.RL_POLICY:
            result = self._run_rl_experiment(effective_config, factory, output_dir, manifest)
        else:
            manifest.evaluation_status = "failed"
            manifest.notes = f"Unsupported algorithm kind: '{metadata.kind}'"
            manifest.save(output_dir / "manifest.json")
            return ExperimentResult(
                experiment_id=experiment_id,
                run_id=run_id,
                output_dir=output_dir,
                manifest=manifest,
                success=False,
                error_message=f"Unsupported algorithm kind: '{metadata.kind}'",
            )

        return result

    # ------------------------------------------------------------------
    # RL experiment
    # ------------------------------------------------------------------

    def _run_rl_experiment(
        self,
        config: ExperimentConfig,
        factory: Any,
        output_dir: Path,
        manifest: ExperimentManifest,
    ) -> ExperimentResult:
        """Run training + evaluation for an RL policy."""
        from adaptive_rl.environments.registry import make_env
        from adaptive_rl.evaluation.evaluator import Evaluator
        from adaptive_rl.training import get_trainer

        # Create model and logs directories only for RL
        (output_dir / "model").mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)

        config.output_dir = output_dir
        config.log_dir = output_dir / "logs"

        training_result = None
        metrics: Dict[str, Any] = {}
        error_msg = ""
        success = True

        try:
            trainer = get_trainer(config=config)
            training_result = trainer.fit()

            model_path = training_result.final_model_path
            if model_path is not None and Path(model_path).exists():
                try:
                    manifest.artifact_paths["model"] = (
                        Path(model_path).relative_to(output_dir).as_posix()
                    )
                except ValueError:
                    manifest.artifact_paths["model"] = Path(model_path).name
            manifest.training_timesteps = training_result.total_timesteps

            # Evaluation
            env = make_env(config.environment.name, **config.environment.parameters)
            if hasattr(factory, "from_pretrained"):
                algo = factory.from_pretrained(model_path, env=env)
            else:
                algo = factory(env=env)
                algo.load(model_path, env=env)

            evaluator = Evaluator(algorithm=algo, env=env)
            eval_metrics = evaluator.evaluate(
                num_episodes=config.evaluation.eval_episodes,
                deterministic=config.evaluation.deterministic,
                base_seed=config.seed + 10000,
            )

            from adaptive_rl.evaluation.metrics import StandardizedExperimentMetrics

            std_metrics = StandardizedExperimentMetrics.from_rl_metrics(eval_metrics)
            metrics = std_metrics.model_dump()
            manifest.evaluation_status = "completed"

            # Save metrics
            metrics_path = output_dir / "metrics.json"
            save_metrics_json(metrics, metrics_path)
            manifest.artifact_paths["metrics"] = "metrics.json"

            # Save CSV
            csv_path = output_dir / "metrics.csv"
            save_metrics_csv(std_metrics.to_csv_dict(), csv_path)
            manifest.artifact_paths["metrics_csv"] = "metrics.csv"

            # Full evaluation report
            eval_path = output_dir / "evaluation.json"
            evaluator.save_report(eval_metrics, eval_path)
            manifest.artifact_paths["evaluation"] = "evaluation.json"

            env.close()

        except KeyboardInterrupt:
            manifest.evaluation_status = "interrupted"
            manifest.failure_type = "KeyboardInterrupt"
            manifest.failure_message = "Execution interrupted by user"
            manifest.failure_traceback = traceback.format_exc()
            manifest.notes = "Execution interrupted by user (KeyboardInterrupt)."
            manifest_path = output_dir / "manifest.json"
            manifest.save(manifest_path)
            raise
        except Exception as exc:
            success = False
            error_msg = str(exc)
            manifest.evaluation_status = "failed"
            manifest.failure_type = type(exc).__name__
            manifest.failure_message = str(exc)
            manifest.failure_traceback = traceback.format_exc()
            manifest.notes = f"RL execution error: {exc}"

        manifest_path = output_dir / "manifest.json"
        manifest.save(manifest_path)

        return ExperimentResult(
            experiment_id=manifest.experiment_id,
            run_id=manifest.run_id,
            output_dir=output_dir,
            manifest=manifest,
            metrics=metrics,
            training_result=training_result,
            success=success,
            error_message=error_msg,
        )

    # ------------------------------------------------------------------
    # Planner experiment
    # ------------------------------------------------------------------

    def _run_planner_experiment(
        self,
        config: ExperimentConfig,
        factory: Any,
        output_dir: Path,
        manifest: ExperimentManifest,
    ) -> ExperimentResult:
        """Run evaluation for a classical motion planner (no training phase)."""
        from adaptive_rl.environments.registry import make_env
        from adaptive_rl.evaluation.evaluator import Evaluator

        # Classical planners do NOT generate model weights or training logs
        metrics: Dict[str, Any] = {}
        error_msg = ""
        success = True

        try:
            env = make_env(config.environment.name, **config.environment.parameters)
            policy = _resolve_planner_policy(factory, env, config.algorithm.parameters)

            evaluator = Evaluator(algorithm=policy, env=env)
            eval_metrics = evaluator.evaluate(
                num_episodes=config.evaluation.eval_episodes,
                deterministic=config.evaluation.deterministic,
                base_seed=config.seed,
            )

            # Record evaluation seeds used for this run
            from adaptive_rl.evaluation.seeding import generate_evaluation_seeds

            manifest.evaluation_seeds = generate_evaluation_seeds(
                config.seed, config.evaluation.eval_episodes
            )

            from adaptive_rl.evaluation.metrics import StandardizedExperimentMetrics

            std_metrics = StandardizedExperimentMetrics.from_rl_metrics(eval_metrics)
            std_metrics.episode_return = None  # Classical planners do not produce RL reward returns
            std_metrics.episode_length = None  # RL step count is not applicable to planner paths
            metrics = std_metrics.model_dump()
            manifest.evaluation_status = "completed"
            manifest.training_timesteps = None

            # Save metrics
            metrics_path = output_dir / "metrics.json"
            save_metrics_json(metrics, metrics_path)
            manifest.artifact_paths["metrics"] = "metrics.json"

            # Save CSV
            csv_path = output_dir / "metrics.csv"
            save_metrics_csv(std_metrics.to_csv_dict(), csv_path)
            manifest.artifact_paths["metrics_csv"] = "metrics.csv"

            # Full evaluation report
            eval_path = output_dir / "evaluation.json"
            evaluator.save_report(eval_metrics, eval_path)
            manifest.artifact_paths["evaluation"] = "evaluation.json"

            env.close()

        except KeyboardInterrupt:
            manifest.evaluation_status = "interrupted"
            manifest.failure_type = "KeyboardInterrupt"
            manifest.failure_message = "Execution interrupted by user"
            manifest.failure_traceback = traceback.format_exc()
            manifest.notes = "Execution interrupted by user (KeyboardInterrupt)."
            manifest_path = output_dir / "manifest.json"
            manifest.save(manifest_path)
            raise
        except Exception as exc:
            success = False
            error_msg = str(exc)
            manifest.evaluation_status = "failed"
            manifest.failure_type = type(exc).__name__
            manifest.failure_message = str(exc)
            manifest.failure_traceback = traceback.format_exc()
            manifest.notes = f"Planner execution error: {exc}"

        manifest_path = output_dir / "manifest.json"
        manifest.save(manifest_path)

        return ExperimentResult(
            experiment_id=manifest.experiment_id,
            run_id=manifest.run_id,
            output_dir=output_dir,
            manifest=manifest,
            metrics=metrics,
            training_result=None,
            success=success,
            error_message=error_msg,
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _save_config_copy(config: ExperimentConfig, path: Path) -> None:
        """Save a copy of the experiment config as YAML."""
        save_config_yaml(config, path)

    @staticmethod
    def _save_metrics_csv(metrics: Dict[str, Any], path: Path) -> Path:
        """Save flat scalar metrics as a single-row CSV file (delegates to artifacts.py)."""
        return save_metrics_csv(metrics, path)

    # ------------------------------------------------------------------
    # Experiment listing / inspection utilities
    # ------------------------------------------------------------------

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all completed experiment runs in the base output directory.

        Returns:
            List of manifest dictionaries, sorted by creation time descending.
            Empty list if no experiments have been run.
        """
        experiments: List[Dict[str, Any]] = []
        if not self.base_output_dir.exists():
            return experiments

        for manifest_path in self.base_output_dir.rglob("manifest.json"):
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    data = json.load(f)
                experiments.append(data)
            except Exception:
                experiments.append(
                    {
                        "experiment_id": manifest_path.parent.parent.name,
                        "run_id": manifest_path.parent.name,
                        "error": "manifest unreadable",
                    }
                )

        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return experiments

    def get_experiment(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Load and return the manifest for a specific run_id or experiment_id.

        Args:
            identifier: run_id or experiment_id string.

        Returns:
            Manifest dictionary, or None if not found.
        """
        if not self.base_output_dir.exists():
            return None

        # Direct path check
        direct_manifest = self.base_output_dir / identifier / "manifest.json"
        if direct_manifest.exists():
            try:
                with open(direct_manifest, encoding="utf-8") as f:
                    return cast(Optional[Dict[str, Any]], json.load(f))
            except Exception:
                return None

        # Search by run_id or experiment_id
        candidates: List[Path] = []
        for manifest_path in self.base_output_dir.rglob("manifest.json"):
            if (
                manifest_path.parent.name == identifier
                or manifest_path.parent.parent.name == identifier
            ):
                candidates.append(manifest_path)

        if not candidates:
            return None

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        try:
            with open(candidates[0], encoding="utf-8") as f:
                return cast(Optional[Dict[str, Any]], json.load(f))
        except Exception:
            return None

    def get_metrics(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Load and return metrics for a specific run_id or experiment_id.

        Args:
            identifier: run_id or experiment_id string.

        Returns:
            Metrics dictionary, or None if metrics file not found.
        """
        manifest_data = self.get_experiment(identifier)
        if not manifest_data:
            return None
        metrics_path_str = manifest_data.get("artifact_paths", {}).get("metrics")
        if metrics_path_str:
            metrics_path = Path(metrics_path_str)
            if not metrics_path.is_absolute():
                run_dir = (
                    self.base_output_dir
                    / manifest_data.get("experiment_id", "")
                    / manifest_data.get("run_id", "")
                )
                metrics_path = run_dir / metrics_path
            elif not metrics_path.exists():
                run_dir = (
                    self.base_output_dir
                    / manifest_data.get("experiment_id", "")
                    / manifest_data.get("run_id", "")
                )
                metrics_path = run_dir / metrics_path.name

            if metrics_path.exists():
                try:
                    with open(metrics_path, encoding="utf-8") as f:
                        return cast(Optional[Dict[str, Any]], json.load(f))
                except Exception:
                    return None
        return None
