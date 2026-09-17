"""Experiment manifest and result data structures for AdaptiveRL."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentManifest:
    """Machine-readable experiment provenance and execution record.

    Written to ``<output_dir>/manifest.json`` on experiment completion or failure.

    Attributes:
        experiment_id: Unique deterministic experiment identifier.
        run_id: Unique identifier for this specific execution run.
        created_at: ISO 8601 UTC timestamp of experiment run creation.
        algorithm: Algorithm name used in this experiment.
        environment: Environment name used in this experiment.
        seed: Random seed.
        config_path: Relative or absolute path to the source YAML config.
        source_config: Original input configuration before runtime overrides.
        overrides: Explicit runtime overrides applied for this run.
        effective_config: The exact effective configuration used for execution.
        training_timesteps: Total training timesteps (None for classical planners).
        git_commit: Full git commit hash at experiment time, if available.
        git_branch: Active git branch name, if available.
        git_dirty: Whether the git working directory had uncommitted changes.
        git_error: Error message if git provenance could not be collected.
        python_version: Python version string.
        platform_info: OS and CPU architecture information.
        package_versions: Key package versions (adaptive-rl, gymnasium, torch, etc.).
        artifact_paths: Dictionary mapping artifact names to their relative filesystem paths.
        evaluation_status: 'completed', 'failed', 'interrupted', or 'pending'.
        evaluation_seeds: List of exact seeds used during evaluation.
        evaluation_seed_strategy: Description of seed generation strategy.
        failure_type: Exception type name if the run failed.
        failure_message: Error description message if the run failed.
        failure_traceback: Formatted traceback string if the run failed.
        notes: Optional free-text notes or failure diagnostic details.
    """

    experiment_id: str
    run_id: str
    created_at: str
    algorithm: str
    environment: str
    seed: int
    config_path: str
    source_config: Dict[str, Any]
    overrides: Dict[str, Any]
    effective_config: Dict[str, Any]
    training_timesteps: Optional[int] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None
    git_error: Optional[str] = None
    python_version: str = ""
    platform_info: str = ""
    package_versions: Dict[str, str] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    evaluation_status: str = "pending"
    evaluation_seeds: List[int] = field(default_factory=list)
    evaluation_seed_strategy: str = "deterministic_derived"
    failure_type: Optional[str] = None
    failure_message: Optional[str] = None
    failure_traceback: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to a plain dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write manifest as formatted JSON.

        Args:
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentManifest:
        """Load manifest instance from dictionary, ignoring unexpected fields."""
        known_fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class ExperimentResult:
    """Structured result returned by :class:`ExperimentManager`.

    Attributes:
        experiment_id: Deterministic experiment identifier.
        run_id: Unique execution identifier for this run.
        output_dir: Root directory containing all run artifacts.
        manifest: Experiment provenance manifest.
        metrics: Standardized evaluation metrics dictionary (JSON-serializable).
        training_result: TrainingResult dataclass (None for planners).
        success: Whether the experiment completed without fatal errors.
        error_message: Error description if success is False.
    """

    experiment_id: str
    run_id: str
    output_dir: Path
    manifest: ExperimentManifest
    metrics: Dict[str, Any] = field(default_factory=dict)
    training_result: Any = None
    success: bool = True
    error_message: str = ""
