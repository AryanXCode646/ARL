"""Artifact serialization and file management for AdaptiveRL experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

from adaptive_rl.config import ExperimentConfig


def sanitize_path_component(name: str) -> str:
    """Sanitize a string for safe usage in filesystem directory names.

    Replaces non-alphanumeric, non-hyphen, non-underscore characters with '_',
    collapses consecutive underscores, and strips leading/trailing periods,
    slashes, and underscores to prevent directory traversal attacks.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(name).strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
    return sanitized or "unknown"


def make_experiment_id(config: ExperimentConfig) -> str:
    """Generate a deterministic, filesystem-safe experiment identifier.

    Derived strictly from the canonical effective experiment definition (algorithm,
    environment, seed, training, evaluation, curriculum parameters). Excludes
    ephemeral runtime paths like output_dir and log_dir to maintain identity
    invariance regardless of execution workspace.
    """
    data = config.model_dump(mode="python")
    data.pop("output_dir", None)
    data.pop("log_dir", None)

    serialized = json.dumps(data, sort_keys=True, default=str)
    config_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]

    env = sanitize_path_component(config.environment.name.lower())
    algo = sanitize_path_component(config.algorithm.name.lower())
    seed = int(config.seed)
    return f"{env}_{algo}_seed{seed}_{config_hash}"


def make_run_id() -> str:
    """Generate a unique run execution identifier with UTC timestamp and random hex."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:8]
    return f"run_{ts}_{rand}"


def resolve_run_directory(base_output_dir: Path, experiment_id: str, run_id: str) -> Path:
    """Resolve and create isolated directory for an experiment run."""
    run_dir = (base_output_dir / experiment_id / run_id).resolve()
    base_resolved = base_output_dir.resolve()
    if not run_dir.is_relative_to(base_resolved):
        raise ValueError(
            f"Path traversal detected: '{run_dir}' escapes base directory '{base_resolved}'."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_unique_run_directory(
    base_output_dir: Path, experiment_id: str, max_attempts: int = 10
) -> tuple[str, Path]:
    """Atomically create a unique run directory under base_output_dir / experiment_id.

    Returns:
        tuple of (run_id, output_dir)

    Raises:
        ValueError: If path traversal attempts to escape base_output_dir.
        RuntimeError: If unique directory cannot be created within max_attempts.
    """
    base_resolved = base_output_dir.resolve()
    for attempt in range(max_attempts):
        run_id = make_run_id()
        candidate_dir = (base_output_dir / experiment_id / run_id).resolve()
        if not candidate_dir.is_relative_to(base_resolved):
            raise ValueError(
                f"Path traversal detected: '{candidate_dir}' escapes base directory '{base_resolved}'."
            )
        try:
            candidate_dir.mkdir(parents=True, exist_ok=False)
            return run_id, candidate_dir
        except FileExistsError:
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Failed to create unique run directory after {max_attempts} attempts."
                )
            time.sleep(0.01)
    raise RuntimeError(f"Failed to create unique run directory after {max_attempts} attempts.")


# Backward-compatible aliases matching internal naming
_sanitize_path_component = sanitize_path_component
_make_experiment_id = make_experiment_id
_make_run_id = make_run_id


def save_metrics_json(metrics: Dict[str, Any], path: Path) -> Path:
    """Save metrics dictionary to formatted JSON, preserving null and 0 values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path


def save_metrics_csv(metrics: Dict[str, Any], path: Path) -> Path:
    """Save flat scalar metrics as a single-row CSV file.

    Scalar metrics (int, float, str) are written as-is. Measured zero (0.0 or 0)
    is strictly preserved. None values are written as empty strings ("").
    Non-scalar collections (lists, dicts) are excluded from the flat CSV.
    """
    csv_metrics: Dict[str, Any] = {}
    for k, v in metrics.items():
        if v is None:
            csv_metrics[k] = ""
        elif isinstance(v, bool):
            csv_metrics[k] = int(v)
        elif isinstance(v, (int, float, str)):
            csv_metrics[k] = v
        # Exclude lists, tuples, dicts from flat scalar CSV

    path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_metrics:
        # Create empty file with empty row if no scalar metrics
        with open(path, "w", newline="", encoding="utf-8") as f:
            pass
        return path

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(csv_metrics.keys()))
        writer.writeheader()
        writer.writerow({k: csv_metrics[k] for k in sorted(csv_metrics.keys())})
    return path


def save_config_yaml(config: ExperimentConfig | Dict[str, Any], path: Path) -> Path:
    """Save experiment configuration to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(config, ExperimentConfig):
        data = config.model_dump(mode="python")
        data["output_dir"] = str(data["output_dir"])
        data["log_dir"] = str(data["log_dir"])
    elif isinstance(config, dict):
        data = dict(config)
    else:
        data = {"config": str(config)}

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    return path
