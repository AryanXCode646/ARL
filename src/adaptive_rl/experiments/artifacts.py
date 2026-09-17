"""Artifact serialization and file management for AdaptiveRL experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from adaptive_rl.config import ExperimentConfig


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
