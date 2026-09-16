"""Configuration system and schemas for AdaptiveRL experiments.

Provides schema validation, YAML loading, and deterministic configuration
management for environments, algorithms, training, and evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ConfigError(Exception):
    """Exception raised for configuration parsing or validation failures."""

    pass


class AlgorithmConfig(BaseModel):
    """Configuration parameters for the reinforcement learning algorithm."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Algorithm name, e.g. 'ppo' or 'sac'")
    learning_rate: float = Field(3e-4, gt=0.0, description="Optimizer learning rate")
    gamma: float = Field(0.99, ge=0.0, le=1.0, description="Discount factor")
    batch_size: int = Field(64, gt=0, description="Minibatch size")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional algorithm-specific hyperparameters"
    )


class EnvironmentConfig(BaseModel):
    """Configuration parameters for the Gymnasium environment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Registered environment name, e.g. 'gridworld'")
    max_steps: int = Field(100, gt=0, description="Maximum steps per episode")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment-specific parameters (e.g. grid size, obstacle count)",
    )


class TrainingConfig(BaseModel):
    """Configuration parameters for the training loop."""

    model_config = ConfigDict(extra="forbid")

    total_timesteps: int = Field(10000, gt=0, description="Total environment steps to train")
    checkpoint_freq: int = Field(
        2000, ge=0, description="Frequency of saving model checkpoints (0 = disabled)"
    )
    log_interval: int = Field(10, gt=0, description="Frequency of logging metrics")


class EvaluationConfig(BaseModel):
    """Configuration parameters for evaluation and benchmarking."""

    model_config = ConfigDict(extra="forbid")

    eval_episodes: int = Field(10, gt=0, description="Number of evaluation episodes")
    deterministic: bool = Field(
        True, description="Whether to use deterministic actions in evaluation"
    )


class ExperimentConfig(BaseModel):
    """Top-level configuration schema for an AdaptiveRL experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique experiment identifier")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    algorithm: AlgorithmConfig
    environment: EnvironmentConfig
    training: TrainingConfig
    evaluation: EvaluationConfig = Field(
        default_factory=lambda: EvaluationConfig(eval_episodes=10, deterministic=True)
    )
    output_dir: Path = Field(
        default_factory=lambda: Path("experiments/results"),
        description="Directory for saving models and evaluations",
    )
    log_dir: Path = Field(
        default_factory=lambda: Path("experiments/logs"),
        description="Directory for logging and tensorboard metrics",
    )


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate an AdaptiveRL experiment configuration from a YAML file.

    Args:
        config_path: Filepath to the YAML configuration file.

    Returns:
        ExperimentConfig: Validated typed configuration instance.

    Raises:
        ConfigError: If the file is missing, contains invalid YAML, or fails schema validation.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML file at {path}: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigError(
            f"Configuration file {path} must contain a YAML mapping/dictionary, got {type(raw_data).__name__}"
        )

    try:
        return ExperimentConfig.model_validate(raw_data)
    except ValidationError as exc:
        formatted_errors = []
        for err in exc.errors():
            loc = " -> ".join(str(p) for p in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            formatted_errors.append(f"  - [{loc}]: {msg}")
        errors_str = "\n".join(formatted_errors)
        raise ConfigError(f"Configuration validation failed for {path}:\n{errors_str}") from exc


def save_config(config: ExperimentConfig, target_path: str | Path) -> None:
    """Save an experiment configuration to a YAML file.

    Args:
        config: The ExperimentConfig instance to serialize.
        target_path: Destination filepath for the YAML configuration.
    """
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="python")
    # Convert Path objects to string for clean YAML representation
    data["output_dir"] = str(data["output_dir"])
    data["log_dir"] = str(data["log_dir"])

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
