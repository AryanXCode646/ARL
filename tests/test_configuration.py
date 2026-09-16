"""Tests verifying configuration schema validation, loading, and serialization."""

from pathlib import Path

import pytest

from adaptive_rl.config import (
    ConfigError,
    ExperimentConfig,
    load_config,
    save_config,
)


def test_load_committed_configs() -> None:
    """Verify all repository sample configurations pass schema validation."""
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    sample_files = ["ppo.yaml", "sac.yaml", "navigation.yaml", "drone.yaml"]

    for filename in sample_files:
        filepath = config_dir / filename
        assert filepath.is_file(), f"Expected configuration file missing: {filepath}"
        cfg = load_config(filepath)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name
        assert cfg.seed >= 0
        assert cfg.algorithm.learning_rate > 0.0
        assert cfg.training.total_timesteps > 0
        assert cfg.evaluation.eval_episodes > 0


def test_missing_config_file_raises_error(tmp_path: Path) -> None:
    """Verify non-existent config path raises descriptive ConfigError."""
    non_existent = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config(non_existent)


def test_invalid_yaml_syntax_raises_error(tmp_path: Path) -> None:
    """Verify malformed YAML syntax raises ConfigError."""
    bad_yaml = tmp_path / "syntax_error.yaml"
    bad_yaml.write_text("name: test\nalgorithm: [unclosed list", encoding="utf-8")
    with pytest.raises(ConfigError, match="Failed to parse YAML file"):
        load_config(bad_yaml)


def test_non_dict_yaml_raises_error(tmp_path: Path) -> None:
    """Verify YAML that parses to a non-dictionary raises ConfigError."""
    list_yaml = tmp_path / "list_data.yaml"
    list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(list_yaml)


def test_extra_forbidden_fields_raises_error(tmp_path: Path) -> None:
    """Verify unknown fields trigger validation error due to extra='forbid'."""
    invalid_yaml = tmp_path / "extra_field.yaml"
    invalid_yaml.write_text(
        """
name: "test_exp"
seed: 42
unknown_extra_field: "should_fail"
algorithm:
  name: "ppo"
environment:
  name: "CartPole-v1"
training:
  total_timesteps: 1000
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Configuration validation failed"):
        load_config(invalid_yaml)


def test_invalid_ranges_raises_error(tmp_path: Path) -> None:
    """Verify out-of-bounds parameters (e.g. negative learning rate) trigger validation errors."""
    invalid_yaml = tmp_path / "bad_range.yaml"
    invalid_yaml.write_text(
        """
name: "test_exp"
seed: 42
algorithm:
  name: "ppo"
  learning_rate: -0.001
environment:
  name: "CartPole-v1"
training:
  total_timesteps: 1000
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Configuration validation failed"):
        load_config(invalid_yaml)


def test_config_serialization_roundtrip(tmp_path: Path) -> None:
    """Verify an ExperimentConfig can be serialized and re-loaded identically."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "ppo.yaml"
    original_cfg = load_config(config_path)

    saved_path = tmp_path / "roundtrip.yaml"
    save_config(original_cfg, saved_path)

    reloaded_cfg = load_config(saved_path)
    assert original_cfg.name == reloaded_cfg.name
    assert original_cfg.seed == reloaded_cfg.seed
    assert original_cfg.algorithm.name == reloaded_cfg.algorithm.name
    assert original_cfg.algorithm.learning_rate == reloaded_cfg.algorithm.learning_rate
    assert original_cfg.training.total_timesteps == reloaded_cfg.training.total_timesteps
