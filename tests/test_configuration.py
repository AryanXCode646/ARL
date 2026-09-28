"""Tests verifying configuration schema validation, loading, and serialization."""

from pathlib import Path

import pytest

from adaptive_rl.config import (
    ConfigError,
    ExperimentConfig,
    compute_config_sha256,
    load_config,
    save_config,
)


def test_load_committed_drone_configs() -> None:
    """Verify committed drone experiment configurations load and validate."""
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    for filename in ["drone_ppo.yaml", "drone_ppo_demo.yaml"]:
        filepath = config_dir / filename
        assert filepath.is_file(), f"Expected configuration file missing: {filepath}"
        cfg = load_config(filepath)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name
        assert cfg.seed >= 0
        assert cfg.algorithm.name == "ppo"
        assert cfg.algorithm.learning_rate > 0.0
        assert cfg.training is not None and cfg.training.total_timesteps > 0
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
  name: "drone"
training:
  total_timesteps: 1000
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Configuration validation failed"):
        load_config(invalid_yaml)


def test_config_serialization_roundtrip(tmp_path: Path) -> None:
    """Verify an ExperimentConfig can be serialized and re-loaded identically."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "drone_ppo.yaml"
    original_cfg = load_config(config_path)

    saved_path = tmp_path / "roundtrip.yaml"
    save_config(original_cfg, saved_path)

    reloaded_cfg = load_config(saved_path)
    assert original_cfg.name == reloaded_cfg.name
    assert original_cfg.seed == reloaded_cfg.seed
    assert original_cfg.algorithm.name == reloaded_cfg.algorithm.name
    assert original_cfg.algorithm.learning_rate == reloaded_cfg.algorithm.learning_rate
    assert original_cfg.training.total_timesteps == reloaded_cfg.training.total_timesteps


def test_config_sha256_deterministic(tmp_path: Path) -> None:
    """compute_config_sha256 is deterministic and independent of runtime paths."""
    cfg_yaml = tmp_path / "test_exp.yaml"
    cfg_yaml.write_text(
        """
name: "sha256_test"
seed: 42
algorithm:
  name: "ppo"
environment:
  name: "drone"
training:
  total_timesteps: 1000
""",
        encoding="utf-8",
    )
    cfg1 = load_config(cfg_yaml)
    hash1 = compute_config_sha256(cfg1)
    assert len(hash1) == 64
    assert isinstance(hash1, str)

    # Change runtime output_dir
    cfg1.output_dir = tmp_path / "somewhere_else"
    cfg1.log_dir = tmp_path / "somewhere_else" / "logs"
    hash2 = compute_config_sha256(cfg1)
    assert hash1 == hash2

    # Changing a hyperparameter changes the hash
    cfg1.seed = 999
    assert compute_config_sha256(cfg1) != hash1
