"""Tests verifying Typer CLI commands and execution."""

from pathlib import Path
from typer.testing import CliRunner
from adaptive_rl.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """Verify adaptive-rl --help prints help and available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AdaptiveRL" in result.output
    assert "config" in result.output
    assert "train" in result.output
    assert "evaluate" in result.output


def test_cli_version() -> None:
    """Verify adaptive-rl version displays package version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "AdaptiveRL" in result.output
    assert "0.1.0" in result.output


def test_cli_info() -> None:
    """Verify adaptive-rl info displays the roadmap table."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Roadmap" in result.output
    assert "Phase 1" in result.output


def test_cli_config_validate_success() -> None:
    """Verify adaptive-rl config validate succeeds for valid YAML configuration."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "ppo.yaml"
    result = runner.invoke(app, ["config", "validate", str(config_path)])
    assert result.exit_code == 0
    assert "Configuration is valid" in result.output


def test_cli_config_validate_failure(tmp_path: Path) -> None:
    """Verify adaptive-rl config validate fails with code 1 for invalid YAML."""
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("invalid: yaml: syntax: [", encoding="utf-8")
    result = runner.invoke(app, ["config", "validate", str(bad_config)])
    assert result.exit_code == 1
    assert "Configuration validation error" in result.output


def test_cli_train_honest_notice() -> None:
    """Verify adaptive-rl train displays honest notice about Phase 4 scheduling."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0
    assert "Phase 4" in result.output


def test_cli_evaluate_honest_notice() -> None:
    """Verify adaptive-rl evaluate displays honest notice about Phase 5 scheduling."""
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 0
    assert "Phase 5" in result.output
