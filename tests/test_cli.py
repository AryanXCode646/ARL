"""Tests verifying Typer CLI commands and execution."""

from pathlib import Path

from typer.testing import CliRunner

from adaptive_rl.cli import app
from adaptive_rl.environments.registry import register, registry
from adaptive_rl.environments.testing import DummyTestEnv

runner = CliRunner()


def test_cli_help() -> None:
    """Verify adaptive-rl --help prints help and available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AdaptiveRL" in result.output
    assert "config" in result.output
    assert "env" in result.output
    assert "train" in result.output
    assert "evaluate" in result.output


def test_cli_version() -> None:
    """Verify adaptive-rl version displays package version and phase."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "AdaptiveRL" in result.output
    assert "Phase 2" in result.output


def test_cli_info() -> None:
    """Verify adaptive-rl info displays the roadmap table with Phase 1 and 2 completed."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Roadmap" in result.output
    assert "Phase 1" in result.output
    assert "Phase 2" in result.output


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


def test_cli_env_list_empty() -> None:
    """Verify adaptive-rl env list displays informative notice when empty."""
    registry.clear()
    result = runner.invoke(app, ["env", "list"])
    assert result.exit_code == 0
    assert "Gymnasium environments" in result.output


def test_cli_env_list_with_registered_env() -> None:
    """Verify adaptive-rl env list displays registered environment metadata."""
    registry.clear()
    register(
        "cli_dummy",
        lambda: DummyTestEnv(),
        metadata={"observation_type": "box", "action_type": "discrete"},
    )
    result = runner.invoke(app, ["env", "list"])
    assert result.exit_code == 0
    assert "cli_dummy" in result.output
    registry.clear()


def test_cli_env_inspect_success() -> None:
    """Verify adaptive-rl env inspect inspects spaces for a standard Gymnasium environment."""
    result = runner.invoke(app, ["env", "inspect", "CartPole-v1"])
    assert result.exit_code == 0
    assert "verified successfully" in result.output
    assert "Observation Space" in result.output
    assert "Action Space" in result.output


def test_cli_env_inspect_failure() -> None:
    """Verify adaptive-rl env inspect fails gracefully for unknown environment."""
    result = runner.invoke(app, ["env", "inspect", "NonExistentEnv-v999"])
    assert result.exit_code == 1
    assert "Environment inspection failed" in result.output


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
