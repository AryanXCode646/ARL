"""Command Line Interface for AdaptiveRL.

Provides commands for inspecting environment status, validating configurations,
and guiding developers through the multi-phase implementation roadmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import adaptive_rl
from adaptive_rl.config import ConfigError, load_config
from adaptive_rl.environments.registry import RegistryError, list_all_metadata, make_env

app = typer.Typer(
    name="adaptive-rl",
    help="AdaptiveRL: Multi-Environment Reinforcement Learning Platform CLI.",
    add_completion=False,
    no_args_is_help=True,
)

config_app = typer.Typer(
    name="config",
    help="Configuration inspection and validation commands.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")

env_app = typer.Typer(
    name="env",
    help="Environment discovery and inspection commands.",
    no_args_is_help=True,
)
app.add_typer(env_app, name="env")

console = Console()


@app.command()
def version() -> None:
    """Show the installed AdaptiveRL version and phase status."""
    console.print(
        f"[bold green]AdaptiveRL[/bold green] version [bold cyan]{adaptive_rl.__version__}[/bold cyan] "
        f"([yellow]Phase 2: Environment Abstraction & Registry[/yellow])"
    )


@app.command()
def info() -> None:
    """Display platform architecture status and implementation roadmap."""
    table = Table(title="AdaptiveRL — Implementation Roadmap Status")
    table.add_column("Phase", style="cyan", no_wrap=True)
    table.add_column("Milestone Name", style="magenta")
    table.add_column("Status", style="green")

    table.add_row("Phase 1", "Repository Foundation and Architecture Skeleton", "[bold green]COMPLETED[/bold green]")
    table.add_row("Phase 2", "Environment Abstraction and Registry", "[bold green]COMPLETED[/bold green]")
    table.add_row("Phase 3", "Procedurally Generated GridWorld", "[yellow]PLANNED[/yellow]")
    table.add_row("Phase 4", "PPO Training Engine (SB3 Wrapper)", "[yellow]PLANNED[/yellow]")
    table.add_row("Phase 5", "Evaluation Engine and Standard Metrics", "[yellow]PLANNED[/yellow]")
    table.add_row("Phase 6", "Continuous 2D Navigation", "[yellow]PLANNED[/yellow]")
    table.add_row("Phase 7", "Curriculum Learning", "[yellow]PLANNED[/yellow]")
    table.add_row("Phase 8", "Traffic Signal Environment", "[yellow]PLANNED[/yellow]")
    table.add_row("Phase 9", "Mathematical Drone Navigation Environment", "[yellow]PLANNED[/yellow]")
    table.add_row("Phase 10", "Drone Disturbances and Constraints", "[yellow]PLANNED[/yellow]")
    table.add_row("Phase 11-17", "Research Baselines, Hardening & Final Audit", "[yellow]PLANNED[/yellow]")

    console.print(table)


@config_app.command(name="validate")
def validate_config(
    path: Path = typer.Argument(..., help="Path to YAML configuration file to validate", exists=False)
) -> None:
    """Validate an experiment YAML configuration file against the schema."""
    try:
        cfg = load_config(path)
        console.print(
            Panel.fit(
                f"[bold green]✓ Configuration is valid![/bold green]\n\n"
                f"• [bold]Experiment:[/bold] {cfg.name}\n"
                f"• [bold]Seed:[/bold] {cfg.seed}\n"
                f"• [bold]Algorithm:[/bold] {cfg.algorithm.name} (LR: {cfg.algorithm.learning_rate}, Gamma: {cfg.algorithm.gamma})\n"
                f"• [bold]Environment:[/bold] {cfg.environment.name} (Max steps: {cfg.environment.max_steps})\n"
                f"• [bold]Training:[/bold] {cfg.training.total_timesteps:,} steps (Checkpoint freq: {cfg.training.checkpoint_freq})\n"
                f"• [bold]Evaluation:[/bold] {cfg.evaluation.eval_episodes} episodes",
                title=f"Valid: {path}",
                border_style="green",
            )
        )
    except ConfigError as err:
        console.print(
            Panel.fit(
                f"[bold red]Configuration validation error:[/bold red]\n\n{err}",
                title=f"Invalid: {path}",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@env_app.command(name="list")
def list_envs() -> None:
    """List all registered environments and their metadata."""
    meta_map = list_all_metadata()
    if not meta_map:
        console.print(
            "[yellow]No custom environments currently registered in AdaptiveRL registry.[/yellow]\n"
            "Standard Gymnasium environments (e.g. 'CartPole-v1', 'Pendulum-v1') are also resolvable by the factory."
        )
        return

    table = Table(title="Registered AdaptiveRL Environments")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Version", style="magenta")
    table.add_column("Obs Space", style="green")
    table.add_column("Action Space", style="green")
    table.add_column("Description")

    for name, meta in meta_map.items():
        table.add_row(
            name,
            meta.version,
            meta.observation_type,
            meta.action_type,
            meta.description or "-",
        )

    console.print(table)


@env_app.command(name="inspect")
def inspect_env(
    name: str = typer.Argument(..., help="Name of registered or Gymnasium environment to inspect")
) -> None:
    """Inspect observation and action spaces of an environment."""
    try:
        env = make_env(name)
        obs, info = env.reset()
        env.close()

        console.print(
            Panel.fit(
                f"[bold green]Environment '{name}' verified successfully![/bold green]\n\n"
                f"• [bold]Type:[/bold] {type(env).__name__}\n"
                f"• [bold]Observation Space:[/bold] {env.observation_space}\n"
                f"• [bold]Action Space:[/bold] {env.action_space}\n"
                f"• [bold]Initial Observation Shape:[/bold] {getattr(obs, 'shape', 'discrete/scalar')}\n"
                f"• [bold]Reset Info:[/bold] {info}",
                title=f"Environment Inspection: {name}",
                border_style="cyan",
            )
        )
    except RegistryError as err:
        console.print(
            Panel.fit(
                f"[bold red]Environment inspection failed:[/bold red]\n\n{err}",
                title=f"Error: {name}",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@app.command()
def train(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to training configuration YAML")
) -> None:
    """Start an agent training run (Scheduled for Phase 4: PPO Training Engine)."""
    console.print(
        "[bold yellow]Training engine is scheduled for Phase 4 (PPO Adapter & Trainer).[/bold yellow]\n"
        "In Phase 2, environment abstraction, Gymnasium contracts, and registry factories are active.\n"
        "To inspect environments, run: [bold cyan]adaptive-rl env list[/bold cyan] or [bold cyan]adaptive-rl env inspect <name>[/bold cyan]"
    )
    raise typer.Exit(code=0)


@app.command()
def evaluate(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to evaluation configuration YAML"),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to model weights artifact"),
) -> None:
    """Evaluate a trained agent (Scheduled for Phase 5: Evaluation Engine)."""
    console.print(
        "[bold yellow]Evaluation engine is scheduled for Phase 5 (Evaluation Engine and Standard Metrics).[/bold yellow]\n"
        "In Phase 2, environment abstraction, Gymnasium contracts, and registry factories are active."
    )
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
