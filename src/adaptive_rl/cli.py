"""Command Line Interface for AdaptiveRL.

Provides commands for inspecting environment status, validating configurations,
and guiding developers through the multi-phase implementation roadmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
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
    help="Environment discovery, inspection, and simulation commands.",
    no_args_is_help=True,
)
app.add_typer(env_app, name="env")

curriculum_app = typer.Typer(
    name="curriculum",
    help="Curriculum learning inspection and preset management commands.",
    no_args_is_help=True,
)
app.add_typer(curriculum_app, name="curriculum")

console = Console()


@app.command()
def version() -> None:
    """Show the installed AdaptiveRL version and phase status."""
    console.print(
        f"[bold green]AdaptiveRL[/bold green] version [bold cyan]{adaptive_rl.__version__}[/bold cyan] "
        f"([yellow]Phase 9: Autonomous 3D Drone Navigation[/yellow])"
    )


@app.command()
def info() -> None:
    """Display platform architecture status and implementation roadmap."""
    table = Table(title="AdaptiveRL — Implementation Roadmap Status")
    table.add_column("Phase", style="cyan", no_wrap=True)
    table.add_column("Milestone Name", style="magenta")
    table.add_column("Status", style="green")

    table.add_row(
        "Phase 1",
        "Repository Foundation and Architecture Skeleton",
        "[bold green]COMPLETED[/bold green]",
    )
    table.add_row(
        "Phase 2", "Environment Abstraction and Registry", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 3", "Procedurally Generated GridWorld", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 4", "PPO Training Engine (SB3 Wrapper)", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 5", "Evaluation Engine and Standard Metrics", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row("Phase 6", "Continuous 2D Navigation", "[bold green]COMPLETED[/bold green]")
    table.add_row("Phase 7", "Curriculum Learning", "[bold green]COMPLETED[/bold green]")
    table.add_row("Phase 8", "Traffic Signal Optimization", "[bold green]COMPLETED[/bold green]")
    table.add_row("Phase 9", "Autonomous 3D Drone Navigation", "[bold green]COMPLETED[/bold green]")
    table.add_row("Phase 10", "Drone Disturbances and Constraints", "[yellow]PLANNED[/yellow]")
    table.add_row(
        "Phase 11-17", "Research Baselines, Hardening & Final Audit", "[yellow]PLANNED[/yellow]"
    )

    console.print(table)


@config_app.command(name="validate")
def validate_config(
    path: Path = typer.Argument(
        ..., help="Path to YAML configuration file to validate", exists=False
    ),
) -> None:
    """Validate an experiment YAML configuration file against the schema."""
    try:
        cfg = load_config(path)
        curr_info = ""
        if cfg.curriculum is not None and cfg.curriculum.enabled:
            curr_preset = cfg.curriculum.preset or f"{len(cfg.curriculum.stages)} custom stages"
            curr_info = f"\n• [bold]Curriculum:[/bold] Enabled ({curr_preset}, Window: {cfg.curriculum.eval_window})"

        console.print(
            Panel.fit(
                f"[bold green]✓ Configuration is valid![/bold green]\n\n"
                f"• [bold]Experiment:[/bold] {cfg.name}\n"
                f"• [bold]Seed:[/bold] {cfg.seed}\n"
                f"• [bold]Algorithm:[/bold] {cfg.algorithm.name} (LR: {cfg.algorithm.learning_rate}, Gamma: {cfg.algorithm.gamma})\n"
                f"• [bold]Environment:[/bold] {cfg.environment.name} (Max steps: {cfg.environment.max_steps})\n"
                f"• [bold]Training:[/bold] {cfg.training.total_timesteps:,} steps (Checkpoint freq: {cfg.training.checkpoint_freq})\n"
                f"• [bold]Evaluation:[/bold] {cfg.evaluation.eval_episodes} episodes"
                f"{curr_info}",
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
    name: str = typer.Argument(..., help="Name of registered or Gymnasium environment to inspect"),
) -> None:
    """Inspect observation and action spaces of an environment."""
    try:
        env = make_env(name)
        obs, info = env.reset()

        panel_content = [
            f"[bold green]Environment '{name}' verified successfully![/bold green]\n",
            f"• [bold]Type:[/bold] {type(env).__name__}",
            f"• [bold]Observation Space:[/bold] {env.observation_space}",
            f"• [bold]Action Space:[/bold] {env.action_space}",
            f"• [bold]Initial Observation Shape:[/bold] {getattr(obs, 'shape', 'discrete/scalar')}",
            f"• [bold]Reset Info:[/bold] {info}",
        ]

        if hasattr(env, "render"):
            rendered = env.render()
            if rendered:
                panel_content.append(f"\n[bold]Initial Layout:[/bold]\n{rendered}")

        env.close()

        console.print(
            Panel.fit(
                "\n".join(panel_content),
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


@env_app.command(name="run")
def run_env(
    name: str = typer.Argument("gridworld", help="Environment to execute"),
    steps: int = typer.Option(10, "--steps", "-s", help="Number of steps to simulate"),
    seed: int = typer.Option(42, "--seed", help="Random seed for environment reset"),
) -> None:
    """Simulate an environment episode with random actions and textual rendering."""
    try:
        env = make_env(name)
        obs, info = env.reset(seed=seed)
        console.print(f"[bold green]Starting simulation for '{name}' (seed={seed})[/bold green]\n")

        if hasattr(env, "render"):
            rendered = env.render()
            if rendered:
                console.print(Panel(str(rendered), title="Initial State"))

        total_reward = 0.0
        step_count = 0

        for s in range(1, steps + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += float(reward)
            step_count += 1
            if isinstance(action, np.ndarray):
                action_desc = np.array2string(action, precision=2, separator=",")
            else:
                action_desc = step_info.get("action_name", str(action))
            console.print(
                f"Step {s:02d}: Action={action_desc:<18} -> Reward={reward:+6.1f} | Terminated={terminated} | Truncated={truncated}"
            )

            if terminated or truncated:
                if step_info.get("collision"):
                    outcome = "COLLISION!"
                elif step_info.get("overflow"):
                    outcome = "QUEUE OVERFLOW!"
                elif step_info.get("success"):
                    outcome = "SUCCESS / GOAL REACHED!"
                else:
                    outcome = "MAX STEPS REACHED"
                console.print(f"\n[bold yellow]Episode ended at step {s}: {outcome}[/bold yellow]")
                if hasattr(env, "render"):
                    rendered = env.render()
                    if rendered:
                        console.print(Panel(str(rendered), title="Final State"))
                break

        console.print(
            Panel.fit(
                f"[bold]Total Steps:[/bold] {step_count}\n"
                f"[bold]Cumulative Reward:[/bold] {total_reward:+.1f}\n"
                f"[bold]Final Observation:[/bold] {obs}",
                title="Simulation Summary",
                border_style="green",
            )
        )
        env.close()
    except RegistryError as err:
        console.print(f"[bold red]Failed to run environment:[/bold red] {err}")
        raise typer.Exit(code=1)


@curriculum_app.command(name="list")
def list_curriculums() -> None:
    """List all available built-in curriculum schedules."""
    from adaptive_rl.curriculum.presets import CURRICULUM_PRESETS

    table = Table(title="AdaptiveRL — Predefined Curriculum Schedules")
    table.add_column("Preset Name", style="cyan", no_wrap=True)
    table.add_column("Stages", style="green")
    table.add_column("Description", style="white")

    for name, builder in sorted(CURRICULUM_PRESETS.items()):
        curr = builder()
        table.add_row(
            name,
            f"{len(curr.stages)} stages",
            f"Progressive curriculum for {curr.name}",
        )
    console.print(table)


@curriculum_app.command(name="inspect")
def inspect_curriculum(
    name: str = typer.Argument("navigation", help="Curriculum preset name to inspect"),
) -> None:
    """Inspect stages, progression thresholds, and parameters of a curriculum preset."""
    from adaptive_rl.curriculum.presets import get_curriculum_preset

    try:
        curr = get_curriculum_preset(name)
    except ValueError as err:
        console.print(f"[bold red]Curriculum lookup failed:[/bold red] {err}")
        raise typer.Exit(code=1)

    table = Table(title=f"Curriculum Stages: {curr.name} ({len(curr.stages)} stages)")
    table.add_column("Stage ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Success Threshold", style="green")
    table.add_column("Mean Reward Threshold", style="yellow")
    table.add_column("Min Episodes", style="blue")
    table.add_column("Parameters", style="white")

    for stage in curr.stages:
        st_str = (
            f"{stage.success_threshold * 100:.0f}%"
            if stage.success_threshold is not None
            else "None"
        )
        mr_str = (
            f"{stage.mean_reward_threshold:.1f}"
            if stage.mean_reward_threshold is not None
            else "None"
        )
        params_str = ", ".join(f"{k}={v}" for k, v in stage.environment_parameters.items())
        table.add_row(
            str(stage.stage_id),
            stage.name,
            st_str,
            mr_str,
            str(stage.min_episodes),
            params_str or "(default)",
        )

    console.print(table)


@app.command()
def train(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to training configuration YAML"
    ),
    timesteps: Optional[int] = typer.Option(
        None, "--timesteps", "-t", help="Override total training timesteps"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help="Override experiment random seed"
    ),
) -> None:
    """Train a reinforcement learning agent using PPO."""
    if config is None:
        default_config = Path("configs/ppo.yaml")
        if default_config.exists():
            config = default_config
        else:
            console.print(
                "[bold red]No configuration file provided.[/bold red] Specify --config <path>"
            )
            raise typer.Exit(code=1)

    try:
        exp_config = load_config(config)
    except ConfigError as err:
        console.print(f"[bold red]Configuration error:[/bold red] {err}")
        raise typer.Exit(code=1)

    if timesteps is not None:
        exp_config.training.total_timesteps = timesteps
    if seed is not None:
        exp_config.seed = seed

    curriculum_line = ""
    if exp_config.curriculum is not None and exp_config.curriculum.enabled:
        preset_info = (
            exp_config.curriculum.preset or f"{len(exp_config.curriculum.stages)} custom stages"
        )
        curriculum_line = f"\n• [bold]Curriculum:[/bold] Enabled ({preset_info})"

    console.print(
        Panel.fit(
            f"[bold green]Starting Training: {exp_config.name}[/bold green]\n\n"
            f"• [bold]Algorithm:[/bold] {exp_config.algorithm.name.upper()}\n"
            f"• [bold]Environment:[/bold] {exp_config.environment.name}\n"
            f"• [bold]Total Timesteps:[/bold] {exp_config.training.total_timesteps:,}\n"
            f"• [bold]Checkpoint Freq:[/bold] {exp_config.training.checkpoint_freq}\n"
            f"• [bold]Seed:[/bold] {exp_config.seed}\n"
            f"• [bold]Output Dir:[/bold] {exp_config.output_dir}"
            f"{curriculum_line}",
            title=f"{exp_config.algorithm.name.upper()} Training Pipeline",
            border_style="cyan",
        )
    )

    from adaptive_rl.training import get_trainer

    try:
        trainer = get_trainer(config=exp_config)
        result = trainer.fit()

        console.print(
            Panel.fit(
                f"[bold green]Training Completed Successfully![/bold green]\n\n"
                f"• [bold]Total Timesteps Trained:[/bold] {result.total_timesteps:,}\n"
                f"• [bold]Episodes Completed:[/bold] {result.episodes_completed}\n"
                f"• [bold]Mean Reward (last window):[/bold] {result.mean_reward:.2f}\n"
                f"• [bold]Saved Model:[/bold] {result.final_model_path}\n"
                f"• [bold]Checkpoints Created:[/bold] {len(result.checkpoints)}",
                title="Training Summary",
                border_style="green",
            )
        )
    except Exception as err:
        console.print(f"[bold red]Training failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to experiment configuration YAML"
    ),
    model: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Path to model weights artifact (.zip)"
    ),
    episodes: Optional[int] = typer.Option(
        None, "--episodes", "-e", help="Number of evaluation episodes"
    ),
    deterministic: bool = typer.Option(
        True, "--deterministic/--stochastic", help="Use deterministic action selection"
    ),
    output_report: Optional[Path] = typer.Option(
        None, "--output-report", "-o", help="Optional path to export JSON metrics report"
    ),
) -> None:
    """Evaluate a trained agent over multiple benchmark episodes."""
    if config is None:
        default_config = Path("configs/gridworld_ppo.yaml")
        if not default_config.exists():
            default_config = Path("configs/ppo.yaml")
        if default_config.exists():
            config = default_config
        else:
            console.print(
                "[bold red]No configuration file provided.[/bold red] Specify --config <path>"
            )
            raise typer.Exit(code=1)

    try:
        exp_config = load_config(config)
    except ConfigError as err:
        console.print(f"[bold red]Configuration error:[/bold red] {err}")
        raise typer.Exit(code=1)

    num_episodes = episodes or exp_config.evaluation.eval_episodes
    det = deterministic if episodes is not None else exp_config.evaluation.deterministic

    # Resolve model path
    if model is None:
        candidate = exp_config.output_dir / "models" / f"{exp_config.name}_final.zip"
        if candidate.exists():
            model = candidate
        else:
            console.print(
                f"[bold red]No model weights provided.[/bold red] Pass --model <path> or train first to generate {candidate}"
            )
            raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold green]Starting Evaluation: {exp_config.name}[/bold green]\n\n"
            f"• [bold]Model:[/bold] {model}\n"
            f"• [bold]Environment:[/bold] {exp_config.environment.name}\n"
            f"• [bold]Episodes:[/bold] {num_episodes}\n"
            f"• [bold]Deterministic:[/bold] {det}",
            title="Evaluation Engine",
            border_style="cyan",
        )
    )

    from adaptive_rl.algorithms.ppo import PPOAlgorithm
    from adaptive_rl.environments.registry import make_env
    from adaptive_rl.evaluation.evaluator import Evaluator

    try:
        env = make_env(
            exp_config.environment.name,
            **exp_config.environment.parameters,
        )
        algo = PPOAlgorithm.from_pretrained(model, env=env)
        evaluator = Evaluator(algorithm=algo, env=env)

        metrics = evaluator.evaluate(
            num_episodes=num_episodes,
            deterministic=det,
            base_seed=exp_config.seed,
        )

        table = Table(title=f"Benchmark Results ({num_episodes} episodes)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Mean Reward", f"{metrics.mean_reward:.2f} ± {metrics.std_reward:.2f}")
        table.add_row("Min / Max Reward", f"{metrics.min_reward:.2f} / {metrics.max_reward:.2f}")
        table.add_row("Success Rate", f"{metrics.success_rate * 100:.1f}%")
        table.add_row("Collision Rate", f"{metrics.collision_rate * 100:.1f}%")
        table.add_row(
            "Mean Episode Length",
            f"{metrics.mean_episode_length:.1f} ± {metrics.std_episode_length:.1f}",
        )

        console.print(table)

        if output_report is not None:
            saved_path = evaluator.save_report(metrics, output_report)
            console.print(f"\n[bold green]Report saved to:[/bold green] {saved_path}")

        env.close()
    except Exception as err:
        console.print(f"[bold red]Evaluation failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
