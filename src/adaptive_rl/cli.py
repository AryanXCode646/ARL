"""Command Line Interface for AdaptiveRL.

Focuses on the core drone reinforcement learning story:
Train a PPO agent to navigate a simulated 3D drone through obstacles toward
a target, evaluate the trained agent, and visualize the flight demonstration.
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
from adaptive_rl.environments.registry import RegistryError, make_env

app = typer.Typer(
    name="adaptive-rl",
    help="AdaptiveRL: Simulated 3D Drone Reinforcement Learning CLI.",
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

console = Console()


@app.command()
def version() -> None:
    """Show the installed AdaptiveRL version and project story."""
    console.print(
        f"[bold green]AdaptiveRL[/bold green] version [bold cyan]{adaptive_rl.__version__}[/bold cyan]\n"
        "[italic]Autonomous 3D Drone Navigation via Reinforcement Learning[/italic]"
    )


@config_app.command(name="validate")
def validate_config(
    path: Path = typer.Argument(..., help="Path to YAML configuration file to validate"),
) -> None:
    """Validate an experiment YAML configuration file against the schema."""
    try:
        cfg = load_config(path)
        training_info = (
            f"{cfg.training.total_timesteps:,} steps (checkpoint freq: {cfg.training.checkpoint_freq})"
            if cfg.training is not None
            else "None"
        )
        console.print(
            Panel.fit(
                f"[bold green]✓ Configuration is valid![/bold green]\n\n"
                f"• [bold]Experiment:[/bold] {cfg.name}\n"
                f"• [bold]Seed:[/bold] {cfg.seed}\n"
                f"• [bold]Algorithm:[/bold] {cfg.algorithm.name.upper()} (LR: {cfg.algorithm.learning_rate}, Gamma: {cfg.algorithm.gamma})\n"
                f"• [bold]Environment:[/bold] {cfg.environment.name} (Max steps: {cfg.environment.max_steps})\n"
                f"• [bold]Training:[/bold] {training_info}\n"
                f"• [bold]Evaluation:[/bold] {cfg.evaluation.eval_episodes} episodes",
                title=f"Valid Configuration: {path.name}",
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


@env_app.command(name="inspect")
def inspect_env(
    name: str = typer.Argument("drone", help="Name of registered environment to inspect"),
) -> None:
    """Inspect observation and action spaces of an environment."""
    try:
        env = make_env(name)
        obs, info = env.reset(seed=42)

        panel_content = [
            f"[bold green]Environment '{name}' verified successfully![/bold green]\n",
            f"• [bold]Type:[/bold] {type(env).__name__}",
            f"• [bold]Observation Space:[/bold] {env.observation_space}",
            f"• [bold]Action Space:[/bold] {env.action_space}",
            f"• [bold]Initial Observation Shape:[/bold] {getattr(obs, 'shape', 'unknown')}",
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
        for candidate in [Path("configs/drone_ppo.yaml"), Path("configs/drone_ppo_demo.yaml")]:
            if candidate.exists():
                config = candidate
                break
        if config is None:
            console.print(
                "[bold red]No configuration file provided.[/bold red] Specify --config <path>"
            )
            raise typer.Exit(code=1)

    try:
        exp_config = load_config(config)
    except ConfigError as err:
        console.print(f"[bold red]Configuration error:[/bold red] {err}")
        raise typer.Exit(code=1)

    if exp_config.training is None:
        console.print("[bold red]Configuration error:[/bold red] 'training' section is required.")
        raise typer.Exit(code=1)

    if timesteps is not None:
        exp_config.training.total_timesteps = timesteps
    if seed is not None:
        exp_config.seed = seed

    console.print(
        Panel.fit(
            f"[bold green]Starting Drone RL Training: {exp_config.name}[/bold green]\n\n"
            f"• [bold]Algorithm:[/bold] {exp_config.algorithm.name.upper()}\n"
            f"• [bold]Environment:[/bold] {exp_config.environment.name}\n"
            f"• [bold]Total Timesteps:[/bold] {exp_config.training.total_timesteps:,}\n"
            f"• [bold]Checkpoint Freq:[/bold] {exp_config.training.checkpoint_freq}\n"
            f"• [bold]Seed:[/bold] {exp_config.seed}\n"
            f"• [bold]Output Dir:[/bold] {exp_config.output_dir}",
            title="PPO Drone Training Pipeline",
            border_style="cyan",
        )
    )

    from adaptive_rl.training.trainer import get_trainer

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
                f"• [bold]Metadata:[/bold] {result.metadata_path}",
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
        None, "--model", "-m", help="Path to trained model weights (.zip)"
    ),
    episodes: Optional[int] = typer.Option(
        20, "--episodes", "-e", help="Number of evaluation episodes"
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
        for candidate in [Path("configs/drone_ppo.yaml"), Path("configs/drone_ppo_demo.yaml")]:
            if candidate.exists():
                config = candidate
                break
        if config is None:
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
            f"• [bold]Deterministic:[/bold] {deterministic}",
            title="Evaluation Engine",
            border_style="cyan",
        )
    )

    from adaptive_rl.algorithms.ppo import PPOAlgorithm
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
            deterministic=deterministic,
            base_seed=exp_config.seed,
        )

        success_pct = (
            f"{metrics.success_rate * 100:.1f}%" if metrics.success_rate is not None else "N/A"
        )
        collision_pct = (
            f"{metrics.collision_rate * 100:.1f}%" if metrics.collision_rate is not None else "N/A"
        )

        console.print("\n[bold]## Evaluation[/bold]")
        console.print(f"Episodes: {metrics.episodes}")
        console.print(f"Success rate: {success_pct}")
        console.print(f"Collision rate: {collision_pct}")
        console.print(f"Mean reward: {metrics.mean_reward:.2f}")
        console.print(f"Mean episode length: {metrics.mean_episode_length:.1f}\n")

        table = Table(title=f"Benchmark Results ({num_episodes} episodes)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Mean Reward", f"{metrics.mean_reward:.2f} ± {metrics.std_reward:.2f}")
        table.add_row("Min / Max Reward", f"{metrics.min_reward:.2f} / {metrics.max_reward:.2f}")
        table.add_row("Success Rate", success_pct)
        table.add_row("Collision Rate", collision_pct)
        table.add_row(
            "Mean Episode Length",
            f"{metrics.mean_episode_length:.1f} ± {metrics.std_episode_length:.1f}",
        )
        console.print(table)

        report_target = output_report or (exp_config.output_dir / "evaluation.json")
        saved_path = evaluator.save_report(metrics, report_target)
        console.print(f"\n[bold green]Report saved to:[/bold green] {saved_path}")

        env.close()
    except Exception as err:
        console.print(f"[bold red]Evaluation failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)


@app.command(name="demo-drone")
def demo_drone(
    model: Path = typer.Option(..., "--model", "-m", help="Path to trained model artifact (.zip)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to experiment YAML"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for deterministic demo"),
    max_steps: int = typer.Option(150, "--max-steps", help="Maximum steps for demo episode"),
) -> None:
    """Run a deterministic, visual demonstration of the trained drone agent."""
    if not model.exists():
        console.print(f"[bold red]Model file not found:[/bold red] {model}")
        raise typer.Exit(code=1)

    env_kwargs = {}
    if config is not None and config.exists():
        cfg = load_config(config)
        env_kwargs = cfg.environment.parameters

    from adaptive_rl.algorithms.ppo import PPOAlgorithm

    env = make_env("drone", **env_kwargs)
    algo = PPOAlgorithm.from_pretrained(model, env=env)

    obs, info = env.reset(seed=seed)
    console.print(
        Panel.fit(
            f"[bold green]Starting Autonomous Drone 3D Navigation Demo[/bold green]\n\n"
            f"• [bold]Model:[/bold] {model}\n"
            f"• [bold]Seed:[/bold] {seed}\n"
            f"• [bold]Start Position:[/bold] {info.get('position')}\n"
            f"• [bold]Target Waypoint:[/bold] {info.get('goal')}\n"
            f"• [bold]Obstacles in Arena:[/bold] {info.get('num_obstacles')}\n"
            f"• [bold]Initial Distance to Goal:[/bold] {info.get('distance_to_goal', 0):.2f}m",
            title="Demonstration Flight",
            border_style="cyan",
        )
    )

    if hasattr(env, "render"):
        rendered = env.render()
        if rendered:
            console.print(rendered)

    total_reward = 0.0
    outcome = "UNKNOWN"
    step_num = 0

    for step_num in range(1, max_steps + 1):
        action, _ = algo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += float(reward)

        if step_num % 10 == 0 or terminated or truncated:
            dist = step_info.get("distance_to_goal", 0.0)
            alt = step_info.get("altitude", 0.0)
            spd = step_info.get("speed", 0.0)
            min_obs = step_info.get("min_obstacle_distance", 0.0)
            console.print(
                f"Step {step_num:03d} | Alt: {alt:4.1f}m | Spd: {spd:4.1f}m/s | "
                f"Dist: {dist:5.1f}m | Min Obs: {min_obs:4.1f}m | Reward: {reward:+6.2f}"
            )

        if terminated or truncated:
            if step_info.get("success"):
                outcome = "SUCCESS"
            elif step_info.get("collision"):
                outcome = f"FAILED / COLLISION ({step_info.get('collision_type', 'obstacle')})"
            else:
                outcome = "FAILED / MAX STEPS REACHED"
            break

    if hasattr(env, "render"):
        rendered = env.render()
        if rendered:
            console.print(rendered)

    env.close()

    style = "bold green" if outcome == "SUCCESS" else "bold red"
    console.print(
        Panel.fit(
            f"[{style}]OUTCOME: {outcome}[/{style}]\n\n"
            f"• [bold]Total Steps:[/bold] {step_num}\n"
            f"• [bold]Cumulative Reward:[/bold] {total_reward:+.2f}\n"
            f"• [bold]Final Distance to Target:[/bold] {step_info.get('distance_to_goal', 0):.2f}m",
            title="Flight Results",
            border_style="green" if outcome == "SUCCESS" else "red",
        )
    )

    if outcome == "SUCCESS":
        console.print("\n[bold green]SUCCESS[/bold green]")
    else:
        console.print("\n[bold red]FAILED / COLLISION[/bold red]")


if __name__ == "__main__":
    app()
