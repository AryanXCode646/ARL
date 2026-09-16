# AdaptiveRL — Multi-Environment Reinforcement Learning Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**AdaptiveRL** is a modular, multi-environment reinforcement learning framework designed to train, evaluate, and benchmark adaptive agents across diverse problem domains—eventually scaling to autonomous 3D drone navigation in complex, dynamic obstacle fields.

---

## 1. Project Goals

* **Domain-Agnostic Core:** Decouple reinforcement learning algorithms from environment specifics using standardized Farama Gymnasium interfaces.
* **Algorithm Adapters:** Wrap battle-tested algorithms (such as Stable-Baselines3 PPO and SAC) behind unified agent interfaces rather than reinventing algorithms from scratch.
* **Reproducibility First:** Enforce deterministic seeding and declarative YAML configuration schemas for every experiment.
* **Progressive Benchmarking:** Progress through discrete GridWorld, continuous 2D navigation, traffic flow control, and autonomous 3D drone navigation.
* **Contributor Friendly:** Modern Python packaging (`src/` layout, `pyproject.toml`), automated test suites, type checking, and clean development workflows.

---

## 2. Current Development Status

AdaptiveRL is developed incrementally across verifiable phases.

| Phase | Milestone | Status | Description |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Repository Foundation & Skeleton** | **Completed** | Project structure, packaging, YAML configuration schemas, honest interfaces, and CLI. |
| **Phase 2** | **Environment Abstraction & Registry** | **Completed** | Gymnasium environment wrapper contract, registry, and factory. |
| **Phase 3** | **Procedural GridWorld** | **Completed** | Procedurally generated 2D grid navigation with BFS path verification and ASCII rendering. |
| **Phase 4** | **PPO Training Engine** | **Current** | Stable-Baselines3 PPO adapter, metric callbacks, checkpointing, and end-to-end trainer. |
| Phase 5 | Evaluation Engine & Standard Metrics | *Upcoming* | 100-episode benchmarking, success rate, and collision tracking. |
| Phase 6 | Continuous 2D Navigation | *Planned* | Continuous velocity control with ray-based obstacle sensing. |
| Phase 7 | Curriculum Learning | *Planned* | Staged obstacle density and disturbance curriculum. |
| Phase 8 | Traffic Signal Environment | *Planned* | Non-spatial queue management demonstrating framework multi-domain versatility. |
| Phase 9-10| Autonomous 3D Drone Environment | *Planned* | 3D kinematics, wind disturbances, dynamic obstacles, and energy constraints. |
| Phase 11-17| Research Baselines & Hardening | *Planned* | Generalization benchmarks, classical planners (A*, RRT*), and CI hardening. |


> [!NOTE]
> In accordance with Phase 1 constraints, concrete reinforcement learning training and concrete environment physics are scheduled for subsequent phases. Current APIs represent honest structural interfaces.

---

## 3. Architecture at a Glance

```
adaptive-rl/
├── configs/                   # Declarative YAML experiment configurations
│   ├── ppo.yaml
│   ├── sac.yaml
│   ├── navigation.yaml
│   └── drone.yaml
├── docs/                      # Architectural specifications and research design
│   ├── ARCHITECTURE.md
│   ├── RESEARCH.md
│   └── EXPERIMENTS.md
├── experiments/               # Experiment output directories (.gitignore tracked)
│   ├── results/
│   └── logs/
├── src/
│   └── adaptive_rl/           # Core platform package
│       ├── algorithms/        # Base algorithm interfaces and future SB3 adapters
│       ├── environments/      # Gymnasium contracts, registry, and factories
│       ├── rewards/           # Modular reward function base interfaces
│       ├── training/          # Trainers, callbacks, and checkpoint managers
│       ├── evaluation/        # Benchmark evaluators, metrics, and scenarios
│       ├── models/            # Model artifact storage and metadata management
│       ├── visualization/     # Renderers and plot generation
│       ├── experiments/       # Experiment orchestration runners
│       ├── config.py          # Pydantic schema validation & YAML parser
│       └── cli.py             # Typer command-line interface
└── tests/                     # Automated pytest suite
```

For in-depth architectural principles, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 4. Installation & Setup

### Prerequisites
* Python 3.10+ (tested through Python 3.14)
* `git`

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/ashishsinghbora/ARL.git
cd ARL

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install AdaptiveRL in editable mode
pip install -e .

# Or install with development dependencies (pytest, ruff, mypy)
pip install -e ".[dev]"
```

---

## 5. Command-Line Interface (CLI)

AdaptiveRL includes a CLI tool (`adaptive-rl`):

```bash
# View help and available commands
adaptive-rl --help

# Show installed version and current milestone
adaptive-rl version

# Inspect development roadmap and completed phases
adaptive-rl info

# List and inspect registered environments
adaptive-rl env list
adaptive-rl env inspect gridworld

# Run an interactive or simulated rollout
adaptive-rl env run gridworld --steps 15 --seed 42
```

---

## 6. Procedural GridWorld Environment

AdaptiveRL provides a procedurally generated 2D grid navigation environment compliant with the Farama Gymnasium contract.

* **Observation Space:** `Box(4,)` containing normalized coordinates `[agent_x, agent_y, goal_x, goal_y]`.
* **Action Space:** `Discrete(4)` corresponding to `UP (0)`, `DOWN (1)`, `LEFT (2)`, `RIGHT (3)`.
* **Rewards:** `+100.0` for reaching goal, `-100.0` for obstacle collision, `-1.0` per step.
* **Solvability Guarantee:** Breadth-First Search (BFS) path verification guarantees a valid collision-free path exists for every generated obstacle layout.

### Python Example

```python
from adaptive_rl.environments import make_env

# Instantiate via factory with custom dimensions and obstacle count
env = make_env("gridworld", width=6, height=5, num_obstacles=3, max_steps=50)

obs, info = env.reset(seed=42)
print("Initial observation:", obs)
env.render()

# Step through the environment
obs, reward, terminated, truncated, info = env.step(1)  # DOWN
env.close()
```

---

## 7. PPO Training Engine

AdaptiveRL features an end-to-end PPO training engine integrating Stable-Baselines3 with automated metric logging callbacks and model checkpoint management.

### Training via CLI

```bash
# Train on GridWorld with PPO for 5,000 steps
adaptive-rl train --config configs/gridworld_ppo.yaml --timesteps 5000

# Train on standard CartPole baseline
adaptive-rl train --config configs/ppo.yaml --timesteps 10000
```

### Training via Python API

```python
from adaptive_rl.config import load_config
from adaptive_rl.training import PPOTrainer

# 1. Load experiment configuration
config = load_config("configs/gridworld_ppo.yaml")

# 2. Instantiate trainer and run optimization
trainer = PPOTrainer(config=config)
result = trainer.fit()

print(f"Trained {result.total_timesteps} steps across {result.episodes_completed} episodes.")
print(f"Final model saved to: {result.final_model_path}")
print(f"Saved {len(result.checkpoints)} periodic checkpoints.")
```

---

## 8. Running Tests

Execute the automated test suite with `pytest`:
```bash
# Run all tests
pytest -v tests/

# Run with coverage
pytest --cov=adaptive_rl tests/
```

---

## 9. Contributing

We welcome contributions! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming conventions, quality gates, and code formatting standards before opening a pull request.

---

## 10. License

This project is licensed under the [MIT License](LICENSE).


