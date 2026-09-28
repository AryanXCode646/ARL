# AdaptiveRL

AdaptiveRL is a clean, reproducible reinforcement learning prototype that trains an agent to navigate a simulated 3D drone through obstacles toward a target coordinate.

## What it does

AdaptiveRL provides a lightweight, focused environment for training and evaluating autonomous 3D navigation policies using Proximal Policy Optimization (PPO). The simulation models kinematic 3D drone translation with aerodynamic drag damping, continuous thrust actions, a 16-ray spherical LiDAR sensor, procedural spherical obstacles, and distance-progress reward shaping.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        CLI Commands                         │
│   adaptive-rl {train | evaluate | demo-drone | env inspect} │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌───────────────────────┐             ┌───────────────────────┐
│     Training Loop     │             │    Evaluation Loop    │
│  Stable-Baselines3    │             │ Deterministic Seeding │
│      (PPO Policy)     │             │ Metrics & JSON Export │
└───────────┬───────────┘             └───────────┬───────────┘
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 DroneNavigation3DEnv                        │
│  - 3-DOF kinematic translation with aerodynamic drag        │
│  - 16-ray spherical LiDAR rangefinder for obstacles         │
│  - Bounded 3D flight arena with procedural obstacles        │
│  - Continuous action space [-1.0, 1.0]^3                    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

Clone the repository and install with Python 3.10+ in a virtual environment:

```bash
git clone https://github.com/StellarResearch/ARL.git
cd ARL
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[rl,dev]"
```

Verify the installation:

```bash
pytest
adaptive-rl --help
adaptive-rl env inspect drone
```

## Train

Train the drone navigation agent with PPO using the fast demonstration configuration (25,000 timesteps, ~25 seconds on CPU):

```bash
adaptive-rl train --config configs/drone_ppo_demo.yaml
```

Artifacts are saved to `artifacts/models/drone_ppo_demo_final.zip` and `artifacts/metadata/training.json`.

For a larger training run (50,000 timesteps):

```bash
adaptive-rl train --config configs/drone_ppo.yaml
```

## Evaluate

Run deterministic multi-episode evaluation on the trained model:

```bash
adaptive-rl evaluate \
  --config configs/drone_ppo_demo.yaml \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 20
```

This calculates success rate, collision rate, average reward, and average episode length, saving the summary to `artifacts/evaluation.json`.

## Demo

Run an interactive demonstration of a trained policy navigating the simulated arena:

```bash
adaptive-rl demo-drone \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --seed 42
```

The CLI outputs the flight trajectory, distance to goal, obstacle proximity, and final outcome:
```text
============================================================
Result: SUCCESS: Reached target coordinate!
Steps: 41 | Total Reward: 82.45
Final Position: [24.88, 24.79, 9.94] | Target: [25.0, 25.0, 10.0]
============================================================
```

## Measured Results

Evaluated across 20 deterministic test episodes on an x86-64 CPU:

| Configuration | Timesteps | Training Time | Success Rate | Collision Rate | Mean Reward |
|---|---|---|---|---|---|
| `drone_ppo_demo.yaml` | 25,000 | ~25s | 40.0% | 35.0% | +24.8 |
| `drone_ppo.yaml` | 50,000 | ~42s | 55.0% | 25.0% | +71.1 |

*Note: Results are empirical and subject to seed stochasticity.*

## Limitations

- **Kinematic Simulation**: The flight dynamics use a 3-DOF point-mass model with aerodynamic drag damping, not a 6-DOF rigid-body simulator with motor dynamics or rotor blade aerodynamics.
- **Sensor Model**: LiDAR sensing is simulated using geometric ray-sphere intersections rather than physical beam reflections or sensor noise.
- **Educational Scope**: Designed as an academic college project demonstrating RL navigation principles, not for direct hardware autopilot deployment.
