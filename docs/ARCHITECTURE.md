# Architecture Overview

AdaptiveRL is structured around a minimal, modular pipeline consisting of four primary components:

```text
src/adaptive_rl/
├── environments/      # 3D kinematic drone navigation environment (Gymnasium)
├── algorithms/        # RL algorithm registry and wrappers (Stable-Baselines3 PPO)
├── training/          # PPO training loop, checkpointing, and artifact export
├── evaluation/        # Multi-episode deterministic evaluator and metrics
├── config.py          # Strongly typed Pydantic configuration schemas
└── cli.py             # User-facing Typer CLI application
```

## Component Details

### 1. Environment (`adaptive_rl.environments.drone`)
- Implements `DroneNavigation3DEnv` conforming to the Gymnasium `Env` interface.
- State: 3D position $p \in \mathbb{R}^3$, 3D velocity $v \in \mathbb{R}^3$, target coordinate $g \in \mathbb{R}^3$, and obstacle positions.
- Observations: 28-dimensional normalized vector:
  - Relative target vector (3 dims)
  - Drone velocity (3 dims)
  - Normalized drone position (3 dims)
  - Normalized target position (3 dims)
  - 16-ray spherical LiDAR rangefinder readings (16 dims)
- Actions: 3-dimensional continuous thrust acceleration $a \in [-1.0, 1.0]^3$.

### 2. Algorithm & Training (`adaptive_rl.training.trainer`)
- Wraps `stable_baselines3.PPO` with standardized hyperparameters:
  - Multi-Layer Perceptron (MlpPolicy) with orthogonal initialization.
  - Generalized Advantage Estimation (GAE) with $\gamma=0.99, \lambda=0.95$.
  - Entropy regularization coefficient $0.01$ for exploration.

### 3. Evaluation Engine (`adaptive_rl.evaluation.evaluator`)
- Evaluates trained checkpoints across deterministic seed sets.
- Tracks binary outcomes: Target Reached (Success), Obstacle Collision, Boundary Violation, Timeout.
- Serializes evaluation reports to `artifacts/evaluation.json`.
