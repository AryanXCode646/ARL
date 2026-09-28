# Architecture Overview

AdaptiveRL is structured around a minimal, modular pipeline consisting of four primary components:

```text
src/adaptive_rl/
├── environments/      # 3D kinematic drone navigation environment (Gymnasium)
├── algorithms/        # RL algorithm registry and wrappers (Stable-Baselines3 PPO & SAC)
├── training/          # Unified RL training loop (RLTrainer/PPOTrainer), checkpointing, and artifact export
├── evaluation/        # Multi-episode deterministic evaluator, policy comparison, and metrics
├── benchmarking/      # Controlled ablations and cross-algorithm benchmarking (PPO vs SAC)
├── config.py          # Strongly typed Pydantic configuration schemas
└── cli.py             # User-facing Typer CLI application
```

## Component Details

### 1. Environment (`adaptive_rl.environments.drone`)
- Implements `DroneNavigation3DEnv` conforming to the Gymnasium `Env` interface.
- State: 3D position $p \in \mathbb{R}^3$, 3D velocity $v \in \mathbb{R}^3$, target coordinate $g \in \mathbb{R}^3$, and obstacle positions.
- Observations: 29-dimensional normalized vector:
  - Relative target vector (3 dims)
  - Drone velocity (3 dims)
  - Normalized drone position (3 dims)
  - Normalized target position (3 dims)
  - Target distance scalar (1 dim)
  - 16-ray spherical LiDAR rangefinder readings (16 dims)
- Actions: 3-dimensional continuous thrust acceleration $a \in [-1.0, 1.0]^3$.

### 2. Algorithms & Training (`adaptive_rl.algorithms`, `adaptive_rl.training`)
- **PPO (`PPOAlgorithm`)**: On-policy actor-critic algorithm wrapped from Stable-Baselines3.
  - Multi-Layer Perceptron (`MlpPolicy`).
  - Generalized Advantage Estimation (GAE) with $\gamma=0.99, \lambda=0.95$.
- **SAC (`SACAlgorithm`)**: Off-policy maximum-entropy actor-critic algorithm wrapped from Stable-Baselines3.
  - Replay buffer size $100{,}000$, soft target update $\tau=0.005$, learning rate $3 \times 10^{-4}$, batch size $256$.
- **Trainer (`RLTrainer` / `PPOTrainer`)**: Generalized trainer dispatching algorithms by configuration (`config.algorithm.name`), orchestrating callbacks, periodic checkpointing, and structured metadata logging.

### 3. Evaluation & Benchmarking (`adaptive_rl.evaluation`, `adaptive_rl.benchmarking`)
- Evaluates trained checkpoints across deterministic seed sets.
- Tracks binary outcomes: Target Reached (Success), Obstacle Collision, Boundary Violation, Timeout.
- Measures trajectory metrics: path length, path efficiency, obstacle surface clearance, max velocity/acceleration.
- Provides comparative benchmarking workflow (`adaptive-rl benchmark compare-algorithms`) under strictly fair, identical environment configurations and evaluation seeds.
- Serializes evaluation reports to `artifacts/evaluation.json` and `artifacts/algorithm_comparison.json`.
