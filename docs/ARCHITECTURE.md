# Architecture Overview

AdaptiveRL is structured around a minimal, modular pipeline consisting of four primary components:

```text
src/adaptive_rl/
├── environments/      # 3D kinematic drone navigation environment (Gymnasium)
├── algorithms/        # RL algorithm registry and wrappers (Stable-Baselines3 PPO)
├── planners/          # Classical motion-planning baselines (A* 3D lattice planner)
├── training/          # PPO training loop, checkpointing, and artifact export
├── evaluation/        # Multi-episode deterministic evaluator and metrics
├── config.py          # Strongly typed Pydantic configuration schemas
└── cli.py             # User-facing Typer CLI application
```

## Component Details

### 1. Environment (`adaptive_rl.environments.drone`)
- Implements `DroneNavigation3DEnv` conforming to the Gymnasium `Env` interface.
- State: 3D position $p \in \mathbb{R}^3$, 3D velocity $v \in \mathbb{R}^3$, target coordinate $g \in \mathbb{R}^3$, and obstacle positions.
- Observations: 29-dimensional normalized vector:
  - Normalized drone position (3 dims)
  - Drone velocity (3 dims)
  - Normalized target position (3 dims)
  - Relative target vector (3 dims)
  - Normalized distance to goal (1 dim)
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
- Computes trajectory safety and efficiency metrics: path length, path efficiency, minimum clearance.
- Serializes evaluation reports to `artifacts/evaluation.json` and `artifacts/evaluation.csv`.

### 4. Classical Planning Baseline (`adaptive_rl.planners.astar3d`)
- Implements `AStar3DPlanner`: a deterministic classical 3D motion-planning baseline operating on a spatial lattice.
- Independent of PyTorch, trained weights, and reinforcement learning dependencies.
- **Search Space & Neighborhood**: Configurable 3D lattice resolution (default: 0.5 m) supporting 6-connected (axis-aligned) or 26-connected (diagonal) motion.
- **Heuristic**: Admissible and consistent Euclidean distance $h(n) = \|pos(n) - goal\|$.
- **Analytical Clearance Geometry**:
  - Every segment $[p_0, p_1]$ is verified using exact segment-to-sphere analytical distance:
    $$t^* = \text{clamp}\left(\frac{(C - p_0) \cdot (p_1 - p_0)}{\|p_1 - p_0\|^2}, 0, 1\right)$$
    $$\text{dist}(segment, C) = \|p_0 + t^* (p_1 - p_0) - C\|$$
    $$\text{clearance} = \text{dist}(segment, C) - (R_{\text{obs}} + R_{\text{coll}})$$
  - Enforces arena boundary clearance: $\min(p_0[i], p_1[i]) - R_{\text{coll}} > 0$ and $\max(p_0[i], p_1[i]) + R_{\text{coll}} < \text{bounds}[i]$.
- **Empty-Arena Behavior**: Returns direct optimal two-point straight line $[start, goal]$ with 100% path efficiency without intermediate lattice waypoints.
- **Fair Benchmarking Adapter**: `compare_with_planner` evaluates PPO, AStar3D, and Random Policy under identical procedural seeds and exports structured comparison reports to `artifacts/evaluation_planner_comparison.json`.
- **CLI Command**:
  ```bash
  adaptive-rl evaluate --model artifacts/models/drone_ppo_demo_final.zip --compare-planner astar
  ```
- **Limitations & Feasibility Paradigms**:
  - **Dynamic Feasibility (PPO & Random Policy)**: Evaluated through closed-loop physics simulation in `DroneNavigation3DEnv`. The drone must generate continuous acceleration control commands to navigate under inertia, velocity drag, and actuation limits. Success requires dynamically executing the trajectory without colliding.
  - **Geometric Feasibility (A* Planner)**: Evaluated as open-loop 3D spatial path planning. Success denotes finding an obstacle-free, boundary-clearing geometric path from start to goal on the discretized spatial lattice. The path is not executed through drone attitude dynamics or closed-loop tracking control.

