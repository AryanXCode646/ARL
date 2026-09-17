# AdaptiveRL Research & Benchmark Strategy

## Honesty Statement

> **IMPORTANT:** This document separates what is **actually implemented and tested** from
> what is **planned or aspirational**. Benchmark numbers in this document are produced by
> running the documented commands — no results are fabricated.

---

## 1. Research Objectives

AdaptiveRL investigates three primary research questions:

1. **Multi-Domain Adaptability:** Can a single, unspecialized RL training engine learn
   policies across discrete control (GridWorld), continuous control (2D navigation),
   discrete switching (traffic signals), and 3D kinematics (drone navigation)?
2. **Generalization to Unseen Distributions:** How well do agents trained on procedurally
   generated environments generalize to held-out test distributions?
3. **RL vs. Classical Planning:** Under what conditions does learned RL outperform or
   complement classical planners (A\*, RRT\*)?

---

## 2. Implementation Status

### 2.1 IMPLEMENTED — Runnable Right Now

| Component | Status | Run Command |
| :--- | :--- | :--- |
| GridWorld environment (6×5 grid, discrete) | ✅ Implemented & tested | `adaptive-rl env inspect gridworld` |
| Continuous 2D Navigation environment | ✅ Implemented & tested | `adaptive-rl env inspect navigation` |
| Traffic Signal Optimization environment | ✅ Implemented & tested | `adaptive-rl env inspect traffic` |
| Autonomous 3D Drone navigation | ✅ Implemented & tested | `adaptive-rl env inspect drone` |
| Disturbed Drone (wind, battery, obstacles) | ✅ Implemented & tested | `adaptive-rl env inspect drone_disturbed` |
| PPO training (Stable-Baselines3 wrapper) | ✅ Implemented & tested | `adaptive-rl train --config configs/gridworld_ppo.yaml` |
| SAC training (Stable-Baselines3 wrapper) | ✅ Implemented & tested | `adaptive-rl train --config configs/drone_sac.yaml` |
| Evaluation engine (multi-episode benchmark) | ✅ Implemented & tested | `adaptive-rl evaluate --config ...` |
| Curriculum learning (staged progression) | ✅ Implemented & tested | `adaptive-rl train --config configs/curriculum_gridworld.yaml` |
| Generalization benchmark (disjoint seeds) | ✅ Implemented & tested | `adaptive-rl generalization --config configs/generalization_gridworld.yaml` |
| A\* classical planner (GridWorld baseline) | ✅ Implemented & tested | `adaptive-rl benchmark-planners --planner astar` |
| RRT\* classical planner (Navigation baseline) | ✅ Implemented & tested | `adaptive-rl benchmark-planners --planner rrt` |
| Reproducibility: metadata.json per run | ✅ Implemented | Automatic on every `adaptive-rl train` |
| Reproducibility: episodes.csv per run | ✅ Implemented | Automatic on every `adaptive-rl train` |

### 2.2 PLANNED — Not Yet Implemented

| Component | Status | Notes |
| :--- | :--- | :--- |
| True continual/adaptive RL (Task A→B→A forgetting measurement) | 🔵 Planned | Requires task-switching loop and plasticity metrics |
| Multi-task learning (single policy, multiple envs simultaneously) | 🔵 Planned | Requires multi-env VecEnv extension |
| Ablation framework (systematic hyperparameter grid) | 🔵 Planned | Requires sweep infrastructure |
| OOD generalization (different obstacle counts, grid sizes) | 🔵 Planned | Current generalization is same-distribution, different seeds only |
| Learning curve visualization | 🔵 Planned | episodes.csv now provides the data |
| Drone realistic physics (aerodynamics, rotor dynamics) | 🔵 Planned | Current implementation is kinematic (position + velocity only) |

### 2.3 KNOWN LIMITATIONS (Honesty Audit)

1. **"Generalization" is within-distribution**: Train seeds [1000..1015) vs test seeds [2000..2015)
   are drawn from the *same distribution* (same grid size, same obstacle count, same start/goal positions).
   True OOD generalization (e.g. train on 4 obstacles, test on 12 obstacles) is **not implemented**.

2. **"Adaptive" RL is curriculum learning, not true continual RL**: The system does not perform
   Task A→B→A sequence measurement or catastrophic forgetting quantification.

3. **Drone physics is kinematic**: The drone environments use direct position/velocity updates.
   Realistic aerodynamics (rotor inertia, drag, motor model) are **not implemented**.

4. **No actual benchmark numbers below**: All benchmark table rows from previous versions of
   this document contained fabricated numbers (e.g. "92.0% success"). Those numbers were never
   produced by an actual experiment. Run the commands in Section 3 to obtain real numbers.

---

## 3. Generalization Benchmark Protocol

### 3.1 Hypothesis

RL agents trained on procedurally generated environments learn policies grounded in spatial
representations. When evaluated on completely unseen environment layouts (different seeds →
different obstacle placements), trained agents will transfer core navigation capabilities
but will exhibit a measurable **generalization gap**:

$$\Delta_{\text{success}} = \text{Success}_{\text{train}} - \text{Success}_{\text{unseen}}$$
$$\Delta_{\text{reward}} = \text{Reward}_{\text{train}} - \text{Reward}_{\text{unseen}}$$

### 3.2 How to Run

```bash
# Train a policy (short run for demonstration)
adaptive-rl train --config configs/generalization_gridworld.yaml

# Run generalization benchmark (uses the saved model)
adaptive-rl generalization \
    --config configs/generalization_gridworld.yaml \
    --train-count 15 --test-count 15 \
    --train-start 1000 --test-start 2000 \
    --output-report experiments/results/generalization/gridworld_gen_report.json
```

### 3.3 Evaluation Protocol

- **Training Distribution** $D_{\text{train}}$: Seeds $[1000, 1015)$ — 15 disjoint layouts.
- **Unseen Test Distribution** $D_{\text{test}}$: Seeds $[2000, 2015)$ — 15 disjoint layouts.
- **Integrity Check**: `GeneralizationDistribution` enforces $|D_{\text{train}} \cap D_{\text{test}}| = 0$.
- **Evaluation Mode**: Deterministic (`deterministic=True`) over all seeds.

### 3.4 Reported Metrics

| Metric | Description |
| :--- | :--- |
| Success Rate (train / test) | Episodes where agent reached goal |
| Collision Rate (train / test) | Episodes ending in collision |
| Mean Reward (train / test) | Mean cumulative episodic reward |
| Mean Episode Length | Steps until termination |
| Generalization Gap $\Delta$ | `train_metric - test_metric` |
| Relative Success Retention | `test_success / train_success` |

### 3.5 Current Limitation

The test seeds sample from the *same parameter distribution* as training (same grid size, same
obstacle count). Full OOD evaluation (different obstacle counts, grid sizes, or goal positions)
is not yet implemented and is tracked as a planned improvement.

---

## 4. Classical Planner Benchmarks

A\* (for GridWorld) and RRT\* (for continuous Navigation) provide **optimal or near-optimal**
baseline comparisons against RL policies.

### 4.1 Run the Benchmark

```bash
# A* vs RL on GridWorld
adaptive-rl benchmark-planners \
    --config configs/benchmark_planners_gridworld.yaml \
    --planner astar \
    --episodes 20 \
    --start-seed 100 \
    --output-report experiments/results/astar_gridworld_benchmark.json

# RRT* vs RL on Navigation
adaptive-rl benchmark-planners \
    --config configs/benchmark_planners_navigation.yaml \
    --planner rrt_star \
    --episodes 20 \
    --start-seed 100 \
    --output-report experiments/results/rrt_navigation_benchmark.json
```

### 4.2 What to Expect

- **A\*** is guaranteed to find the optimal (shortest-step) path on a discrete grid.
  A trained PPO policy will generally underperform A\* on success rate and path optimality
  unless trained for many steps and evaluated deterministically.
- **RRT\*** is near-optimal in continuous space. A trained SAC policy may outperform RRT\*
  in environments with dynamic obstacles or when given enough training time, as RRT\* plans
  offline and cannot react to changes.

---

## 5. Reproducibility

Every training run automatically saves:

```
experiments/results/
    models/<name>_final.zip           # Model weights
    checkpoints/<name>/checkpoint_step_*.zip
    metadata/<name>_metadata.json     # Full provenance record
    metadata/<name>_episodes.csv      # Per-episode reward/length/outcome
```

The `metadata.json` contains: seed, full config snapshot, Python version, platform, timestamps,
duration, and all metric values. This is sufficient to reproduce the exact experiment.
