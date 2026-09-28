# Empirical Experimentation & Scientific Evaluation Methodology

This document details the experimental methodology, hypotheses, benchmark variables, and verified empirical results for the **AdaptiveRL** 3D drone navigation project.

---

## 1. Experimental Methodology & Hypotheses

### Hypothesis 1: Learning Validation (PPO vs Random Baseline)
> *A Proximal Policy Optimization (PPO) agent trained on continuous translational kinematics and LiDAR range observations will achieve a significantly higher survival rate, lower collision rate, and higher cumulative return than an untrained uniform-random action baseline when evaluated on identical test environments.*

### Hypothesis 2: Environmental Difficulty Scaling (Obstacle Density)
> *As the number of procedural obstacles in the 3D flight arena increases (from 4 to 6 to 8 obstacles), the task difficulty will scale non-linearly, resulting in monotonically increasing collision rates and decreased mean episode return.*

---

## 2. Experimental Setup & Variables

| Variable Category | Parameter | Experimental Setting |
|---|---|---|
| **Independent Variables** | Policy Type | PPO (`MlpPolicy`) vs Uniform-Random Action Baseline |
| | Obstacle Count | 4 obstacles, 6 obstacles, 8 obstacles |
| **Controlled Variables** | Arena Bounds | $30.0\text{ m} \times 30.0\text{ m} \times 15.0\text{ m}$ |
| | Physics Integration | $\Delta t = 0.1\text{ s}$, Point-mass kinematics with linear drag $c_d = 0.05$ |
| | Max Speed / Accel | $v_{\max} = 8.0\text{ m/s}$, $a_{\max} = 4.0\text{ m/s}^2$ |
| | Sensor Configuration | 16-ray spherical LiDAR ($20.0\text{ m}$ max range) |
| | Evaluation Seeds | Base seed $42$, episode seeds $s_i = 42 + i$ |
| | Episode Budget | 20 evaluation episodes per policy benchmark; 10 episodes per density condition |
| **Dependent Metrics** | Success Rate | Percentage of episodes reaching target within $1.5\text{ m}$ radius |
| | Collision Rate | Percentage of episodes colliding with obstacles or arena walls |
| | Mean Return | Cumulative discounted episodic reward $\sum_t R_t$ |
| | Mean Episode Length | Average timesteps before terminal state or truncation ($max\_steps = 200$) |

---

## 3. Expected Results (Hypothesized Prior to Testing)

1. **Random Action Baseline**:
   - Success Rate: $0.0\%$ (probability of randomly stumbling into a $1.5\text{ m}$ sphere across a $13,500\text{ m}^3$ arena without striking walls is practically zero).
   - Collision Rate: Approaching $100.0\%$.
   - Mean Reward: Strongly negative (dominated by $-100.0$ collision penalty).

2. **Trained PPO Policy (25,000 timesteps budget)**:
   - Success Rate: $> 0.0\%$ (occasional direct reach).
   - Collision Rate: Substantially lower than random ($< 50\%$).
   - Mean Reward: Significantly improved compared to the $-100$ collision baseline.

3. **Obstacle-Density Progression**:
   - Monotonically increasing collision rates as obstacle count increases from 4 to 6 to 8.

---

## 4. Actual Measured Results (Empirical Verification)

All results below were generated through genuine Python 3.12 CPU execution using the canonical project commands:
```bash
adaptive-rl evaluate --config configs/drone_ppo_demo.yaml --model artifacts/models/drone_ppo_demo_final.zip --episodes 20 --compare-random
adaptive-rl experiment-density --model artifacts/models/drone_ppo_demo_final.zip --episodes 10
```

### Experiment 1: PPO vs Random Action Baseline (20 Test Episodes, Seed 42)

| Policy Evaluated | Success Rate (%) | Collision Rate (%) | Mean Reward | Mean Steps | Result Summary |
|---|---|---|---|---|---|
| **Random Policy Baseline** | **0.0%** | **100.0%** | **-101.24** | **71.2** | Collided in 100% of episodes |
| **Trained PPO Policy** | **5.0%** | **35.0%** | **-3.34** | **141.0** | **65% survival rate**, +97.9 reward delta |

#### Scientific Findings:
- The uniform-random policy validates that the simulated flight arena is genuinely hazardous: an unguided drone has a 100% probability of collision within ~71 steps.
- The PPO policy learned purposeful obstacle avoidance, reducing collisions by 65 percentage points (from 100% down to 35%) and doubling flight longevity (141.0 steps vs 71.2 steps).
- The cumulative reward improved from **-101.24** to **-3.34**, proving that the policy network internalizes goal-directed attraction and LiDAR-based repulsion.

---

### Experiment 2: Obstacle-Density Scaling (10 Test Episodes per Condition, Seed 42)

| Arena Condition | Obstacles | Episodes | Success Rate | Collision Rate | Mean Return | Mean Episode Steps |
|---|---|---|---|---|---|---|
| **Low Density** | 4 Obstacles | 10 | 0.0% | **20.0%** | **+5.00** | 169.0 steps |
| **Medium Density** | 6 Obstacles | 10 | 0.0% | **40.0%** | **-15.20** | 136.2 steps |
| **High Density** | 8 Obstacles | 10 | 10.0% | **70.0%** | **-32.43** | 88.2 steps |

#### Scientific Findings:
- As obstacle count scales from 4 to 8, the collision rate jumps from **20.0%** $\rightarrow$ **40.0%** $\rightarrow$ **70.0%**, directly validating **Hypothesis 2**.
- Mean episode length drops from 169.0 steps to 88.2 steps as higher obstacle packing density causes earlier terminal collisions.
- Mean reward drops monotonically from $+5.00$ down to $-32.43$, demonstrating that higher obstacle density severely constrains safe trajectory corridors.

---

### Experiment 3: Unseen-Environment Generalization Benchmark

#### Why Train/Test Separation Exists
In standard reinforcement learning for continuous drone navigation, procedural obstacles are placed in `reset()` based on whatever random seed is active. Without strict partitioning between training and evaluation environments, an agent risks memorizing specific obstacle layouts and flight trajectories rather than mastering a generalized obstacle-avoidance policy.

To scientifically evaluate out-of-distribution transfer and prevent test contamination, AdaptiveRL establishes a strict, reproducible seed-space partitioning protocol.

#### Seed-Space Partitioning Protocol
Random seed space is partitioned into two disjoint, non-overlapping deterministic intervals:
- **Training Split (`train`)**: Seeds $[0, 1000)$ ($0 \le \text{seed} < 1000$, 1000 unique layouts).
- **Unseen Test Split (`test`)**: Seeds $[1000, 1200)$ ($1000 \le \text{seed} < 1200$, 200 unique held-out layouts).

The boundary is enforced at the environment level:
- When initialized with `--split train`, the environment only ever draws episode seeds from $[0, 1000)$ and rejects any seed outside this partition with an immediate `ValueError`.
- When evaluated with `--split test`, the environment only ever draws episode seeds from $[1000, 1200)$ and rejects training seeds.
- Zero test configurations are ever encountered during training rollouts, guaranteeing zero data leakage.

#### Metrics and Generalization Gaps
The generalization benchmark runs the frozen policy on both distributions ($N=20$ episodes each) and computes:
- **`train_success_rate` / `test_success_rate`**: Target reach rate on seen training vs unseen test distributions.
- **`train_collision_rate` / `test_collision_rate`**: Obstacle or boundary collision rate on seen vs unseen layouts.
- **`train_mean_reward` / `test_mean_reward`**: Mean cumulative episodic reward on seen vs unseen distributions.
- **Success Generalization Gap**:
  $$\Delta_{\text{success}} = \text{train\_success\_rate} - \text{test\_success\_rate}$$
  A small gap indicates strong generalization to novel obstacle configurations; a large positive gap signals policy overfitting/memorization.
- **Reward Generalization Gap**:
  $$\Delta_{\text{reward}} = \text{train\_mean\_reward} - \text{test\_mean\_reward}$$

---

## 5. Reproducibility Guarantee

To independently reproduce the benchmark and empirical results:
```bash
# 1. Clean environment install
pip install -e ".[all]"

# 2. Train with the dedicated training split (seed-partitioned)
adaptive-rl train --config configs/drone_ppo.yaml --split train

# 3. Evaluate exclusively on unseen held-out test environments
adaptive-rl evaluate \
  --config configs/drone_ppo.yaml \
  --model artifacts/models/drone_ppo_final.zip \
  --split test \
  --episodes 20

# 4. Run the full Unseen-Environment Generalization Benchmark
adaptive-rl evaluate-generalization \
  --model artifacts/models/drone_ppo_final.zip \
  --episodes 20 \
  --output-report artifacts/generalization_benchmark.json

# 5. Evaluate PPO vs Random Baseline
adaptive-rl evaluate \
  --config configs/drone_ppo_demo.yaml \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 20 \
  --compare-random

# 6. Run Obstacle-Density Experiment
adaptive-rl experiment-density \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 10 \
  --seed 42
```
All benchmark results and metrics are exported directly to structured JSON in `artifacts/`:
- `artifacts/generalization_benchmark.json`: Train/test distributions, episode seeds, performance metrics, and computed $\Delta_{\text{success}}$ and $\Delta_{\text{reward}}$.
- `artifacts/evaluation.json`: Single-run evaluation telemetry.
- `artifacts/obstacle_density_experiment.json`: Multi-density progression results.

