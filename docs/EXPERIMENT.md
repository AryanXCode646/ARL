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

## 5. Reproducibility Guarantee

To independently reproduce the identical metrics on any student laptop:
```bash
# 1. Clean environment install
pip install -e ".[all]"

# 2. Train with seed 42 (25k timesteps, ~25s on CPU)
adaptive-rl train --config configs/drone_ppo_demo.yaml

# 3. Evaluate PPO vs Random Baseline
adaptive-rl evaluate \
  --config configs/drone_ppo_demo.yaml \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 20 \
  --compare-random

# 4. Run Obstacle-Density Experiment
adaptive-rl experiment-density \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 10 \
  --seed 42
```
All outputs are saved directly to `artifacts/evaluation.json` and `artifacts/obstacle_density_experiment.json`.

---

## 6. Reward-Function Ablation Study

### Rationale
In continuous 3D drone navigation, reward shaping balances progress incentive against collision aversion, flight time, and control effort. Without structured empirical ablations, multi-term reward formulations remain unvalidated heuristics that can inadvertently induce suboptimal failure modes (e.g. hovering defensively to avoid effort penalties, or rushing blindly into obstacles due to excessive step penalties).

The reward-function ablation study isolates the contribution of each reward component under strictly controlled conditions.

### The Five Reward Terms
The continuous 3D navigation reward function comprises five distinct terms:
1. **$w_{\text{progress}} \times (d_{t-1} - d_t)$**: Distance-progress reward attracting the drone toward the target waypoint ($w_{\text{progress}} = 2.0$).
2. **$\text{goal\_reward}$**: Sparse terminal bonus granted upon reaching the goal within target radius ($+100.0$).
3. **$\text{collision\_reward}$**: Terminal penalty assessed upon collision with obstacles or arena boundary ($-100.0$).
4. **$\text{step\_penalty}$**: Constant time penalty incurred at each step to incentivize efficient paths ($-0.05$).
5. **$-w_{\text{effort}} \times \|\mathbf{a}_t\|_2^2$**: Smoothness/effort penalty minimizing excessive actuator chatter ($w_{\text{effort}} = 0.01$).

### The Four Controlled Variants
The study evaluates an exact four-variant progression:

| Variant | Variant Name | Progress Weight ($w_{\text{prog}}$) | Goal Reward | Collision Penalty | Step Penalty ($c_{\text{step}}$) | Action Effort Weight ($w_{\text{effort}}$) | Description |
|---|---|---|---|---|---|---|---|
| **A** | **Progress Only** | 2.0 | +100.0 | 0.0 | 0.0 | 0.0 | Pure progress delta; no step, effort, or collision penalties. |
| **B** | **Progress + Collision** | 2.0 | +100.0 | -100.0 | 0.0 | 0.0 | Adds terminal collision avoidance incentive. |
| **C** | **Progress + Collision + Step** | 2.0 | +100.0 | -100.0 | -0.05 | 0.0 | Adds step time penalty to encourage rapid goal-seeking. |
| **D** | **Full Baseline** | 2.0 | +100.0 | -100.0 | -0.05 | 0.01 | Full standard formulation (matches default environment). |

### Experimental Controls & Methodology
To ensure rigorous empirical comparison:
- **Identical Training Budget**: Each variant is trained for exactly the same number of timesteps (default 25,000 steps).
- **Identical PPO Hyperparameters**: Policy network architecture (`MlpPolicy`), learning rate ($3 \times 10^{-4}$), discount factor ($\gamma = 0.99$), batch size (64), rollout steps ($n_{\text{steps}} = 1024$), and clip range ($0.2$) are held strictly constant.
- **Identical Random Initialization**: Each variant begins from the identical base seed (default `42`), enforcing identical initial neural network weight states and identical environment procedural generation sequences.
- **Identical Held-Out Evaluation**: All four trained policies are evaluated on the identical held-out test seed distribution (`eval_seed = base_seed + 1000`) over a fixed evaluation episode count (default 20 episodes).
- **Identical Environment Geometry**: Flight arena bounds ($30\text{ m} \times 30\text{ m} \times 15\text{ m}$), obstacle count (4), start/goal coordinates, and kinematics remain identical.

### Metric Definitions
- **Success Rate**: Fraction of evaluation episodes terminating inside the calibrated target radius ($\le 1.5\text{ m}$).
- **Collision Rate**: Fraction of evaluation episodes terminating due to contact with spherical obstacles or boundary walls.
- **Timeout Rate**: Fraction of evaluation episodes truncated by reaching the maximum step limit ($max\_steps = 200$) without arrival or collision.
- **Mean Reward**: Average cumulative return per episode on the standardized held-out benchmark.
- **Mean Path Efficiency**: Ratio of straight-line distance ($D_0 = \|\mathbf{g} - \mathbf{p}_0\|$) to actual path length ($L = \sum_t \|\mathbf{p}_{t+1} - \mathbf{p}_t\|$), clamped to $[0.0, 1.0]$.
- **Convergence Speed**: The first training timestep at which the intermediate held-out evaluation success rate strictly exceeds $70\%$ ($> 0.70$). If the $70\%$ threshold is never reached during training, convergence speed is recorded as `null` (`not reached`).

### How to Run the Experiment
Run the ablation study via the command line:

```bash
# Standard 25,000-timestep research benchmark across all 4 variants
adaptive-rl experiment-ablation --timesteps 25000 --episodes 20 --seed 42

# Fast verification run (e.g. for testing)
adaptive-rl experiment-ablation --timesteps 500 --episodes 5 --seed 42
```

Outputs are automatically exported to:
- `artifacts/benchmarks/reward_ablation.json` (detailed per-variant results and configuration metadata)
- `artifacts/benchmarks/reward_ablation.csv` (tabular benchmark data for analysis)
