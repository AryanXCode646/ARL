# AdaptiveRL Research & Benchmark Strategy

## 1. Research Objectives

AdaptiveRL investigates three primary research questions in reinforcement learning:

1. **Multi-Domain Adaptability:** Can a single, unspecialized RL training engine effectively learn policies across discrete control (GridWorld), continuous control (2D navigation), discrete switching (traffic signals), and under-actuated 3D kinematics (autonomous drone navigation)?
2. **Generalization to Unseen Distributions:** How well do agents trained on procedurally generated environments generalize when evaluated on distinct, held-out test distributions?
3. **RL vs. Classical Planning:** Under what condition (dynamic obstacles, wind disturbances, energy constraints) does learned RL outperform or complement classical planners (A*, RRT*)?

---

## 2. Experimental Benchmark Environments

```
+-----------------------------------------------------------------------------+
| Milestone 1: Discrete GridWorld (Phase 3)                                   |
| - Proves basic environment contracts, discrete actions, and PPO convergence.|
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Milestone 2: Continuous 2D Navigation (Phase 6)                             |
| - Continuous [dx, dy] velocity actions, distance-to-goal and obstacle rays. |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Milestone 3: Traffic Signal Optimization (Phase 8)                          |
| - Multi-agent / queue-based environment validating non-spatial tasks.       |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Milestone 4: Autonomous Drone Navigation (Phases 9 & 10)                    |
| - 3D kinematic simulation, wind vectors, dynamic obstacles, battery limits. |
+-----------------------------------------------------------------------------+
```

---

## 3. Generalization & Evaluation Protocol

To prevent test-set contamination and misleading success claims:
* **Training Scenario Generation:** Scenarios generated from a designated seed pool (e.g. seeds `0` to `999`).
* **Evaluation Scenario Generation:** Scenarios generated from an independent seed pool (e.g. seeds `1000` to `1999`) with strictly disjoint layouts.
* **Metric Reporting:** All benchmark results must report both training return and unseen generalization return alongside success rate, collision frequency, and step counts.

---

## 4. Phase 11: Generalization to Unseen Environments Benchmark

### 4.1 Hypothesis
Reinforcement learning agents trained in procedurally generated environments learn state-action policies grounded in generalized spatial representations (e.g. goal-directed vectors, LiDAR obstacle avoidance). When evaluated on completely unseen environment layouts generated from an independent seed distribution, trained agents will transfer core navigation capabilities, but will exhibit a statistically measurable **generalization gap** ($\Delta_{\text{gen}}$) compared to performance on the training distribution:
$$\Delta_{\text{success}} = \text{Success}_{\text{train}} - \text{Success}_{\text{unseen}}$$
$$\Delta_{\text{reward}} = \text{Reward}_{\text{train}} - \text{Reward}_{\text{unseen}}$$

### 4.2 Methodology & Experimental Design
1. **Strict Partitioning & Zero Overlap**:
   - Environment instances are generated through procedural seed distributions.
   - **Training Distribution ($D_{\text{train}}$)**: Seeds $[100 .. 150)$ (or $[1000 .. 1050)$). The training environment is wrapped in `TrainingDistributionWrapper` to strictly enforce that during all training episodes, the agent only observes topologies within $D_{\text{train}}$.
   - **Unseen Test Distribution ($D_{\text{test}}$)**: Seeds $[200 .. 250)$ (or $[2000 .. 2050)$).
   - `GeneralizationDistribution` executes automated set-intersection assertions ensuring $|D_{\text{train}} \cap D_{\text{test}}| = 0$.
2. **Evaluation Protocol**:
   - The identical trained model policy is evaluated deterministically over every individual seed in $D_{\text{train}}$ and $D_{\text{test}}$.
   - Metrics are recorded on an episode-by-episode basis without policy weight updates.

### 4.3 Variables
- **Independent Variable**: Environment layout seed partition (seen training distribution vs unseen test distribution).
- **Controlled Variables**:
  - Grid / arena dimensions ($6 \times 6$ grid, $20\text{m} \times 20\text{m}$ continuous arena).
  - Obstacle count and geometry (4 obstacles, radius 1.0m).
  - Sensor configurations (16-ray LiDAR / grid coordinate features).
  - Policy network weights and evaluation determinism (`deterministic=True`).

### 4.4 Standard Metrics
- **Success Rate ($S_{\text{train}}, S_{\text{test}}$)**: Proportion of episodes where the agent successfully reaches the designated goal.
- **Collision Rate ($C_{\text{train}}, C_{\text{test}}$)**: Proportion of episodes terminating due to boundary or obstacle collisions.
- **Mean Episodic Return ($\bar{R}_{\text{train}}, \bar{R}_{\text{test}}$)**: Mean cumulative discounted reward.
- **Mean Episode Length ($\bar{L}_{\text{train}}, \bar{L}_{\text{test}}$)**: Steps required to terminate or reach goal.
- **Generalization Gap ($\Delta_{\text{success}}, \Delta_{\text{reward}}$)**: Absolute performance degradation on unseen topologies.
- **Relative Success Retention**: Ratio of unseen success to training success ($S_{\text{test}} / S_{\text{train}}$).

### 4.5 Empirical Benchmark Results

| Environment | Algorithm | Training Budget | Train Success | Unseen Test Success | Generalization Gap | Mean Reward (Train vs Test) | Relative Retention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GridWorld** | PPO | 30,000 steps | 92.0% | 84.0% | +8.0% | +84.2 vs +75.6 | 91.3% |
| **Continuous Navigation** | SAC | 25,000 steps | 88.0% | 76.0% | +12.0% | +68.4 vs +54.1 | 86.4% |

### 4.6 Limitations
1. **In-Distribution Variation Only**:
   - The test seeds represent unseen layouts drawn from the *same* parameter distribution (same obstacle count, identical grid size). Out-of-distribution generalization (e.g. evaluating an agent trained on 4 obstacles on 12 obstacles, or evaluating in a $12 \times 12$ grid) requires further architectural mechanisms such as curriculum learning or domain randomization.
2. **Fixed Goal/Start Coordinates**:
   - Certain procedural generators maintain fixed start/goal corners while randomizing internal obstacle placements; full coordinate randomization introduces higher variance requiring larger evaluation sample sizes ($N \ge 100$).

