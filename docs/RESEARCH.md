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
