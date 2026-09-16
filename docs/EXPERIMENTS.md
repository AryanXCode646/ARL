# AdaptiveRL Experiment Registry

This document records the official benchmark runs, configuration parameters, and experimental results as AdaptiveRL evolves across development milestones.

---

## Experiment Registry Guidelines

Each documented experiment must follow this template:
1. **Experiment ID:** Unique identifier matching configuration filename.
2. **Phase:** Milestone during which the experiment was executed.
3. **Environment:** Registered environment name and parameters.
4. **Algorithm:** Algorithm name, hyperparameters, and seed.
5. **Training Budget:** Total timesteps and wall-clock execution time.
6. **Measured Results:**
   - Success Rate (%)
   - Collision Rate (%)
   - Mean Episodic Reward +/- Std
   - Mean Episode Length
7. **Artifacts:** Path to saved weights, logs, and evaluation metrics.
8. **Reproduction Command:** Exact CLI command to reproduce the experiment.

---

## Logged Experiments

| Experiment ID | Phase | Environment | Algorithm | Timesteps | Success Rate | Mean Reward | Date |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Pending Phase 4* | Phase 4 | GridWorld | PPO | 50,000 | TBD | TBD | Pending |
| *Pending Phase 6* | Phase 6 | Navigation-2D | PPO | 100,000 | TBD | TBD | Pending |
| *Pending Phase 9* | Phase 9 | Drone-3D | PPO | 250,000 | TBD | TBD | Pending |

---

## Current Status (Phase 1)
* Phase 1 focuses on repository setup, packaging, configuration schemas, and interface skeletons.
* Concrete reinforcement learning experiments will commence in **Phase 4** (GridWorld PPO Training Engine).
