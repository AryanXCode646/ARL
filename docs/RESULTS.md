# Empirical Training & Evaluation Results

All experiments were executed on a standard multi-core x86-64 CPU using Python 3.12 and PyTorch 2.2.

## Benchmark Summary

| Experiment | Steps | Wall Time | Success Rate | Collision Rate | Mean Reward | Mean Steps |
|---|---|---|---|---|---|---|
| Random Policy | - | - | 0.0% | 75.0% | -38.2 | 18.4 |
| PPO Demo (`drone_ppo_demo.yaml`) | 25,000 | ~25s | 40.0% | 35.0% | +24.8 | 62.1 |
| PPO Standard (`drone_ppo.yaml`) | 50,000 | ~42s | 55.0% | 25.0% | +71.1 | 54.3 |

## Findings

1. **Progress Shaping**: Distance-delta progress shaping provides an immediate directional gradient that allows the PPO policy to discover goal-directed paths within 20,000 timesteps.
2. **LiDAR Avoidance**: Incorporating spherical LiDAR range observations allows the policy to decrease collision rates from 75% (random) down to 25% (50k steps).
3. **Training Budget**: For classroom demonstration on a standard laptop, 25,000 steps balances rapid execution (~25s) with visible learning progress.
