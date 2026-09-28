"""Benchmark simulation and inference throughput (steps/sec)."""

import time

import numpy as np

from adaptive_rl.environments.drone import DroneNavigation3DEnv


def main():
    env = DroneNavigation3DEnv()
    env.reset(seed=42)
    num_steps = 10000

    start = time.perf_counter()
    for _ in range(num_steps):
        action = np.zeros(3, dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    duration = time.perf_counter() - start

    fps = num_steps / duration
    print(f"Simulation Throughput: {fps:.1f} steps/second ({num_steps} steps in {duration:.3f}s)")


if __name__ == "__main__":
    main()
