"""Standalone script demonstrating trained PPO policy on DroneNavigation3DEnv."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from adaptive_rl.environments.drone import DroneNavigation3DEnv


def main():
    parser = argparse.ArgumentParser(description="Demonstrate trained drone policy.")
    parser.add_argument("--model", type=str, default="artifacts/models/drone_ppo_demo_final.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = DroneNavigation3DEnv()
    obs, info = env.reset(seed=args.seed)

    model_path = Path(args.model)
    if model_path.exists():
        model = PPO.load(str(model_path))
        print(f"Loaded policy from {model_path}")
    else:
        print(f"Model {model_path} not found; running with random actions.")
        model = None

    print(f"Starting Drone Demo with Seed {args.seed}")
    print(f"Initial Position: {env.drone_state.position}, Target: {env.target}")
    total_reward = 0.0

    for step in range(1, 201):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        pos = env.drone_state.position
        d_goal = float(info.get("distance_to_goal", 0.0))
        d_obs = float(info.get("min_obstacle_distance", 0.0))
        print(
            f"Step {step:03d} | Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | d_goal: {d_goal:.2f}m | d_obs: {d_obs:.2f}m"
        )

        if terminated or truncated:
            reason = (
                "SUCCESS: Reached target!"
                if info.get("is_success")
                else "FAILED: Collision or boundary violation"
            )
            print(f"\nResult: {reason}")
            print(f"Total Steps: {step}, Total Reward: {total_reward:.2f}")
            break


if __name__ == "__main__":
    main()
