"""Phase 3 GridWorld Verification Script.

Executes:
1. Gymnasium environment checker validation
2. GridWorld instantiation via factory (make_env("gridworld"))
3. reset() with deterministic seed
4. Textual grid rendering
5. Stepping with actions and reward verification
6. Termination on obstacle collision
7. Termination on goal reached
8. Comparison of layouts across multiple seeds
"""

from __future__ import annotations

from gymnasium.utils.env_checker import check_env

from adaptive_rl.environments.gridworld.grid import GridWorldEnv
from adaptive_rl.environments.registry import make_env


def main() -> None:
    print("=== AdaptiveRL Phase 3: GridWorld Verification ===\n")

    # 1. Gymnasium Env Checker
    print("1. Validating GridWorld against Farama Gymnasium check_env...")
    env = GridWorldEnv(width=6, height=5, num_obstacles=4, max_steps=50)
    check_env(env)
    print("   ✓ Farama Gymnasium check_env passed with 0 errors!")

    # 2. Factory creation
    print("\n2. Instantiating GridWorld via AdaptiveRL make_env('gridworld')...")
    factory_env = make_env("gridworld", width=6, height=5, num_obstacles=4)
    print(f"   ✓ Instantiated type: {type(factory_env).__name__}")
    print(f"   ✓ Action space: {factory_env.action_space}")
    print(f"   ✓ Observation space: {factory_env.observation_space}")

    # 3. Deterministic reset and rendering
    print("\n3. Resetting with seed 42 and rendering initial layout...")
    obs, info = factory_env.reset(seed=42)
    print(f"   ✓ Normalized Observation: {obs}")
    print(f"   ✓ Info: {info}")
    print("   Initial ASCII Grid Layout:")
    print("   " + "\n   ".join(factory_env.render().split("\n")))

    # 4. Multi-seed layout comparison
    print("\n4. Verifying Determinism Across Multiple Seeds...")
    env_seed1_a = make_env("gridworld", width=6, height=5, num_obstacles=4)
    _, info1_a = env_seed1_a.reset(seed=100)

    env_seed1_b = make_env("gridworld", width=6, height=5, num_obstacles=4)
    _, info1_b = env_seed1_b.reset(seed=100)

    env_seed2 = make_env("gridworld", width=6, height=5, num_obstacles=4)
    _, info2 = env_seed2.reset(seed=200)

    assert info1_a["obstacles"] == info1_b["obstacles"], (
        "Identical seeds must produce identical obstacles!"
    )
    assert info1_a["obstacles"] != info2["obstacles"], (
        "Different seeds must produce different obstacles!"
    )
    print(f"   ✓ Seed 100 Obstacles: {info1_a['obstacles']}")
    print(f"   ✓ Seed 200 Obstacles: {info2['obstacles']}")
    print("   ✓ Procedural generation is perfectly deterministic!")

    # 5. Stepping toward obstacle (collision test)
    print("\n5. Testing Collision Dynamics and Penalty...")
    fixed_env = GridWorldEnv(
        width=4,
        height=3,
        start_pos=(0, 0),
        goal_pos=(3, 2),
        fixed_obstacles=[(1, 0)],
        terminate_on_collision=True,
    )
    fixed_env.reset()
    print("   Fixed Layout for Collision Test:")
    print("   " + "\n   ".join(fixed_env.render().split("\n")))

    # Move RIGHT into obstacle at (1, 0)
    print("   Stepping RIGHT (action=3) directly into obstacle (1, 0)...")
    c_obs, c_reward, c_term, c_trunc, c_info = fixed_env.step(3)
    print(f"   ✓ Reward: {c_reward} (expected -100.0)")
    print(f"   ✓ Terminated: {c_term} | Collision: {c_info['collision']}")
    assert c_reward == -100.0 and c_term is True, (
        "Collision must yield -100.0 reward and terminate!"
    )

    # 6. Stepping into goal (success test)
    print("\n6. Testing Goal Reaching Dynamics and Success Reward...")
    goal_env = GridWorldEnv(
        width=3,
        height=3,
        start_pos=(0, 0),
        goal_pos=(1, 0),
        fixed_obstacles=[],
    )
    goal_env.reset()
    print("   Stepping RIGHT (action=3) directly into goal (1, 0)...")
    g_obs, g_reward, g_term, g_trunc, g_info = goal_env.step(3)
    print(f"   ✓ Reward: {g_reward} (expected +100.0)")
    print(f"   ✓ Terminated: {g_term} | Success: {g_info['success']}")
    assert g_reward == 100.0 and g_term is True, (
        "Goal reached must yield +100.0 reward and terminate!"
    )

    env.close()
    factory_env.close()
    fixed_env.close()
    goal_env.close()

    print("\n=== Phase 3 GridWorld Verification COMPLETE: All Quality Gates Passed! ===")


if __name__ == "__main__":
    main()
