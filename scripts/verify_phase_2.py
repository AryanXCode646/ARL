"""Phase 2 Verification Script.

Demonstrates:
1. Registry creation
2. Environment registration
3. Environment discovery
4. Environment creation via factory
5. reset()
6. step()
"""

from __future__ import annotations

from adaptive_rl.environments.metadata import EnvironmentMetadata
from adaptive_rl.environments.registry import EnvironmentRegistry
from adaptive_rl.environments.testing import DummyTestEnv


def main() -> None:
    print("=== AdaptiveRL Phase 2 Verification ===")

    # 1. Registry creation
    print("\n1. Creating Environment Registry...")
    registry = EnvironmentRegistry()
    print("   ✓ Registry created successfully.")

    # 2. Environment registration
    print("\n2. Registering Environment 'demo_env'...")
    metadata = EnvironmentMetadata(
        name="demo_env",
        description="Demo environment for Phase 2 verification",
        observation_type="box",
        action_type="discrete",
        version="0.1.0",
        max_episode_steps=5,
    )
    registry.register(
        "demo_env",
        lambda **kw: DummyTestEnv(step_limit=5, **kw),
        metadata=metadata,
    )
    print("   ✓ 'demo_env' registered with metadata.")

    # 3. Environment discovery
    print("\n3. Discovering Registered Environments...")
    envs = registry.list_environments()
    print(f"   ✓ Registered environments: {envs}")
    meta = registry.get_metadata("demo_env")
    print(f"   ✓ Metadata for 'demo_env': {meta.model_dump()}")

    # 4. Environment creation
    print("\n4. Creating Environment Instance via Factory...")
    env = registry.create("demo_env", reward_step=5.0)
    print(f"   ✓ Instantiated environment: {type(env).__name__}")
    print(f"   ✓ Observation Space: {env.observation_space}")
    print(f"   ✓ Action Space: {env.action_space}")

    # 5. reset()
    print("\n5. Executing reset(seed=42)...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Initial observation: {obs}")
    print(f"   ✓ Reset info: {info}")

    # 6. step()
    print("\n6. Executing step(action=1)...")
    next_obs, reward, terminated, truncated, step_info = env.step(1)
    print(f"   ✓ Next observation: {next_obs}")
    print(f"   ✓ Reward: {reward}")
    print(f"   ✓ Terminated: {terminated}")
    print(f"   ✓ Truncated: {truncated}")
    print(f"   ✓ Step info: {step_info}")

    env.close()
    print("\n=== Phase 2 Verification COMPLETE: All contract steps verified! ===")


if __name__ == "__main__":
    main()
