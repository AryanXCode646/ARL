"""Tests verifying environment registration, factory behavior, and Gymnasium compatibility."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from adaptive_rl.environments.metadata import EnvironmentMetadata
from adaptive_rl.environments.registry import (
    EnvironmentRegistry,
    RegistryError,
    list_environments,
    make_env,
    register,
)
from adaptive_rl.environments.testing import DummyTestEnv


def test_dummy_env_gymnasium_checker() -> None:
    """Verify DummyTestEnv complies 100% with Farama Gymnasium's env_checker."""
    env = DummyTestEnv(step_limit=15)
    # check_env raises exceptions or warnings if the contract is violated
    check_env(env)
    env.close()


def test_dummy_env_reset_contract() -> None:
    """Verify reset() returns (obs, info) and honors deterministic seeding."""
    env = DummyTestEnv(step_limit=10)

    obs1, info1 = env.reset(seed=42)
    assert isinstance(obs1, np.ndarray)
    assert obs1.shape == (2,)
    assert isinstance(info1, dict)
    assert "step" in info1

    # Same seed must yield identical initial observation
    obs2, info2 = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)

    # Different seed should yield different initial observation
    obs3, _ = env.reset(seed=999)
    assert not np.array_equal(obs1, obs3)
    env.close()


def test_dummy_env_step_contract() -> None:
    """Verify step() returns (obs, reward, terminated, truncated, info) with correct semantics."""
    env = DummyTestEnv(step_limit=3, reward_step=2.5)
    env.reset(seed=123)

    # Step 1
    obs, reward, terminated, truncated, info = env.step(1)
    assert isinstance(obs, np.ndarray)
    assert reward == 2.5
    assert not terminated
    assert not truncated
    assert info["step"] == 1
    assert info["action_taken"] == 1

    # Step 2
    obs, reward, terminated, truncated, info = env.step(0)
    assert not terminated

    # Step 3 (step_limit reached)
    obs, reward, terminated, truncated, info = env.step(1)
    assert terminated
    assert not truncated
    assert info["step"] == 3
    env.close()


def test_dummy_env_invalid_action_raises() -> None:
    """Verify passing an out-of-bounds action raises ValueError."""
    env = DummyTestEnv()
    env.reset()
    with pytest.raises(ValueError, match="invalid for action space"):
        env.step(99)  # valid actions are only 0 and 1
    env.close()


def test_registry_registration_and_retrieval() -> None:
    """Verify registering a factory and retrieving it."""
    reg = EnvironmentRegistry()
    meta = EnvironmentMetadata(
        name="test_dummy",
        description="A dummy test environment",
        observation_type="box",
        action_type="discrete",
    )
    reg.register("test_dummy", lambda **kw: DummyTestEnv(**kw), metadata=meta)

    assert "test_dummy" in reg.list_environments()
    factory = reg.get("test_dummy")
    env = factory()
    assert isinstance(env, DummyTestEnv)
    env.close()

    retrieved_meta = reg.get_metadata("test_dummy")
    assert retrieved_meta.name == "test_dummy"
    assert retrieved_meta.observation_type == "box"
    assert retrieved_meta.action_type == "discrete"


def test_registry_metadata_from_dict() -> None:
    """Verify metadata can be registered using a standard dictionary."""
    reg = EnvironmentRegistry()
    reg.register(
        "dict_meta_env",
        lambda: DummyTestEnv(),
        metadata={"description": "Metadata from dict", "observation_type": "box"},
    )
    meta = reg.get_metadata("dict_meta_env")
    assert isinstance(meta, EnvironmentMetadata)
    assert meta.name == "dict_meta_env"
    assert meta.description == "Metadata from dict"


def test_duplicate_registration_raises_error() -> None:
    """Verify registering the same environment name twice raises RegistryError."""
    reg = EnvironmentRegistry()
    reg.register("env_name", lambda: DummyTestEnv())

    with pytest.raises(RegistryError, match="already registered"):
        reg.register("env_name", lambda: DummyTestEnv())


def test_unknown_environment_raises_error() -> None:
    """Verify retrieving an unknown environment raises RegistryError with available environments."""
    reg = EnvironmentRegistry()
    reg.register("alpha_env", lambda: DummyTestEnv())

    with pytest.raises(RegistryError, match="Unknown environment 'beta_env'"):
        reg.get("beta_env")


def test_empty_or_invalid_name_raises_error() -> None:
    """Verify empty or non-string environment names raise RegistryError."""
    reg = EnvironmentRegistry()
    with pytest.raises(RegistryError, match="must be a non-empty string"):
        reg.register("", lambda: DummyTestEnv())

    with pytest.raises(RegistryError, match="must be callable"):
        reg.register("bad_factory", "not_callable")  # type: ignore


def test_factory_behavior_create_with_kwargs() -> None:
    """Verify factory create() instantiates environment and forwards kwargs."""
    reg = EnvironmentRegistry()
    reg.register("custom_dummy", lambda **kw: DummyTestEnv(**kw))

    env = reg.create("custom_dummy", step_limit=5, reward_step=10.0)
    assert isinstance(env, DummyTestEnv)
    assert env.step_limit == 5
    assert env.reward_step == 10.0
    env.close()


def test_invalid_environment_configuration_raises_error() -> None:
    """Verify passing invalid parameters to the factory raises RegistryError."""
    reg = EnvironmentRegistry()
    reg.register("test_env", lambda **kw: DummyTestEnv(**kw))

    with pytest.raises(RegistryError, match="Failed to instantiate registered environment"):
        reg.create("test_env", invalid_kwarg_that_does_not_exist="fail")


def test_factory_returning_non_gym_env_raises_error() -> None:
    """Verify that a factory returning a non-Gymnasium object raises RegistryError."""
    reg = EnvironmentRegistry()
    reg.register("not_an_env", lambda: "plain_string")

    with pytest.raises(RegistryError, match="not an instance of gymnasium.Env"):
        reg.create("not_an_env")


def test_gymnasium_fallback_resolution() -> None:
    """Verify that create() falls back to resolving standard Gymnasium environments."""
    reg = EnvironmentRegistry()
    # CartPole-v1 is part of standard Gymnasium
    env = reg.create("CartPole-v1")
    assert isinstance(env, gym.Env)
    obs, info = env.reset(seed=42)
    assert obs is not None
    env.close()


def test_global_convenience_functions() -> None:
    """Verify module-level convenience functions (register, make_env, list_environments)."""
    # Clean global registry for isolated test
    from adaptive_rl.environments.registry import registry

    registry.clear()

    register("global_dummy", lambda: DummyTestEnv(), metadata={"observation_type": "box"})
    assert "global_dummy" in list_environments()

    env = make_env("global_dummy")
    assert isinstance(env, DummyTestEnv)
    env.close()

    # Clean up after test
    registry.clear()
