"""Tests verifying environment registration, retrieval, and error handling."""

import pytest
from adaptive_rl.environments.registry import EnvironmentRegistry, RegistryError


def test_registry_registration_and_retrieval() -> None:
    """Verify environments can be registered and retrieved by factory."""
    reg = EnvironmentRegistry()

    def dummy_factory(**kwargs):
        return "dummy_instance"

    reg.register("dummy_env", dummy_factory, metadata={"obs_type": "discrete"})

    assert "dummy_env" in reg.list_environments()
    factory = reg.get("dummy_env")
    assert factory() == "dummy_instance"
    metadata = reg.get_metadata("dummy_env")
    assert metadata["obs_type"] == "discrete"


def test_duplicate_registration_raises_error() -> None:
    """Verify attempting to register the same environment name twice raises RegistryError."""
    reg = EnvironmentRegistry()
    reg.register("my_env", lambda: None)

    with pytest.raises(RegistryError, match="already registered"):
        reg.register("my_env", lambda: None)


def test_unknown_environment_raises_error() -> None:
    """Verify retrieving an unregistered environment raises RegistryError with available list."""
    reg = EnvironmentRegistry()
    reg.register("env_a", lambda: None)

    with pytest.raises(RegistryError, match="Unknown environment 'non_existent'"):
        reg.get("non_existent")


def test_empty_or_invalid_name_raises_error() -> None:
    """Verify registering an empty or non-string environment name raises RegistryError."""
    reg = EnvironmentRegistry()
    with pytest.raises(RegistryError, match="must be a non-empty string"):
        reg.register("", lambda: None)


def test_registry_clear() -> None:
    """Verify clear() removes all registered environments."""
    reg = EnvironmentRegistry()
    reg.register("env_1", lambda: None)
    assert len(reg.list_environments()) == 1

    reg.clear()
    assert len(reg.list_environments()) == 0
