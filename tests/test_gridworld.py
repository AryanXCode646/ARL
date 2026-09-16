"""Comprehensive tests for GridWorldEnv, BFS generator, and Gymnasium compliance."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from adaptive_rl.environments import make_env, register_default_environments
from adaptive_rl.environments.gridworld.generator import (
    generate_procedural_obstacles,
    is_path_available,
)
from adaptive_rl.environments.gridworld.grid import GridWorldEnv


def test_gridworld_gymnasium_checker() -> None:
    """Verify GridWorldEnv complies 100% with Farama Gymnasium's check_env."""
    env = GridWorldEnv(width=5, height=5, num_obstacles=3, max_steps=20)
    check_env(env)
    env.close()


def test_gridworld_spaces() -> None:
    """Verify observation space is Box(4,) and action space is Discrete(4)."""
    env = GridWorldEnv(width=6, height=5)
    assert env.action_space.n == 4
    assert env.observation_space.shape == (4,)
    assert env.observation_space.low.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert env.observation_space.high.tolist() == [1.0, 1.0, 1.0, 1.0]
    env.close()


def test_gridworld_deterministic_seeding() -> None:
    """Verify identical seeds produce identical obstacle layouts and initial observations."""
    env1 = GridWorldEnv(width=6, height=5, num_obstacles=4)
    obs1, info1 = env1.reset(seed=123)

    env2 = GridWorldEnv(width=6, height=5, num_obstacles=4)
    obs2, info2 = env2.reset(seed=123)

    np.testing.assert_array_equal(obs1, obs2)
    assert info1["obstacles"] == info2["obstacles"]

    # Different seed produces different layout
    _, info3 = env2.reset(seed=999)
    assert info1["obstacles"] != info3["obstacles"]

    env1.close()
    env2.close()


def test_gridworld_movement_in_all_directions() -> None:
    """Verify agent moves correctly in all 4 cardinal directions."""
    # Place agent at (2, 2) in a 5x5 grid with no obstacles
    env = GridWorldEnv(width=5, height=5, start_pos=(2, 2), goal_pos=(4, 4), num_obstacles=0)
    env.reset()

    # 0: UP -> (2, 1)
    obs, reward, term, trunc, info = env.step(0)
    assert info["agent_pos"] == (2, 1)
    assert reward == -1.0
    assert not term and not trunc

    # 1: DOWN -> (2, 2)
    obs, reward, term, trunc, info = env.step(1)
    assert info["agent_pos"] == (2, 2)

    # 2: LEFT -> (1, 2)
    obs, reward, term, trunc, info = env.step(2)
    assert info["agent_pos"] == (1, 2)

    # 3: RIGHT -> (2, 2)
    obs, reward, term, trunc, info = env.step(3)
    assert info["agent_pos"] == (2, 2)

    env.close()


def test_gridworld_boundary_collision_clamping() -> None:
    """Verify agent cannot move past grid boundaries."""
    # Start at top-left (0, 0)
    env = GridWorldEnv(width=4, height=4, start_pos=(0, 0), goal_pos=(3, 3), num_obstacles=0)
    env.reset()

    # Attempt UP from y=0 -> stays at (0, 0)
    _, reward, term, _, info = env.step(0)
    assert info["agent_pos"] == (0, 0)
    assert reward == -1.0
    assert not term

    # Attempt LEFT from x=0 -> stays at (0, 0)
    _, reward, term, _, info = env.step(2)
    assert info["agent_pos"] == (0, 0)
    assert reward == -1.0
    assert not term

    env.close()


def test_gridworld_goal_reaching_termination() -> None:
    """Verify reaching goal awards +100 and sets terminated=True."""
    # Start at (0, 0), goal at (1, 0)
    env = GridWorldEnv(width=4, height=4, start_pos=(0, 0), goal_pos=(1, 0), num_obstacles=0)
    env.reset()

    # Step RIGHT into goal
    obs, reward, terminated, truncated, info = env.step(3)
    assert reward == 100.0
    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["collision"] is False
    assert info["agent_pos"] == (1, 0)
    env.close()


def test_gridworld_obstacle_collision_termination() -> None:
    """Verify hitting obstacle awards -100 and sets terminated=True."""
    # Start at (0, 0), obstacle at (1, 0)
    env = GridWorldEnv(
        width=4,
        height=4,
        start_pos=(0, 0),
        goal_pos=(3, 3),
        fixed_obstacles=[(1, 0)],
        terminate_on_collision=True,
    )
    env.reset()

    # Step RIGHT into obstacle
    obs, reward, terminated, truncated, info = env.step(3)
    assert reward == -100.0
    assert terminated is True
    assert truncated is False
    assert info["collision"] is True
    assert info["success"] is False
    assert info["agent_pos"] == (0, 0)  # Remains at previous position
    env.close()


def test_gridworld_step_truncation() -> None:
    """Verify reaching max_steps without goal/collision sets truncated=True."""
    env = GridWorldEnv(width=5, height=5, max_steps=3, num_obstacles=0)
    env.reset()

    _, _, term1, trunc1, _ = env.step(1)  # step 1
    assert not term1 and not trunc1

    _, _, term2, trunc2, _ = env.step(1)  # step 2
    assert not term2 and not trunc2

    _, _, term3, trunc3, _ = env.step(1)  # step 3
    assert not term3
    assert trunc3 is True  # Max steps reached
    env.close()


def test_gridworld_invalid_action_raises_error() -> None:
    """Verify passing invalid action raises ValueError."""
    env = GridWorldEnv()
    env.reset()
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(4)
    env.close()


def test_gridworld_smallest_valid_grid() -> None:
    """Verify 2x2 grid functions properly."""
    env = GridWorldEnv(width=2, height=2, start_pos=(0, 0), goal_pos=(1, 1), num_obstacles=0)
    obs, info = env.reset()
    assert info["agent_pos"] == (0, 0)
    assert info["goal_pos"] == (1, 1)

    # Move RIGHT -> (1, 0)
    _, _, term, _, info = env.step(3)
    assert info["agent_pos"] == (1, 0)
    assert not term

    # Move DOWN -> (1, 1) Goal!
    _, reward, term, _, info = env.step(1)
    assert info["agent_pos"] == (1, 1)
    assert reward == 100.0
    assert term is True
    env.close()


def test_gridworld_invalid_dimensions_raises_error() -> None:
    """Verify dimensions less than 2x2 raise ValueError."""
    with pytest.raises(ValueError, match="at least 2x2"):
        GridWorldEnv(width=1, height=5)
    with pytest.raises(ValueError, match="at least 2x2"):
        GridWorldEnv(width=5, height=1)


def test_gridworld_invalid_positions_raises_error() -> None:
    """Verify positions outside grid raise ValueError."""
    with pytest.raises(ValueError, match="outside grid bounds"):
        GridWorldEnv(width=5, height=5, start_pos=(-1, 0))
    with pytest.raises(ValueError, match="outside grid bounds"):
        GridWorldEnv(width=5, height=5, goal_pos=(5, 5))


def test_gridworld_start_equals_goal() -> None:
    """Verify edge case where start equals goal."""
    env = GridWorldEnv(width=4, height=4, start_pos=(2, 2), goal_pos=(2, 2), num_obstacles=0)
    obs, info = env.reset()
    assert info["agent_pos"] == info["goal_pos"]
    assert info["distance_to_goal"] == 0
    env.close()


def test_gridworld_textual_renderer() -> None:
    """Verify textual ASCII rendering."""
    env = GridWorldEnv(
        width=4,
        height=3,
        start_pos=(0, 0),
        goal_pos=(3, 2),
        fixed_obstacles=[(1, 1)],
    )
    env.reset()
    rendered = env.render()
    assert rendered is not None
    lines = rendered.strip().split("\n")
    assert len(lines) == 3
    # Check that start/agent 'A' is at top-left and goal 'G' is at bottom-right
    assert lines[0].startswith("A")
    assert lines[1].split()[1] == "X"
    assert lines[2].endswith("G")
    env.close()


def test_bfs_path_checker_logic() -> None:
    """Verify BFS path validation detects open and blocked layouts accurately."""
    # Direct path available
    assert is_path_available(3, 3, (0, 0), (2, 2), set()) is True

    # Goal completely surrounded by obstacles
    blocked_obstacles = {(1, 2), (2, 1)}
    assert is_path_available(3, 3, (0, 0), (2, 2), blocked_obstacles) is False

    # Obstacles present but path snakes around them
    maze_obstacles = {(1, 0), (1, 1)}
    assert is_path_available(3, 3, (0, 0), (2, 0), maze_obstacles) is True


def test_procedural_generator_guarantees_path() -> None:
    """Verify procedural generation creates requested obstacle count and guarantees a path."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        obstacles = generate_procedural_obstacles(
            width=6,
            height=5,
            start=(0, 0),
            goal=(5, 4),
            num_obstacles=6,
            rng=rng,
        )
        assert len(obstacles) <= 6
        assert (0, 0) not in obstacles
        assert (5, 4) not in obstacles
        assert is_path_available(6, 5, (0, 0), (5, 4), obstacles) is True


def test_gridworld_registered_in_factory() -> None:
    """Verify 'gridworld' is discoverable and instantiable via the AdaptiveRL factory."""
    register_default_environments()
    env = make_env("gridworld", width=5, height=5, num_obstacles=2)
    assert isinstance(env, GridWorldEnv)
    obs, info = env.reset(seed=77)
    assert obs.shape == (4,)
    assert info["max_steps"] == 100
    env.close()
