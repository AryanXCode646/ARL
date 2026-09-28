"""Tests for browser-based GUI visualizer and simulation playback."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from adaptive_rl.environments.drone import DroneNavigation3DEnv, ObstacleSphere3D
from adaptive_rl.gui.visualizer import (
    _generate_box_wireframe,
    _generate_sphere_surface,
    build_arena_3d_figure,
    get_available_models,
    run_drone_simulation_episode,
)


def test_sphere_surface_generator() -> None:
    """Verify parametric sphere coordinates generation."""
    x, y, z = _generate_sphere_surface(center=[5.0, 5.0, 5.0], radius=2.0, n_points=25)
    assert x.shape == (25, 25)
    assert y.shape == (25, 25)
    assert z.shape == (25, 25)
    # Check max radius range
    assert np.isclose(np.max(x), 7.0, atol=0.01)
    assert np.isclose(np.min(x), 3.0, atol=0.01)


def test_box_wireframe_generator() -> None:
    """Verify 12 edges generation for 3D bounding arena."""
    xs, ys, zs = _generate_box_wireframe(bounds=(30.0, 30.0, 15.0))
    # 12 edges * 3 items (start, end, None) = 36 items
    assert len(xs) == 36
    assert len(ys) == 36
    assert len(zs) == 36
    assert None in xs


def test_build_arena_3d_figure() -> None:
    """Verify complete 3D Plotly figure generation with drone, obstacles, and LiDAR."""
    bounds = (30.0, 30.0, 15.0)
    obstacles = [ObstacleSphere3D(center=np.array([15.0, 15.0, 7.0]), radius=2.0)]
    target = np.array([25.0, 25.0, 10.0])
    start = np.array([5.0, 5.0, 5.0])
    trajectory = [np.array([5.0, 5.0, 5.0]), np.array([6.0, 6.0, 5.5])]
    current_pos = np.array([6.0, 6.0, 5.5])
    current_vel = np.array([1.0, 1.0, 0.5])
    lidar_rays = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    lidar_ranges = [10.0, 15.0]

    fig = build_arena_3d_figure(
        bounds=bounds,
        obstacles=obstacles,
        target=target,
        start_pos=start,
        trajectory=trajectory,
        current_pos=current_pos,
        current_vel=current_vel,
        lidar_rays=lidar_rays,
        lidar_ranges=lidar_ranges,
        show_lidar=True,
    )

    assert isinstance(fig, go.Figure)
    trace_names = [trace.name for trace in fig.data if hasattr(trace, "name")]
    assert "Arena Boundary" in trace_names
    assert "Obstacle 1" in trace_names
    assert "Target Waypoint" in trace_names
    assert "Start Position" in trace_names
    assert "Flight Path" in trace_names
    assert "16-Ray LiDAR" in trace_names
    assert "Drone" in trace_names
    assert "Velocity Vector" in trace_names


def test_run_drone_simulation_episode() -> None:
    """Verify deterministic episode execution and trajectory data structure."""
    env = DroneNavigation3DEnv(num_obstacles=2, max_steps=15)
    result = run_drone_simulation_episode(env=env, model=None, seed=42)

    assert "trajectory" in result
    assert "steps" in result
    assert "total_reward" in result
    assert "outcome" in result
    assert len(result["steps"]) > 1
    assert result["seed"] == 42
    assert result["outcome"] in ["SUCCESS", "COLLISION / TIMEOUT"]

    first_step = result["steps"][0]
    assert first_step["step"] == 0
    assert "position" in first_step
    assert "lidar_ranges" in first_step
    assert len(first_step["lidar_ranges"]) == env.num_lidar_rays


def test_get_available_models(tmp_path: Path) -> None:
    """Verify scanning for model checkpoints."""
    models = get_available_models()
    assert isinstance(models, list)
