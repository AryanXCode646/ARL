"""Ray-casting sensor mathematics and procedural obstacle generation for continuous 2D navigation."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def ray_cast_circle(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    circle_center: np.ndarray,
    circle_radius: float,
    max_range: float,
) -> float:
    """Compute distance along ray to intersection with a circle.

    Args:
        ray_origin: (2,) array [x, y] of ray source.
        ray_dir: (2,) unit direction vector [cos(theta), sin(theta)].
        circle_center: (2,) array [cx, cy] of circle center.
        circle_radius: Radius of the obstacle.
        max_range: Maximum sensor sensing range.

    Returns:
        float: Distance to intersection in [0.0, max_range]. Returns max_range if no hit.
    """
    m = ray_origin - circle_center
    b = float(np.dot(m, ray_dir))
    c = float(np.dot(m, m)) - circle_radius * circle_radius

    # If origin is inside circle, intersection distance is 0.0
    if c <= 0:
        return 0.0

    # Ray points away from circle
    if b > 0:
        return max_range

    discriminant = b * b - c
    if discriminant < 0:
        return max_range

    t = -b - math.sqrt(discriminant)
    if 0.0 <= t <= max_range:
        return t
    return max_range


def ray_cast_arena_boundaries(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    width: float,
    height: float,
    max_range: float,
) -> float:
    """Compute distance along ray to intersection with rectangular arena perimeter.

    Args:
        ray_origin: (2,) array [x, y] of ray source.
        ray_dir: (2,) unit direction vector [cos(theta), sin(theta)].
        width: Arena width (x in [0, width]).
        height: Arena height (y in [0, height]).
        max_range: Maximum sensor sensing range.

    Returns:
        float: Distance to perimeter in [0.0, max_range].
    """
    min_dist = max_range
    px, py = float(ray_origin[0]), float(ray_origin[1])
    dx, dy = float(ray_dir[0]), float(ray_dir[1])

    # Left wall (x = 0)
    if dx < -1e-6:
        t = -px / dx
        y_hit = py + t * dy
        if 0.0 <= y_hit <= height and 0.0 <= t < min_dist:
            min_dist = t

    # Right wall (x = width)
    if dx > 1e-6:
        t = (width - px) / dx
        y_hit = py + t * dy
        if 0.0 <= y_hit <= height and 0.0 <= t < min_dist:
            min_dist = t

    # Bottom wall (y = 0)
    if dy < -1e-6:
        t = -py / dy
        x_hit = px + t * dx
        if 0.0 <= x_hit <= width and 0.0 <= t < min_dist:
            min_dist = t

    # Top wall (y = height)
    if dy > 1e-6:
        t = (height - py) / dy
        x_hit = px + t * dx
        if 0.0 <= x_hit <= width and 0.0 <= t < min_dist:
            min_dist = t

    return min_dist


def compute_lidar_readings(
    agent_pos: np.ndarray,
    obstacles: List[Tuple[float, float, float]],
    arena_width: float,
    arena_height: float,
    num_rays: int = 8,
    max_range: float = 10.0,
) -> np.ndarray:
    """Calculate multi-directional normalized rangefinder / lidar distance readings.

    Args:
        agent_pos: (2,) array [x, y].
        obstacles: List of (x, y, radius) circular obstacles.
        arena_width: Arena width.
        arena_height: Arena height.
        num_rays: Number of lidar rays evenly distributed across 360 degrees.
        max_range: Maximum ray sensing range.

    Returns:
        np.ndarray: (num_rays,) array of distances normalized to [0.0, 1.0].
    """
    readings = np.full(num_rays, max_range, dtype=np.float32)

    for i in range(num_rays):
        angle = 2.0 * math.pi * i / num_rays
        ray_dir = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

        # Boundary intersection
        boundary_dist = ray_cast_arena_boundaries(
            ray_origin=agent_pos,
            ray_dir=ray_dir,
            width=arena_width,
            height=arena_height,
            max_range=max_range,
        )
        closest_dist = boundary_dist

        # Circular obstacle intersections
        for ox, oy, radius in obstacles:
            center = np.array([ox, oy], dtype=np.float32)
            obs_dist = ray_cast_circle(
                ray_origin=agent_pos,
                ray_dir=ray_dir,
                circle_center=center,
                circle_radius=radius,
                max_range=closest_dist,
            )
            if obs_dist < closest_dist:
                closest_dist = obs_dist

        readings[i] = closest_dist / max_range

    return readings


def generate_navigation_obstacles(
    width: float,
    height: float,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    num_obstacles: int,
    obstacle_radius: float,
    rng: np.random.Generator,
    clearance_margin: float = 2.0,
) -> List[Tuple[float, float, float]]:
    """Procedurally place non-overlapping circular obstacles avoiding start and goal clearance areas.

    Args:
        width: Arena width.
        height: Arena height.
        start_pos: (2,) start coordinate.
        goal_pos: (2,) goal coordinate.
        num_obstacles: Number of obstacles to place.
        obstacle_radius: Radius of each obstacle.
        rng: NumPy random generator.
        clearance_margin: Minimum buffer distance around start and goal centers.

    Returns:
        List[Tuple[float, float, float]]: List of (x, y, radius) tuples.
    """
    obstacles: List[Tuple[float, float, float]] = []
    min_dist_from_wall = obstacle_radius + 0.5
    attempts = 0
    max_attempts = num_obstacles * 100

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        ox = float(rng.uniform(min_dist_from_wall, width - min_dist_from_wall))
        oy = float(rng.uniform(min_dist_from_wall, height - min_dist_from_wall))
        candidate = np.array([ox, oy], dtype=np.float32)

        # Clearance from start and goal
        if float(np.linalg.norm(candidate - start_pos)) < (obstacle_radius + clearance_margin):
            continue
        if float(np.linalg.norm(candidate - goal_pos)) < (obstacle_radius + clearance_margin):
            continue

        # Clearance from existing obstacles (avoid excessive overlap)
        overlaps = False
        for ex_x, ex_y, ex_r in obstacles:
            if float(np.linalg.norm(candidate - np.array([ex_x, ex_y]))) < (
                obstacle_radius + ex_r + 0.5
            ):
                overlaps = True
                break

        if not overlaps:
            obstacles.append((ox, oy, obstacle_radius))

    return obstacles
