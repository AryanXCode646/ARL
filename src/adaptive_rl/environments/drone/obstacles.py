"""3D Spherical obstacles, procedural field generation, and 3D LiDAR ray-casting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ObstacleSphere3D:
    """A spherical 3D obstacle defined by its center position and radius."""

    center: np.ndarray  # [x, y, z] in meters
    radius: float  # radius in meters

    def distance_to(self, point: np.ndarray) -> float:
        """Compute surface distance from a 3D point to this obstacle (negative if inside)."""
        dist_to_center = float(np.linalg.norm(point - self.center))
        return dist_to_center - self.radius

    def collides_with(self, point: np.ndarray, collision_radius: float) -> bool:
        """Check if a point with a given physical radius collides with this obstacle."""
        dist_to_center = float(np.linalg.norm(point - self.center))
        return dist_to_center <= (self.radius + collision_radius)


def ray_cast_sphere_3d(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    center: np.ndarray,
    radius: float,
    max_range: float = 20.0,
) -> float:
    """Analytically compute ray intersection distance with a 3D sphere.

    Ray equation: p(t) = origin + t * direction, t >= 0
    Sphere equation: ||p - center||^2 = radius^2

    Returns:
        float: Distance to closest intersection in [0.0, max_range].
    """
    orig = np.asarray(ray_origin, dtype=np.float64)
    direction = np.asarray(ray_direction, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)

    # Normalize direction just in case
    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-9:
        return float(max_range)
    d = direction / dir_norm

    m = orig - c
    b = float(np.dot(m, d))
    c_const = float(np.dot(m, m)) - radius * radius

    # Origin is inside sphere
    if c_const <= 0.0:
        return 0.0

    # Ray points away from sphere
    if b > 0.0 and c_const > 0.0:
        return float(max_range)

    discriminant = b * b - c_const
    if discriminant < 0.0:
        # No intersection
        return float(max_range)

    t = -b - float(np.sqrt(discriminant))
    if 0.0 <= t <= max_range:
        return float(t)
    return float(max_range)


def ray_cast_box_boundaries_3d(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    bounds: Tuple[float, float, float],
    max_range: float = 20.0,
) -> float:
    """Compute distance from inside a 3D bounding box to the nearest boundary along a ray.

    Bounding arena: [0, bounds[0]] x [0, bounds[1]] x [0, bounds[2]]

    Returns:
        float: Distance along ray to the bounding perimeter in [0.0, max_range].
    """
    orig = np.asarray(ray_origin, dtype=np.float64)
    direction = np.asarray(ray_direction, dtype=np.float64)

    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-9:
        return float(max_range)
    d = direction / dir_norm

    t_exit = float("inf")
    for axis in range(3):
        axis_dir = d[axis]
        if abs(axis_dir) > 1e-7:
            if axis_dir > 0.0:
                t = (bounds[axis] - orig[axis]) / axis_dir
            else:
                t = (0.0 - orig[axis]) / axis_dir
            if 0.0 <= t < t_exit:
                t_exit = t

    return float(min(t_exit, max_range))


def generate_lidar_3d_ray_directions(num_rays: int = 16) -> np.ndarray:
    """Generate unit vector directions for 3D rangefinder rays.

    Default 16 rays:
    - 8 equatorial horizontal rays (elevation 0 deg, azimuth in [0..315] deg)
    - 4 upper hemispheric rays (elevation +45 deg, azimuth in [0, 90, 180, 270] deg)
    - 4 lower hemispheric rays (elevation -45 deg, azimuth in [0, 90, 180, 270] deg)

    Returns:
        np.ndarray: Array of shape (num_rays, 3) containing unit direction vectors.
    """
    if num_rays == 16:
        rays: List[List[float]] = []
        # 8 horizontal rays
        for h_deg in np.linspace(0.0, 315.0, 8):
            az_h = float(np.radians(float(h_deg)))
            rays.append([float(np.cos(az_h)), float(np.sin(az_h)), 0.0])

        # 4 upper rays (pitch = +45 deg)
        elevation_up = float(np.radians(45.0))
        cos_el = float(np.cos(elevation_up))
        sin_el = float(np.sin(elevation_up))
        for u_deg in [0.0, 90.0, 180.0, 270.0]:
            az_u = float(np.radians(u_deg))
            rays.append([float(np.cos(az_u) * cos_el), float(np.sin(az_u) * cos_el), sin_el])

        # 4 lower rays (pitch = -45 deg)
        elevation_down = float(np.radians(-45.0))
        cos_el_down = float(np.cos(elevation_down))
        sin_el_down = float(np.sin(elevation_down))
        for d_deg in [0.0, 90.0, 180.0, 270.0]:
            az_d = float(np.radians(d_deg))
            rays.append(
                [float(np.cos(az_d) * cos_el_down), float(np.sin(az_d) * cos_el_down), sin_el_down]
            )

        arr = np.array(rays, dtype=np.float32)
        # Normalize to unit length
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return (arr / norms).astype(np.float32)

    # General fallback: Fibonacci sphere spiral for arbitrary num_rays
    indices = np.arange(0, num_rays, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / num_rays)
    theta = np.pi * (1.0 + 5.0**0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    rays_fib = np.stack([x, y, z], axis=1).astype(np.float32)
    return rays_fib


def compute_lidar_3d_readings(
    origin: np.ndarray,
    ray_directions: np.ndarray,
    obstacles: List[ObstacleSphere3D],
    bounds: Tuple[float, float, float],
    max_range: float = 20.0,
) -> np.ndarray:
    """Compute normalized rangefinder distance readings along 3D ray directions.

    Returns:
        np.ndarray: Array of shape (num_rays,) with normalized values in [0.0, 1.0].
    """
    orig = np.asarray(origin, dtype=np.float64)
    readings: List[float] = []

    for ray_dir in ray_directions:
        min_dist = ray_cast_box_boundaries_3d(orig, ray_dir, bounds, max_range=max_range)
        for obs in obstacles:
            dist = ray_cast_sphere_3d(orig, ray_dir, obs.center, obs.radius, max_range=max_range)
            if dist < min_dist:
                min_dist = dist
        readings.append(min_dist / max_range)

    return np.clip(readings, 0.0, 1.0).astype(np.float32)


def generate_drone_obstacles(
    bounds: Tuple[float, float, float],
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    num_obstacles: int = 8,
    obstacle_radius: float = 2.0,
    clearance_radius: float = 3.0,
    rng: np.random.Generator | None = None,
) -> List[ObstacleSphere3D]:
    """Procedurally place 3D spherical obstacles with guaranteed clearance from start and goal.

    Args:
        bounds: (X_max, Y_max, Z_max) arena extents.
        start_pos: Initial drone position [x, y, z].
        goal_pos: Target waypoint position [x, y, z].
        num_obstacles: Number of spherical obstacles to generate.
        obstacle_radius: Default radius of each obstacle sphere in meters.
        clearance_radius: Minimum clearance required around start and goal positions.
        rng: NumPy random generator instance.

    Returns:
        List[ObstacleSphere3D]: Generated non-colliding obstacles.
    """
    if rng is None:
        rng = np.random.default_rng()

    obstacles: List[ObstacleSphere3D] = []
    start = np.asarray(start_pos, dtype=np.float64)
    goal = np.asarray(goal_pos, dtype=np.float64)

    min_x, max_x = obstacle_radius + 1.0, bounds[0] - obstacle_radius - 1.0
    min_y, max_y = obstacle_radius + 1.0, bounds[1] - obstacle_radius - 1.0
    min_z, max_z = obstacle_radius + 1.0, bounds[2] - obstacle_radius - 1.0

    # Handle small bounds gracefully
    min_x, max_x = min(min_x, max_x), max(min_x, max_x)
    min_y, max_y = min(min_y, max_y), max(min_y, max_y)
    min_z, max_z = min(min_z, max_z), max(min_z, max_z)

    attempts = 0
    max_attempts = num_obstacles * 100

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        cx = float(rng.uniform(min_x, max_x))
        cy = float(rng.uniform(min_y, max_y))
        cz = float(rng.uniform(min_z, max_z))
        candidate_center = np.array([cx, cy, cz], dtype=np.float64)

        # Clearance from start and goal
        if float(np.linalg.norm(candidate_center - start)) < (obstacle_radius + clearance_radius):
            continue
        if float(np.linalg.norm(candidate_center - goal)) < (obstacle_radius + clearance_radius):
            continue

        # Clearance from already placed obstacles to avoid excessive clustering
        overlap = False
        for existing in obstacles:
            if float(np.linalg.norm(candidate_center - existing.center)) < (
                obstacle_radius + existing.radius + 1.0
            ):
                overlap = True
                break

        if not overlap:
            obstacles.append(ObstacleSphere3D(center=candidate_center, radius=obstacle_radius))

    return obstacles
