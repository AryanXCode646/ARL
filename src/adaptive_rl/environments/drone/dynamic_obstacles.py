"""Dynamic moving 3D spherical obstacles with boundary collision physics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from adaptive_rl.environments.drone.obstacles import ObstacleSphere3D


@dataclass
class DynamicObstacleSphere3D:
    """A moving 3D spherical obstacle with constant velocity and boundary reflection."""

    center: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    radius: float  # radius in meters
    bounds: Tuple[float, float, float]  # [X_max, Y_max, Z_max]

    def to_static_sphere(self) -> ObstacleSphere3D:
        """Convert current position to static ObstacleSphere3D representation for ray-casting."""
        return ObstacleSphere3D(center=self.center.copy(), radius=self.radius)

    def collides_with(self, point: np.ndarray, collision_radius: float) -> bool:
        """Check collision against a point with given radius."""
        dist = float(np.linalg.norm(point - self.center))
        return dist <= (self.radius + collision_radius)

    def distance_to(self, point: np.ndarray) -> float:
        """Compute surface distance from a point to obstacle surface."""
        dist_to_center = float(np.linalg.norm(point - self.center))
        return dist_to_center - self.radius

    def step(self, dt: float) -> np.ndarray:
        """Advance obstacle position forward by time dt and reflect off perimeter walls.

        Returns:
            np.ndarray: Updated center position.
        """
        self.center += self.velocity * dt

        # Boundary bouncing logic
        for axis in range(3):
            min_bound = self.radius
            max_bound = self.bounds[axis] - self.radius

            if self.center[axis] <= min_bound:
                self.center[axis] = min_bound
                self.velocity[axis] = abs(self.velocity[axis])
            elif self.center[axis] >= max_bound:
                self.center[axis] = max_bound
                self.velocity[axis] = -abs(self.velocity[axis])

        return self.center.copy()


def generate_dynamic_drone_obstacles(
    bounds: Tuple[float, float, float],
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    num_obstacles: int = 4,
    obstacle_radius: float = 1.8,
    speed: float = 2.0,
    clearance_radius: float = 4.0,
    rng: np.random.Generator | None = None,
) -> List[DynamicObstacleSphere3D]:
    """Procedurally place dynamic moving obstacles with random 3D velocities.

    Args:
        bounds: (X_max, Y_max, Z_max) arena extents.
        start_pos: Initial drone takeoff position [x, y, z].
        goal_pos: Target waypoint position [x, y, z].
        num_obstacles: Number of moving obstacles to spawn.
        obstacle_radius: Radius of each moving obstacle.
        speed: Nominal scalar speed of moving obstacles in m/s.
        clearance_radius: Minimum clearance from start and goal waypoints.
        rng: NumPy random generator instance.

    Returns:
        List[DynamicObstacleSphere3D]: Generated dynamic obstacles.
    """
    if rng is None:
        rng = np.random.default_rng()

    obstacles: List[DynamicObstacleSphere3D] = []
    start = np.asarray(start_pos, dtype=np.float64)
    goal = np.asarray(goal_pos, dtype=np.float64)

    min_x, max_x = obstacle_radius + 2.0, bounds[0] - obstacle_radius - 2.0
    min_y, max_y = obstacle_radius + 2.0, bounds[1] - obstacle_radius - 2.0
    min_z, max_z = obstacle_radius + 2.0, bounds[2] - obstacle_radius - 2.0

    attempts = 0
    max_attempts = num_obstacles * 100

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        cx = float(rng.uniform(min_x, max_x))
        cy = float(rng.uniform(min_y, max_y))
        cz = float(rng.uniform(min_z, max_z))
        pos = np.array([cx, cy, cz], dtype=np.float64)

        # Clearance from start and goal
        if float(np.linalg.norm(pos - start)) < (obstacle_radius + clearance_radius):
            continue
        if float(np.linalg.norm(pos - goal)) < (obstacle_radius + clearance_radius):
            continue

        # Random 3D velocity unit vector
        raw_v = rng.normal(size=3)
        v_norm = float(np.linalg.norm(raw_v))
        if v_norm < 1e-6:
            continue
        velocity = (raw_v / v_norm) * speed

        obstacles.append(
            DynamicObstacleSphere3D(
                center=pos,
                velocity=velocity,
                radius=obstacle_radius,
                bounds=bounds,
            )
        )

    return obstacles
