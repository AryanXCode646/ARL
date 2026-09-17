"""Gymnasium-compatible 3D autonomous drone navigation environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from adaptive_rl.environments.base import AdaptiveRLEnv
from adaptive_rl.environments.drone.kinematics import DroneKinematics3D
from adaptive_rl.environments.drone.obstacles import (
    ObstacleSphere3D,
    compute_lidar_3d_readings,
    generate_drone_obstacles,
    generate_lidar_3d_ray_directions,
)


class DroneNavigation3DEnv(AdaptiveRLEnv[np.ndarray, np.ndarray]):
    """3D Autonomous Drone Navigation Gymnasium environment.

    Simulates continuous 3D quadrotor translation physics, aerodynamic damping,
    procedural spherical obstacles, and multi-directional 3D LiDAR sensors.

    Actions:
        Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        Continuous 3D acceleration command [ax, ay, az], scaled by `max_acceleration`.

    Observations:
        Box(-1.0, 1.0, shape=(29,), dtype=np.float32)
        [0:3]   : Normalized 3D position [x/X_max, y/Y_max, z/Z_max] in [0, 1]
        [3:6]   : Normalized 3D velocity [vx/v_max, vy/v_max, vz/v_max] in [-1, 1]
        [6:9]   : Normalized target goal [gx/X_max, gy/Y_max, gz/Z_max] in [0, 1]
        [9:12]  : Relative target vector [(gx - x)/X_max, (gy - y)/Y_max, (gz - z)/Z_max] in [-1, 1]
        [12]    : Normalized distance to goal ||goal - pos|| / max_diagonal in [0, 1]
        [13:29] : 16-ray 3D LiDAR distance rangefinder readings in [0, 1]

    Rewards:
        +100.0 : Target waypoint reached within `target_radius` (terminates episode)
        -100.0 : Collision with obstacle or boundary perimeter (terminates episode)
        +progress: Distance delta toward goal (prev_dist - curr_dist) * progress_weight
        -0.05  : Per-step flight time penalty
        -0.01  : Action effort / control smoothness penalty (-action_penalty_weight * ||u||^2)
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        bounds: Tuple[float, float, float] = (50.0, 50.0, 25.0),
        start_pos: Optional[Tuple[float, float, float] | np.ndarray] = None,
        goal_pos: Optional[Tuple[float, float, float] | np.ndarray] = None,
        num_obstacles: int = 8,
        obstacle_radius: float = 2.0,
        target_radius: float = 1.5,
        collision_radius: float = 0.8,
        lidar_range: float = 20.0,
        num_lidar_rays: int = 16,
        dt: float = 0.1,
        max_velocity: float = 8.0,
        max_acceleration: float = 4.0,
        linear_damping: float = 0.05,
        max_steps: int = 300,
        step_penalty: float = -0.05,
        goal_reward: float = 100.0,
        collision_reward: float = -100.0,
        progress_weight: float = 1.0,
        action_penalty_weight: float = 0.01,
        terminate_on_collision: bool = True,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the 3D Drone Navigation environment."""
        super().__init__()
        if any(b <= 0.0 for b in bounds):
            raise ValueError(f"Bounds must be strictly positive, got {bounds}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if target_radius <= 0.0:
            raise ValueError(f"target_radius must be positive, got {target_radius}")
        if collision_radius <= 0.0:
            raise ValueError(f"collision_radius must be positive, got {collision_radius}")

        self.bounds: Tuple[float, float, float] = (
            float(bounds[0]),
            float(bounds[1]),
            float(bounds[2]),
        )
        self.default_start = (
            np.asarray(start_pos, dtype=np.float64)
            if start_pos is not None
            else np.array([5.0, 5.0, 5.0], dtype=np.float64)
        )
        self.default_goal = (
            np.asarray(goal_pos, dtype=np.float64)
            if goal_pos is not None
            else np.array(
                [self.bounds[0] - 5.0, self.bounds[1] - 5.0, self.bounds[2] - 5.0], dtype=np.float64
            )
        )
        self.num_obstacles = num_obstacles
        self.obstacle_radius = float(obstacle_radius)
        self.target_radius = float(target_radius)
        self.collision_radius = float(collision_radius)
        self.lidar_range = float(lidar_range)
        self.num_lidar_rays = num_lidar_rays
        self.max_steps = max_steps
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.collision_reward = float(collision_reward)
        self.progress_weight = float(progress_weight)
        self.action_penalty_weight = float(action_penalty_weight)
        self.terminate_on_collision = terminate_on_collision
        self.render_mode = render_mode

        self.max_diagonal = float(np.linalg.norm(self.bounds))

        # Define Farama Gymnasium Action Space: 3D acceleration [ax, ay, az] in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # Observation Space: 29 continuous values in [-1.0, 1.0]
        # (3 pos + 3 vel + 3 goal + 3 rel_goal + 1 dist + 16 lidar = 29)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(13 + self.num_lidar_rays,),
            dtype=np.float32,
        )

        # Precompute 3D LiDAR ray directions
        self.lidar_rays = generate_lidar_3d_ray_directions(num_rays=self.num_lidar_rays)

        # Kinematics engine
        self.kinematics = DroneKinematics3D(
            dt=dt,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            linear_damping=linear_damping,
        )

        # Dynamic state
        self._position: np.ndarray = self.default_start.copy()
        self._velocity: np.ndarray = np.zeros(3, dtype=np.float64)
        self._goal: np.ndarray = self.default_goal.copy()
        self._obstacles: List[ObstacleSphere3D] = []
        self._current_step = 0
        self._prev_distance_to_goal: float = float(np.linalg.norm(self._default_distance()))

    def _default_distance(self) -> float:
        return float(np.linalg.norm(self.default_goal - self.default_start))

    def _get_obs(self) -> np.ndarray:
        """Construct normalized 29-dimensional observation vector."""
        bx, by, bz = self.bounds
        v_max = self.kinematics.max_velocity

        norm_pos = [
            self._position[0] / bx,
            self._position[1] / by,
            self._position[2] / bz,
        ]
        norm_vel = [
            self._velocity[0] / v_max,
            self._velocity[1] / v_max,
            self._velocity[2] / v_max,
        ]
        norm_goal = [
            self._goal[0] / bx,
            self._goal[1] / by,
            self._goal[2] / bz,
        ]
        rel_goal = [
            (self._goal[0] - self._position[0]) / bx,
            (self._goal[1] - self._position[1]) / by,
            (self._goal[2] - self._position[2]) / bz,
        ]
        curr_dist = float(np.linalg.norm(self._goal - self._position))
        norm_dist = [min(1.0, curr_dist / self.max_diagonal)]

        lidar_readings = compute_lidar_3d_readings(
            origin=self._position,
            ray_directions=self.lidar_rays,
            obstacles=self._obstacles,
            bounds=self.bounds,
            max_range=self.lidar_range,
        )

        raw = np.concatenate([norm_pos, norm_vel, norm_goal, rel_goal, norm_dist, lidar_readings])
        return np.asarray(np.clip(raw, -1.0, 1.0), dtype=np.float32)

    def _check_collision(self, pos: np.ndarray) -> Tuple[bool, str]:
        """Check if drone collides with arena boundaries or obstacle spheres."""
        r = self.collision_radius
        bx, by, bz = self.bounds

        # Boundary collisions
        if pos[0] - r <= 0.0 or pos[0] + r >= bx:
            return True, "boundary_x"
        if pos[1] - r <= 0.0 or pos[1] + r >= by:
            return True, "boundary_y"
        if pos[2] - r <= 0.0 or pos[2] + r >= bz:
            return True, "boundary_z"

        # Obstacle collisions
        for obs in self._obstacles:
            if obs.collides_with(pos, r):
                return True, "obstacle"

        return False, "none"

    def _get_info(self) -> Dict[str, Any]:
        """Construct comprehensive metrics and state dictionary."""
        dist = float(np.linalg.norm(self._goal - self._position))
        speed = float(np.linalg.norm(self._velocity))

        min_obs_dist = float("inf")
        for obs in self._obstacles:
            d = obs.distance_to(self._position) - self.collision_radius
            if d < min_obs_dist:
                min_obs_dist = d

        return {
            "step": self._current_step,
            "max_steps": self.max_steps,
            "position": self._position.copy(),
            "velocity": self._velocity.copy(),
            "speed": speed,
            "goal": self._goal.copy(),
            "distance_to_goal": dist,
            "num_obstacles": len(self._obstacles),
            "min_obstacle_distance": min_obs_dist if self._obstacles else float("inf"),
            "altitude": float(self._position[2]),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the drone navigation environment with deterministic seeding."""
        super().reset(seed=seed, options=options)
        self._current_step = 0

        # Optional parameter overrides via reset options
        num_obs = self.num_obstacles
        if options and "num_obstacles" in options:
            num_obs = options["num_obstacles"]

        self._position = self.default_start.copy()
        self._velocity = np.zeros(3, dtype=np.float64)
        self._goal = self.default_goal.copy()

        self.kinematics.reset(self._position, self._velocity)

        # Procedurally generate 3D obstacle field
        self._obstacles = generate_drone_obstacles(
            bounds=self.bounds,
            start_pos=self._position,
            goal_pos=self._goal,
            num_obstacles=num_obs,
            obstacle_radius=self.obstacle_radius,
            clearance_radius=self.collision_radius + 2.0,
            rng=self.np_random,
        )

        self._prev_distance_to_goal = float(np.linalg.norm(self._goal - self._position))

        if self.render_mode == "human":
            print(self.render())

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one continuous 3D acceleration action step."""
        act_arr = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(act_arr):
            raise ValueError(f"Action {act_arr} outside action space bounds {self.action_space}")

        self._current_step += 1

        # Scale action from [-1, 1] to [-max_acc, max_acc]
        acc_command = act_arr * self.kinematics.max_acceleration

        # Step 3D translational kinematics
        new_pos, new_vel = self.kinematics.step(acc_command)
        self._position = new_pos
        self._velocity = new_vel

        # Evaluate distance and progress
        curr_distance = float(np.linalg.norm(self._goal - self._position))
        dist_delta = self._prev_distance_to_goal - curr_distance
        self._prev_distance_to_goal = curr_distance

        # Collision detection
        is_collision, collision_type = self._check_collision(self._position)

        # Goal arrival check
        is_success = curr_distance <= self.target_radius

        terminated = False
        truncated = False
        info = self._get_info()
        info["collision"] = is_collision
        info["collision_type"] = collision_type
        info["success"] = is_success

        if is_collision:
            reward = self.collision_reward
            info["success"] = False
            if self.terminate_on_collision:
                terminated = True
        elif is_success:
            reward = self.goal_reward
            info["success"] = True
            terminated = True
        else:
            action_effort = float(np.sum(np.square(act_arr)))
            progress_reward = self.progress_weight * dist_delta
            effort_penalty = self.action_penalty_weight * action_effort
            reward = float(progress_reward + self.step_penalty - effort_penalty)

        if self._current_step >= self.max_steps and not terminated:
            truncated = True

        if self.render_mode == "human":
            print(self.render())

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render textual 3D flight dashboard and projections."""
        dist = float(np.linalg.norm(self._goal - self._position))
        speed = float(np.linalg.norm(self._velocity))
        pos_str = f"[{self._position[0]:5.1f}, {self._position[1]:5.1f}, {self._position[2]:5.1f}]"
        vel_str = f"[{self._velocity[0]:5.1f}, {self._velocity[1]:5.1f}, {self._velocity[2]:5.1f}]"
        goal_str = f"[{self._goal[0]:5.1f}, {self._goal[1]:5.1f}, {self._goal[2]:5.1f}]"

        lines = [
            "+----------------------------------------------------------------+",
            "|               AUTONOMOUS 3D DRONE FLIGHT DECK                  |",
            "+----------------------------------------------------------------+",
            f"| Step: {self._current_step:03d}/{self.max_steps:03d} | Altitude (Z): {self._position[2]:5.1f}m | Speed: {speed:4.1f} m/s          |",
            f"| Position [X, Y, Z]: {pos_str:<28} |",
            f"| Velocity [Vx,Vy,Vz]: {vel_str:<28} |",
            f"| Waypoint [Gx,Gy,Gz]: {goal_str:<28} |",
            f"| Range to Target: {dist:5.1f}m | Obstacles in Area: {len(self._obstacles):02d}                  |",
            "+----------------------------------------------------------------+",
            "  Flight Arena Boundaries: [0..{0:.0f}, 0..{1:.0f}, 0..{2:.0f}] m".format(
                self.bounds[0], self.bounds[1], self.bounds[2]
            ),
            "+----------------------------------------------------------------+",
        ]
        dashboard = "\n".join(lines)

        if self.render_mode == "human":
            print(dashboard)
            return None
        return dashboard
