"""Continuous 2D Navigation Gymnasium environment with rangefinder sensors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from adaptive_rl.environments.base import AdaptiveRLEnv
from adaptive_rl.environments.navigation.generator import (
    compute_lidar_readings,
    generate_navigation_obstacles,
)


class ContinuousNavigation2DEnv(AdaptiveRLEnv[np.ndarray, np.ndarray]):
    """Gymnasium-compliant Continuous 2D Navigation environment with multi-directional LiDAR.

    An agent navigates a 2D rectangular arena to reach a goal while avoiding circular obstacles
    and arena perimeter walls. Equipped with rangefinder (LiDAR) sensors.

    Action Space:
        Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32):
        Continuous velocity control vector [vx, vy].

    Observation Space:
        Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32):
        - [0:2]: Agent position normalized to [0.0, 1.0]: [x / arena_width, y / arena_height]
        - [2:4]: Goal position normalized to [0.0, 1.0]: [gx / arena_width, gy / arena_height]
        - [4:6]: Relative goal vector normalized to [-1.0, 1.0]: [(gx - x) / arena_width, (gy - y) / arena_height]
        - [6:14]: 8 LiDAR rangefinder ray readings normalized to [0.0, 1.0]

    Rewards:
        +100.0: Goal reached (distance <= goal_radius, terminates episode)
        -100.0: Collision with obstacle or arena wall (terminates if terminate_on_collision is True)
        Step reward: step_penalty (-0.05) + progress shaping (prev_dist - curr_dist)
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        arena_width: float = 20.0,
        arena_height: float = 20.0,
        start_pos: Optional[Tuple[float, float]] = None,
        goal_pos: Optional[Tuple[float, float]] = None,
        num_obstacles: int = 5,
        obstacle_radius: float = 1.0,
        fixed_obstacles: Optional[List[Tuple[float, float, float]]] = None,
        agent_radius: float = 0.3,
        goal_radius: float = 0.8,
        max_speed: float = 1.0,
        dt: float = 0.1,
        num_lidar_rays: int = 8,
        lidar_range: float = 10.0,
        max_steps: int = 200,
        terminate_on_collision: bool = True,
        step_reward: float = -0.05,
        progress_weight: float = 1.0,
        goal_reward: float = 100.0,
        collision_reward: float = -100.0,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if arena_width <= 0.0 or arena_height <= 0.0:
            raise ValueError(f"Arena dimensions must be positive, got {arena_width}x{arena_height}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if num_lidar_rays < 4:
            raise ValueError(f"num_lidar_rays must be >= 4, got {num_lidar_rays}")

        self.arena_width = float(arena_width)
        self.arena_height = float(arena_height)
        self.default_start = (
            np.array(start_pos, dtype=np.float32)
            if start_pos is not None
            else np.array([2.0, 2.0], dtype=np.float32)
        )
        self.default_goal = (
            np.array(goal_pos, dtype=np.float32)
            if goal_pos is not None
            else np.array([self.arena_width - 2.0, self.arena_height - 2.0], dtype=np.float32)
        )
        self.num_obstacles = num_obstacles
        self.obstacle_radius = float(obstacle_radius)
        self.fixed_obstacles = fixed_obstacles
        self.agent_radius = float(agent_radius)
        self.goal_radius = float(goal_radius)
        self.max_speed = float(max_speed)
        self.dt = float(dt)
        self.num_lidar_rays = num_lidar_rays
        self.lidar_range = float(lidar_range)
        self.max_steps = max_steps
        self.terminate_on_collision = terminate_on_collision
        self.step_reward = float(step_reward)
        self.progress_weight = float(progress_weight)
        self.goal_reward = float(goal_reward)
        self.collision_reward = float(collision_reward)
        self.render_mode = render_mode

        # Define Gymnasium spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        obs_dim = 2 + 2 + 2 + self.num_lidar_rays
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Dynamic state
        self._agent_pos = self.default_start.copy()
        self._goal_pos = self.default_goal.copy()
        self._obstacles: List[Tuple[float, float, float]] = []
        self._current_step = 0
        self._last_lidar: np.ndarray = np.ones(self.num_lidar_rays, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Construct normalized observation vector."""
        norm_pos = np.array(
            [
                self._agent_pos[0] / self.arena_width,
                self._agent_pos[1] / self.arena_height,
            ],
            dtype=np.float32,
        )
        norm_goal = np.array(
            [
                self._goal_pos[0] / self.arena_width,
                self._goal_pos[1] / self.arena_height,
            ],
            dtype=np.float32,
        )
        rel_goal = np.array(
            [
                (self._goal_pos[0] - self._agent_pos[0]) / self.arena_width,
                (self._goal_pos[1] - self._agent_pos[1]) / self.arena_height,
            ],
            dtype=np.float32,
        )
        self._last_lidar = compute_lidar_readings(
            agent_pos=self._agent_pos,
            obstacles=self._obstacles,
            arena_width=self.arena_width,
            arena_height=self.arena_height,
            num_rays=self.num_lidar_rays,
            max_range=self.lidar_range,
        )

        obs = np.concatenate([norm_pos, norm_goal, rel_goal, self._last_lidar])
        clipped_obs: np.ndarray = np.asarray(np.clip(obs, -1.0, 1.0), dtype=np.float32)
        return clipped_obs

    def _get_info(self) -> Dict[str, Any]:
        """Provide telemetry and diagnostic state."""
        dist = float(np.linalg.norm(self._agent_pos - self._goal_pos))
        return {
            "agent_pos": self._agent_pos.copy(),
            "goal_pos": self._goal_pos.copy(),
            "distance_to_goal": dist,
            "step": self._current_step,
            "max_steps": self.max_steps,
            "lidar": self._last_lidar.copy(),
            "obstacles": list(self._obstacles),
        }

    def _check_collision(self, pos: np.ndarray) -> bool:
        """Determine if position collides with arena boundaries or circular obstacles."""
        px, py = float(pos[0]), float(pos[1])

        # Wall perimeter check
        if (
            px - self.agent_radius <= 0.0
            or px + self.agent_radius >= self.arena_width
            or py - self.agent_radius <= 0.0
            or py + self.agent_radius >= self.arena_height
        ):
            return True

        # Obstacle check
        for ox, oy, r in self._obstacles:
            dist_sq = (px - ox) ** 2 + (py - oy) ** 2
            min_dist = self.agent_radius + r
            if dist_sq <= min_dist * min_dist:
                return True

        return False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the navigation environment state with deterministic seeding."""
        super().reset(seed=seed, options=options)
        self._current_step = 0
        self._agent_pos = self.default_start.copy()
        self._goal_pos = self.default_goal.copy()

        if self.fixed_obstacles is not None:
            self._obstacles = list(self.fixed_obstacles)
        else:
            self._obstacles = generate_navigation_obstacles(
                width=self.arena_width,
                height=self.arena_height,
                start_pos=self._agent_pos,
                goal_pos=self._goal_pos,
                num_obstacles=self.num_obstacles,
                obstacle_radius=self.obstacle_radius,
                rng=self.np_random,
            )

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            print(self.render())

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one continuous simulation step."""
        self._current_step += 1

        # Clip action to valid bounds and ensure float32 array
        action_arr = np.asarray(action, dtype=np.float32).flatten()
        if action_arr.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got shape {action_arr.shape}")
        clipped_action = np.clip(action_arr, -1.0, 1.0)

        prev_pos = self._agent_pos.copy()
        prev_dist = float(np.linalg.norm(prev_pos - self._goal_pos))

        # Kinematic translation
        delta = clipped_action * (self.max_speed * self.dt)
        candidate_pos = prev_pos + delta

        # Collision detection
        is_collision = self._check_collision(candidate_pos)

        # Goal detection
        curr_dist = float(np.linalg.norm(candidate_pos - self._goal_pos))
        is_goal = curr_dist <= self.goal_radius

        terminated = False
        truncated = False
        info = self._get_info()
        info["action_taken"] = clipped_action

        if is_collision:
            reward = self.collision_reward
            info["collision"] = True
            info["success"] = False
            if self.terminate_on_collision:
                terminated = True
            # Keep position at boundary/impact point
            self._agent_pos = np.clip(
                candidate_pos,
                [self.agent_radius, self.agent_radius],
                [self.arena_width - self.agent_radius, self.arena_height - self.agent_radius],
            ).astype(np.float32)
        elif is_goal:
            self._agent_pos = candidate_pos.astype(np.float32)
            reward = self.goal_reward
            terminated = True
            info["collision"] = False
            info["success"] = True
        else:
            self._agent_pos = candidate_pos.astype(np.float32)
            progress = prev_dist - curr_dist
            reward = self.step_reward + self.progress_weight * progress
            info["collision"] = False
            info["success"] = False

        if self._current_step >= self.max_steps and not terminated:
            truncated = True

        info["agent_pos"] = self._agent_pos.copy()
        info["distance_to_goal"] = float(np.linalg.norm(self._agent_pos - self._goal_pos))
        info["step"] = self._current_step

        obs = self._get_obs()

        if self.render_mode == "human":
            print(self.render())

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render a 2D ASCII visualization of the arena."""
        grid_w = 40
        grid_h = 20

        grid = [[" " for _ in range(grid_w)] for _ in range(grid_h)]

        # Render obstacles
        for ox, oy, r in self._obstacles:
            gx = int(ox / self.arena_width * (grid_w - 1))
            gy = int((self.arena_height - oy) / self.arena_height * (grid_h - 1))
            gr = max(1, int(r / self.arena_width * grid_w))
            for dy in range(-gr, gr + 1):
                for dx in range(-gr, gr + 1):
                    if dx * dx + dy * dy <= gr * gr:
                        px = gx + dx
                        py = gy + dy
                        if 0 <= px < grid_w and 0 <= py < grid_h:
                            grid[py][px] = "#"

        # Render start
        sx = int(self.default_start[0] / self.arena_width * (grid_w - 1))
        sy = int((self.arena_height - self.default_start[1]) / self.arena_height * (grid_h - 1))
        if 0 <= sx < grid_w and 0 <= sy < grid_h:
            grid[sy][sx] = "S"

        # Render goal
        gx = int(self._goal_pos[0] / self.arena_width * (grid_w - 1))
        gy = int((self.arena_height - self._goal_pos[1]) / self.arena_height * (grid_h - 1))
        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            grid[gy][gx] = "G"

        # Render agent
        ax = int(self._agent_pos[0] / self.arena_width * (grid_w - 1))
        ay = int((self.arena_height - self._agent_pos[1]) / self.arena_height * (grid_h - 1))
        if 0 <= ax < grid_w and 0 <= ay < grid_h:
            dist = float(np.linalg.norm(self._agent_pos - self._goal_pos))
            grid[ay][ax] = "@" if dist <= self.goal_radius else "A"

        # Build framed text
        border_top = "+" + "-" * grid_w + "+"
        rows = [border_top]
        for row in grid:
            rows.append("|" + "".join(row) + "|")
        rows.append(border_top)

        dist = float(np.linalg.norm(self._agent_pos - self._goal_pos))
        status_line = (
            f"Step: {self._current_step}/{self.max_steps} | "
            f"Agent: ({self._agent_pos[0]:.1f}, {self._agent_pos[1]:.1f}) | "
            f"Dist to Goal: {dist:.2f}"
        )
        rows.append(status_line)

        output = "\n".join(rows)
        if self.render_mode == "human":
            print(output)
            return None
        return output
