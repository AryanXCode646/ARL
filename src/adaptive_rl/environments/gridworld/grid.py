"""Procedurally generated 2D GridWorld Gymnasium environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from gymnasium import spaces

from adaptive_rl.environments.base import AdaptiveRLEnv
from adaptive_rl.environments.gridworld.generator import Coordinate, generate_procedural_obstacles


class GridWorldEnv(AdaptiveRLEnv[np.ndarray, int]):
    """Gymnasium-compatible 2D GridWorld environment with obstacle navigation.

    Actions:
        0: UP    (decrements Y)
        1: DOWN  (increments Y)
        2: LEFT  (decrements X)
        3: RIGHT (increments X)

    Observations:
        Box(4,) with values in [0.0, 1.0]:
        [agent_x / (width - 1), agent_y / (height - 1), goal_x / (width - 1), goal_y / (height - 1)]

    Rewards:
        +100.0 : Goal reached (terminates episode)
        -100.0 : Obstacle collision (terminates episode if terminate_on_collision is True)
          -1.0 : Each step transition penalty
    """

    metadata = {"render_modes": ["ansi", "human"]}

    # Action mappings: (dx, dy)
    ACTION_TO_DELTA: Dict[int, Tuple[int, int]] = {
        0: (0, -1),  # UP
        1: (0, 1),  # DOWN
        2: (-1, 0),  # LEFT
        3: (1, 0),  # RIGHT
    }

    ACTION_NAMES: Dict[int, str] = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
    }

    def __init__(
        self,
        width: int = 6,
        height: int = 5,
        start_pos: Optional[Coordinate] = None,
        goal_pos: Optional[Coordinate] = None,
        num_obstacles: int = 4,
        fixed_obstacles: Optional[Set[Coordinate] | List[Coordinate]] = None,
        max_steps: int = 100,
        terminate_on_collision: bool = True,
        step_reward: float = -1.0,
        goal_reward: float = 100.0,
        collision_reward: float = -100.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if width < 2 or height < 2:
            raise ValueError(f"Grid dimensions must be at least 2x2, got {width}x{height}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")

        self.width = width
        self.height = height
        self.default_start = start_pos or (0, 0)
        self.default_goal = goal_pos or (width - 1, height - 1)
        self.num_obstacles = num_obstacles
        self.fixed_obstacles: Optional[Set[Coordinate]] = (
            set(fixed_obstacles) if fixed_obstacles is not None else None
        )
        self.max_steps = max_steps
        self.terminate_on_collision = terminate_on_collision
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.collision_reward = collision_reward
        self.render_mode = render_mode

        # Verify default positions within bounds
        self._validate_position(self.default_start, "start_pos")
        self._validate_position(self.default_goal, "goal_pos")

        # Define Farama Gymnasium spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Dynamic state
        self._agent_pos: Coordinate = self.default_start
        self._goal_pos: Coordinate = self.default_goal
        self._obstacles: Set[Coordinate] = set()
        self._current_step = 0

    def _validate_position(self, pos: Coordinate, name: str) -> None:
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"{name} {pos} is outside grid bounds {self.width}x{self.height}")

    def _get_obs(self) -> np.ndarray:
        """Return normalized [agent_x, agent_y, goal_x, goal_y] observation vector."""
        max_x = max(1, self.width - 1)
        max_y = max(1, self.height - 1)
        return np.array(
            [
                self._agent_pos[0] / max_x,
                self._agent_pos[1] / max_y,
                self._goal_pos[0] / max_x,
                self._goal_pos[1] / max_y,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> Dict[str, Any]:
        """Return debugging and telemetry information."""
        return {
            "agent_pos": self._agent_pos,
            "goal_pos": self._goal_pos,
            "step": self._current_step,
            "max_steps": self.max_steps,
            "obstacles": sorted(list(self._obstacles)),
            "distance_to_goal": abs(self._agent_pos[0] - self._goal_pos[0])
            + abs(self._agent_pos[1] - self._goal_pos[1]),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the GridWorld environment to start state with deterministic seeding."""
        super().reset(seed=seed, options=options)
        self._current_step = 0
        self._agent_pos = self.default_start
        self._goal_pos = self.default_goal

        # Generate or assign obstacles
        if self.fixed_obstacles is not None:
            self._obstacles = set(self.fixed_obstacles)
        else:
            self._obstacles = generate_procedural_obstacles(
                width=self.width,
                height=self.height,
                start=self._agent_pos,
                goal=self._goal_pos,
                num_obstacles=self.num_obstacles,
                rng=self.np_random,
            )

        if self.render_mode == "human":
            print(self.render())

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the grid."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Valid actions are 0, 1, 2, 3.")

        self._current_step += 1
        dx, dy = self.ACTION_TO_DELTA[action]
        target_x = self._agent_pos[0] + dx
        target_y = self._agent_pos[1] + dy

        # Boundary collision: clamp to grid borders
        target_x = max(0, min(self.width - 1, target_x))
        target_y = max(0, min(self.height - 1, target_y))
        new_pos = (target_x, target_y)

        terminated = False
        truncated = False
        info = self._get_info()
        info["action_name"] = self.ACTION_NAMES[action]

        # Case 1: Obstacle collision
        if new_pos in self._obstacles:
            reward = self.collision_reward
            info["collision"] = True
            info["success"] = False
            if self.terminate_on_collision:
                terminated = True
            # Keep agent at previous position on obstacle hit
            self._agent_pos = self._agent_pos

        # Case 2: Goal reached
        elif new_pos == self._goal_pos:
            self._agent_pos = new_pos
            reward = self.goal_reward
            terminated = True
            info["collision"] = False
            info["success"] = True

        # Case 3: Normal step
        else:
            self._agent_pos = new_pos
            reward = self.step_reward
            info["collision"] = False
            info["success"] = False

        # Truncation check
        if self._current_step >= self.max_steps and not terminated:
            truncated = True

        # Refresh state in info
        info["agent_pos"] = self._agent_pos
        info["step"] = self._current_step
        info["distance_to_goal"] = abs(self._agent_pos[0] - self._goal_pos[0]) + abs(
            self._agent_pos[1] - self._goal_pos[1]
        )

        if self.render_mode == "human":
            print(self.render())

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render textual representation of the current grid."""
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos == self._agent_pos and pos == self._goal_pos:
                    row.append("@")  # Agent reached goal
                elif pos == self._agent_pos:
                    row.append("A")
                elif pos == self._goal_pos:
                    row.append("G")
                elif pos == self.default_start:
                    row.append("S")
                elif pos in self._obstacles:
                    row.append("X")
                else:
                    row.append(".")
            lines.append(" ".join(row))

        grid_str = "\n".join(lines)
        if self.render_mode == "human":
            print(grid_str)
            return None
        return grid_str
