"""Procedural map and obstacle generation with path validity verification."""

from __future__ import annotations

from collections import deque
from typing import Set, Tuple

import numpy as np

Coordinate = Tuple[int, int]


def is_path_available(
    width: int,
    height: int,
    start: Coordinate,
    goal: Coordinate,
    obstacles: Set[Coordinate],
) -> bool:
    """Check whether a navigable orthogonal path exists from start to goal using BFS.

    Args:
        width: Grid width (X dimension).
        height: Grid height (Y dimension).
        start: (x, y) starting coordinate.
        goal: (x, y) target coordinate.
        obstacles: Set of blocked (x, y) obstacle coordinates.

    Returns:
        bool: True if a path exists or start equals goal, False otherwise.
    """
    if start == goal:
        return True

    if start in obstacles or goal in obstacles:
        return False

    queue: deque[Coordinate] = deque([start])
    visited: Set[Coordinate] = {start}

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == goal:
            return True

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            if 0 <= nx < width and 0 <= ny < height:
                if neighbor not in obstacles and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return False


def generate_procedural_obstacles(
    width: int,
    height: int,
    start: Coordinate,
    goal: Coordinate,
    num_obstacles: int,
    rng: np.random.Generator,
    max_attempts: int = 200,
) -> Set[Coordinate]:
    """Generate a random obstacle set while guaranteeing at least one valid path from start to goal.

    Args:
        width: Grid width.
        height: Grid height.
        start: (x, y) start coordinate.
        goal: (x, y) goal coordinate.
        num_obstacles: Desired number of obstacle cells.
        rng: Initialized NumPy random number generator.
        max_attempts: Maximum generation trials before reducing obstacle count.

    Returns:
        Set[Coordinate]: Set of (x, y) obstacle coordinates.
    """
    all_cells = [
        (x, y) for x in range(width) for y in range(height) if (x, y) != start and (x, y) != goal
    ]

    if num_obstacles <= 0 or not all_cells:
        return set()

    max_possible = len(all_cells)
    target_count = min(num_obstacles, max_possible)

    for _ in range(max_attempts):
        chosen_indices = rng.choice(len(all_cells), size=target_count, replace=False)
        candidate_obstacles = {all_cells[i] for i in chosen_indices}

        if is_path_available(width, height, start, goal, candidate_obstacles):
            return candidate_obstacles

    # If dense obstacles prevented finding a path, greedily add obstacles one by one
    safe_obstacles: Set[Coordinate] = set()
    shuffled_cells = [all_cells[i] for i in rng.permutation(len(all_cells))]

    for cell in shuffled_cells:
        if len(safe_obstacles) >= target_count:
            break
        test_obstacles = safe_obstacles | {cell}
        if is_path_available(width, height, start, goal, test_obstacles):
            safe_obstacles.add(cell)

    return safe_obstacles
