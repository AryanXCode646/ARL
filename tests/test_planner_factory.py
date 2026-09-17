"""Tests for the authoritative planner factory and failure semantics."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.planners.factory import make_planner


def test_make_planner_astar() -> None:
    """Verify make_planner successfully instantiates AStar with valid parameters."""
    planner = make_planner("astar", heuristic="euclidean", allow_diagonal=True, seed=42)
    assert planner is not None


def test_make_planner_rrt_star() -> None:
    """Verify make_planner successfully instantiates RRTStar with valid parameters and aliases."""
    planner1 = make_planner("rrt_star", step_size=0.5, search_radius=1.0)
    assert planner1 is not None

    planner2 = make_planner("rrt*", step_size=0.2, max_iterations=1000)
    assert planner2 is not None


def test_make_planner_rejects_plain_rrt() -> None:
    """Verify make_planner rejects 'rrt' with informative error suggesting 'rrt_star'."""
    with pytest.raises(ValueError, match="Planner name 'rrt' is not supported.*Did you mean 'rrt_star'"):
        make_planner("rrt")


def test_make_planner_rejects_unknown() -> None:
    """Verify make_planner rejects unknown planner identifiers."""
    with pytest.raises(ValueError, match="Unknown planner 'dijkstra'"):
        make_planner("dijkstra")


def test_make_planner_validates_parameters() -> None:
    """Verify extra or invalid parameters are rejected via schema validation."""
    with pytest.raises(ValidationError):
        make_planner("astar", unsupported_field="invalid")


def test_base_algorithm_contract() -> None:
    """Verify BaseAlgorithm ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseAlgorithm()  # type: ignore[abstract]
