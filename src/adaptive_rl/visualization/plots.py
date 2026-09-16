"""Plotting interfaces for metric trajectories and reward curves."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union


class PlotManager:
    """Manages generation and export of training curves and benchmark graphs."""

    @staticmethod
    def format_reward_summary(episode_rewards: List[float]) -> str:
        """Return a simple textual summary of reward progression."""
        if not episode_rewards:
            return "No reward data recorded."
        min_r = min(episode_rewards)
        max_r = max(episode_rewards)
        avg_r = sum(episode_rewards) / len(episode_rewards)
        return f"Episodes: {len(episode_rewards)} | Min: {min_r:.2f} | Avg: {avg_r:.2f} | Max: {max_r:.2f}"
