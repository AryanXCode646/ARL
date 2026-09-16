"""Model and experiment checkpoint management for AdaptiveRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


class CheckpointManager:
    """Manages periodic, best-metric, and final checkpoint creation and recovery."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, Any]] = []

    def record_checkpoint(self, path: str | Path, step: int, metric_value: float) -> None:
        """Track saved checkpoint metadata in manager history."""
        self._history.append({
            "path": str(path),
            "step": step,
            "metric_value": metric_value,
        })

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Return recorded checkpoint history."""
        return list(self._history)
