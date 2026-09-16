"""Model management and artifact metadata tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Metadata describing a saved model artifact."""

    model_name: str = Field(..., description="Name of the model")
    algorithm: str = Field(..., description="Algorithm used to train the model")
    environment: str = Field(..., description="Environment on which the model was trained")
    total_timesteps: int = Field(..., description="Total training steps")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    artifact_path: str = Field(..., description="Filesystem path to model weights")


class ModelManager:
    """Manages model artifact persistence, discovery, and metadata."""

    def __init__(self, models_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def resolve_model_path(self, model_name: str) -> Optional[Path]:
        """Check if a model exists and return its path."""
        candidate = self.models_dir / model_name
        if candidate.exists():
            return candidate
        return None
