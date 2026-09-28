"""Training orchestrator, callbacks, and checkpointing for AdaptiveRL."""

from adaptive_rl.training.callbacks import (
    BaseCallback,
    CheckpointCallback,
    MetricLoggerCallback,
    SB3CallbackAdapter,
)
from adaptive_rl.training.checkpointing import CheckpointManager
from adaptive_rl.training.trainer import (
    PPOTrainer,
    TrainingResult,
    get_trainer,
)

BaseTrainer = PPOTrainer

__all__ = [
    "BaseCallback",
    "BaseTrainer",
    "CheckpointCallback",
    "CheckpointManager",
    "MetricLoggerCallback",
    "PPOTrainer",
    "SB3CallbackAdapter",
    "TrainingResult",
    "get_trainer",
]
