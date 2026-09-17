from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = ["launch_studio"]


def launch_studio(output_dir: Optional[Path] = None) -> int:
    """Launch Studio lazily so importing :mod:`adaptive_rl` stays lightweight."""
    from adaptive_rl.studio.app import launch_studio as _launch_studio

    return _launch_studio(output_dir=output_dir)
