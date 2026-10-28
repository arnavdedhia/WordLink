"""Settings for live screenshot capture and board-change detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = PACKAGE_DIR / "captures"


@dataclass(frozen=True)
class CaptureConfig:
    # How often to grab a frame while watching the phone.
    poll_interval_seconds: float = 0.45
    # How often to retry after a disconnect or capture error.
    reconnect_interval_seconds: float = 2.0

    # Mean absolute difference of the board crop (0-1). Above this, the board is
    # treated as having changed (or still animating).
    change_threshold: float = 0.035
    # Consecutive frames must stay at or below this difference to count as settled.
    settle_threshold: float = 0.012
    settle_frames: int = 3
    settle_timeout_seconds: float = 5.0

    # Seconds to wait for the Trust dialog during pairing.
    pair_timeout_seconds: float = 60.0
    # Seconds a single screenshot request may take.
    capture_timeout_seconds: float = 20.0

    save_captures: bool = False
    captures_dir: Path = CAPTURES_DIR


DEFAULT_CAPTURE_CONFIG = CaptureConfig()
