"""Decide when the on-screen board has finished changing.

Only the board rectangle is compared, so the timer, score, and other UI
chrome do not trigger a re-solve. A new board is accepted only after several
nearly identical frames, so falling-letter animations are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from word_solver.capture.config import DEFAULT_CAPTURE_CONFIG, CaptureConfig
from word_solver.vision.board_detector import configured_rect
from word_solver.vision.config import DEFAULT_VISION_CONFIG, VisionConfig


class BoardPhase(str, Enum):
    UNCHANGED = "unchanged"
    SETTLING = "settling"
    STABLE = "stable"  # a new board that has settled and should be solved


def crop_board(image: np.ndarray, vision: VisionConfig = DEFAULT_VISION_CONFIG) -> np.ndarray:
    """Grayscale crop of the configured board rectangle."""
    import cv2

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = configured_rect(image.shape, vision)
    return gray[rect.top : rect.bottom, rect.left : rect.right]


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Difference in 0-1. Images of different sizes are resized to match ``a``."""
    import cv2

    if a.size == 0 or b.size == 0:
        return 1.0
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) / 255.0


@dataclass
class DetectorState:
    phase: BoardPhase
    diff: float
    stable_count: int


class ChangeDetector:
    def __init__(
        self,
        capture_config: CaptureConfig = DEFAULT_CAPTURE_CONFIG,
        vision_config: VisionConfig = DEFAULT_VISION_CONFIG,
    ) -> None:
        self.capture_config = capture_config
        self.vision_config = vision_config
        self._prev: np.ndarray | None = None
        self._pending_change = True  # first frames after connect must settle, then solve
        self._stable = 0

    def reset(self) -> None:
        self._prev = None
        self._pending_change = True
        self._stable = 0

    def observe(self, image: np.ndarray) -> DetectorState:
        crop = crop_board(image, self.vision_config)
        if self._prev is None:
            self._prev = crop
            self._stable = 0
            return DetectorState(BoardPhase.SETTLING, 1.0, 0)

        diff = mean_abs_diff(crop, self._prev)
        self._prev = crop
        cfg = self.capture_config

        if diff >= cfg.change_threshold:
            self._pending_change = True
            self._stable = 0
            return DetectorState(BoardPhase.SETTLING, diff, 0)

        if self._pending_change:
            if diff <= cfg.settle_threshold:
                self._stable += 1
                if self._stable >= cfg.settle_frames:
                    self._pending_change = False
                    self._stable = 0
                    return DetectorState(BoardPhase.STABLE, diff, cfg.settle_frames)
            else:
                self._stable = 0
            return DetectorState(BoardPhase.SETTLING, diff, self._stable)

        return DetectorState(BoardPhase.UNCHANGED, diff, 0)
