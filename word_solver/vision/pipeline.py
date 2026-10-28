"""One call from screenshot file to recognized board, with optional debug output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from word_solver.vision.board_reader import BoardReading, read_board
from word_solver.vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from word_solver.vision.debug import save_debug
from word_solver.vision.image_processor import load_image


@dataclass
class ScreenshotResult:
    reading: BoardReading
    debug_dir: Path | None


def recognize_screenshot(
    image_path: str | Path,
    config: VisionConfig = DEFAULT_VISION_CONFIG,
    debug: bool = False,
) -> ScreenshotResult:
    reading = read_board(load_image(image_path), config)
    debug_dir = None
    if debug:
        debug_dir = save_debug(reading, config, run_name=f"{Path(image_path).stem}")
    return ScreenshotResult(reading, debug_dir)
