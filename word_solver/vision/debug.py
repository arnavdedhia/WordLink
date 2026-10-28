"""Debug output: every intermediate image plus an annotated screenshot.

Files written to ``<debug_dir>/<run_name>/``::

    00_original.png         the screenshot as loaded
    01_board_crop.png       the detected board rectangle
    02_annotated.png        screenshot with grid, letter regions, and predictions
    tiles/rRcC_tile.png     full cell crops (16)
    letters/rRcC_glyph.png  normalized letter images fed to the recognizer (16)
    letters/rRcC_crop.png   raw letter-region crops before thresholding (16)
    recognition.txt         recognized board and per-cell confidence
    recognition.json        the same, machine readable
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from word_solver.vision.board_reader import BoardReading
from word_solver.vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from word_solver.vision.image_processor import save_image

GREEN = (60, 200, 60)
RED = (40, 40, 230)
YELLOW = (0, 200, 255)
BLUE = (255, 140, 0)


def annotate(reading: BoardReading) -> np.ndarray:
    """Draw the board, cell grid, letter regions, and predicted letters on a copy of the screenshot."""
    out = reading.image.copy()
    board = reading.region.rect
    cell_h = board.height / reading.rows
    thick = max(1, round(cell_h / 50))
    cv2.rectangle(out, (board.left, board.top), (board.right - 1, board.bottom - 1), BLUE, thick + 1)

    def text(label: str, org: tuple[int, int], size: float, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, label, org, font, size, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(out, label, org, font, size, color, thick, cv2.LINE_AA)

    for cell in reading.cells:
        rect = cell.image.cell.rect
        lr = cell.image.cell.letter_rect
        rec = cell.recognition
        color = RED if rec.flagged else GREEN
        cv2.rectangle(out, (rect.left, rect.top), (rect.right - 1, rect.bottom - 1), color, thick)
        cv2.rectangle(out, (lr.left, lr.top), (lr.right - 1, lr.bottom - 1), YELLOW, 1)
        text(rec.letter, (rect.right - round(cell_h * 0.27), rect.top + round(cell_h * 0.27)), cell_h / 110, color)
        text(f"{rec.confidence:.2f}", (rect.left + 3, rect.bottom - 4), cell_h / 230, color)

    status = "OK" if reading.is_reliable else f"{len(reading.flagged)} FLAGGED"
    text(f"Board ({reading.region.method}) {status}", (board.left, max(20, board.top - 10)), cell_h / 170, BLUE)
    return out


def format_reading(reading: BoardReading) -> str:
    """Text summary: recognized board, then one line per cell."""
    lines = ["Recognized board:"]
    for row in reading.board:
        lines.append("  " + " ".join(row))
    lines.append("")
    lines.append("Cell     Letter  Confidence  Runner-up  Margin  Template   Status")
    for cell in reading.cells:
        rec = cell.recognition
        status = f"FLAGGED: {rec.reason}" if rec.flagged else "ok"
        lines.append(
            f"[{cell.row}, {cell.col}]  {rec.letter:^6}  {rec.confidence:>10.3f}  "
            f"{(rec.runner_up or '-'):^9}  {rec.margin:>6.3f}  {(rec.source or '-'):<9}  {status}"
        )
    return "\n".join(lines)


def save_debug(reading: BoardReading, config: VisionConfig = DEFAULT_VISION_CONFIG, run_name: str | None = None) -> Path:
    """Write all debug artifacts for ``reading``. Returns the run folder."""
    run_dir = Path(config.debug_dir) / (run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    save_image(run_dir / "00_original.png", reading.image)
    save_image(run_dir / "01_board_crop.png", reading.region.rect.crop(reading.image))
    save_image(run_dir / "02_annotated.png", annotate(reading))
    for cell in reading.cells:
        tag = f"r{cell.row}c{cell.col}"
        save_image(run_dir / "tiles" / f"{tag}_tile.png", cell.image.tile)
        save_image(run_dir / "letters" / f"{tag}_crop.png", cell.image.letter_crop)
        save_image(run_dir / "letters" / f"{tag}_glyph.png", cell.image.glyph)

    (run_dir / "recognition.txt").write_text(format_reading(reading) + "\n", encoding="utf-8")
    payload = {
        "board": reading.board,
        "board_rect": vars(reading.region.rect),
        "board_method": reading.region.method,
        "cells": [
            {
                "row": c.row,
                "col": c.col,
                "letter": c.recognition.letter,
                "confidence": round(c.recognition.confidence, 4),
                "runner_up": c.recognition.runner_up,
                "margin": round(c.recognition.margin, 4),
                "template_source": c.recognition.source,
                "flagged": c.recognition.flagged,
                "reason": c.recognition.reason,
            }
            for c in reading.cells
        ],
    }
    (run_dir / "recognition.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_dir
