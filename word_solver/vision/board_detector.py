"""Locating the board in a screenshot and splitting it into cells.

Detection has two steps:

1. Convert the normalized ``board_left/top/right/bottom`` fractions from
   ``VisionConfig`` into pixels for this screenshot's size.
2. Optionally refine: search a slightly larger window for the light, square
   tiles and, if exactly ``rows * cols`` of them are found, snap the board
   to their combined bounding box. This absorbs small layout differences
   between phone models. If refinement fails, the configured rectangle is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from word_solver.vision.config import DEFAULT_VISION_CONFIG, VisionConfig


@dataclass(frozen=True)
class Rect:
    """Axis-aligned pixel rectangle; ``right`` and ``bottom`` are exclusive."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.top : self.bottom, self.left : self.right]


@dataclass(frozen=True)
class BoardRegion:
    rect: Rect
    method: str  # "refined" or "configured"


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    rect: Rect  # the full cell, in screenshot pixels
    letter_rect: Rect  # the part of the cell holding the big letter, in screenshot pixels


def configured_rect(image_shape: tuple[int, ...], config: VisionConfig = DEFAULT_VISION_CONFIG) -> Rect:
    """The board rectangle from the normalized config values, in pixels."""
    h, w = image_shape[:2]
    return Rect(
        left=round(config.board_left * w),
        top=round(config.board_top * h),
        right=round(config.board_right * w),
        bottom=round(config.board_bottom * h),
    )


def _find_tiles(image: np.ndarray, window: Rect, cell_w: float, cell_h: float) -> list[Rect]:
    """Find light, low-saturation, roughly cell-sized blobs inside ``window``."""
    hsv = cv2.cvtColor(window.crop(image), cv2.COLOR_BGR2HSV)
    light = ((hsv[..., 1] < 70) & (hsv[..., 2] > 140)).astype(np.uint8) * 255
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(light, connectivity=4)
    tiles = []
    for x, y, bw, bh, _area in stats[1:].tolist():
        if 0.7 * cell_w <= bw <= 1.05 * cell_w and 0.7 * cell_h <= bh <= 1.05 * cell_h:
            tiles.append(Rect(window.left + x, window.top + y, window.left + x + bw, window.top + y + bh))
    return tiles


def detect_board(image: np.ndarray, config: VisionConfig = DEFAULT_VISION_CONFIG) -> BoardRegion:
    """Return the board rectangle for ``image``."""
    base = configured_rect(image.shape, config)
    if not config.refine_board:
        return BoardRegion(base, "configured")

    h, w = image.shape[:2]
    mx = round(base.width * config.refine_margin)
    my = round(base.height * config.refine_margin)
    window = Rect(max(0, base.left - mx), max(0, base.top - my), min(w, base.right + mx), min(h, base.bottom + my))
    expected = config.grid_rows * config.grid_cols
    tiles = _find_tiles(image, window, base.width / config.grid_cols, base.height / config.grid_rows)
    if len(tiles) != expected:
        return BoardRegion(base, "configured")

    rect = Rect(
        left=min(t.left for t in tiles),
        top=min(t.top for t in tiles),
        right=max(t.right for t in tiles),
        bottom=max(t.bottom for t in tiles),
    )
    return BoardRegion(rect, "refined")


def split_cells(board: Rect, config: VisionConfig = DEFAULT_VISION_CONFIG) -> list[Cell]:
    """Divide the board into ``grid_rows x grid_cols`` equal cells, row by row."""
    cell_w = board.width / config.grid_cols
    cell_h = board.height / config.grid_rows
    cells = []
    for r in range(config.grid_rows):
        for c in range(config.grid_cols):
            x0 = board.left + c * cell_w
            y0 = board.top + r * cell_h
            rect = Rect(round(x0), round(y0), round(x0 + cell_w), round(y0 + cell_h))
            letter_rect = Rect(
                round(x0 + config.letter_left * cell_w),
                round(y0 + config.letter_top * cell_h),
                round(x0 + config.letter_right * cell_w),
                round(y0 + config.letter_bottom * cell_h),
            )
            cells.append(Cell(r, c, rect, letter_rect))
    return cells
