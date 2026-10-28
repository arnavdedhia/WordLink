"""Settings for reading a board from a screenshot.

Positions are normalized fractions (0.0 to 1.0) so they work at any
screenshot resolution. Copy and tweak with ``dataclasses.replace``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VISION_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = VISION_DIR / "templates"
DEBUG_DIR = VISION_DIR / "debug"


@dataclass(frozen=True)
class VisionConfig:
    # Board rectangle as fractions of the screenshot width (left/right) and
    # height (top/bottom). The rectangle covers the 16 tiles edge to edge,
    # excluding the dark frame around them.
    board_left: float = 0.0975
    board_top: float = 0.3643
    board_right: float = 0.9025
    board_bottom: float = 0.7363

    grid_rows: int = 4
    grid_cols: int = 4

    # If True, look for the 16 light tiles near the configured rectangle and
    # snap the board to them. Falls back to the configured rectangle if the
    # tiles cannot be found.
    refine_board: bool = True
    # How far (fraction of board size) beyond the configured rectangle to search when refining.
    refine_margin: float = 0.08

    # Part of each cell that contains the big letter, as fractions of the cell.
    # The bottom edge stops above the row of dots under the letter.
    letter_left: float = 0.15
    letter_top: float = 0.12
    letter_right: float = 0.85
    letter_bottom: float = 0.70

    # Grayscale threshold for "ink". None means choose automatically (Otsu).
    ink_threshold: int | None = None
    # Ink blobs smaller than this fraction of the largest blob are discarded (dots, noise).
    min_blob_fraction: float = 0.15

    # Side length (pixels) of the normalized glyph images and templates.
    glyph_size: int = 32
    # Gaussian blur (sigma, pixels) applied to glyphs and templates before
    # matching, so small differences in stroke weight and alignment matter less. 0 disables.
    match_blur_sigma: float = 1.0

    # A cell is flagged when its best match scores below this value...
    min_confidence: float = 0.80
    # ...or when the runner-up letter scores within this margin of the best.
    min_margin: float = 0.07

    templates_dir: Path = TEMPLATES_DIR
    debug_dir: Path = DEBUG_DIR


DEFAULT_VISION_CONFIG = VisionConfig()
