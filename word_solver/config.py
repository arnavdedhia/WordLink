"""Central configuration for the word solver.

Every game rule lives here so it can be changed in one place without
touching the search code. ``SolverConfig`` is an immutable dataclass; create
a modified copy with ``dataclasses.replace(DEFAULT_CONFIG, min_word_length=4)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"

# (row_delta, col_delta) offsets for each movement style.
ORTHOGONAL_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)
DIAGONAL_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
ALL_OFFSETS = ORTHOGONAL_OFFSETS + DIAGONAL_OFFSETS


@dataclass(frozen=True)
class SolverConfig:
    """Game rules and solver settings."""

    # Board dimensions. Boards that do not match are rejected.
    rows: int = 4
    cols: int = 4

    # Which neighboring tiles a path may step to.
    neighbor_offsets: tuple[tuple[int, int], ...] = ALL_OFFSETS

    # Whether a tile may appear more than once in the same word.
    allow_tile_reuse: bool = False

    # Shortest word (in letters) that counts as valid.
    min_word_length: int = 3

    # Optional cap on word length; None means unlimited.
    max_word_length: int | None = None

    # How many ranked words to show in the report.
    top_n: int = 10

    # Dictionary files.
    words_path: Path = DATA_DIR / "words.txt"
    allowed_words_path: Path = DATA_DIR / "allowed_words.txt"
    blocked_words_path: Path = DATA_DIR / "blocked_words.txt"


DEFAULT_CONFIG = SolverConfig()

DEFAULT_BOARD: list[list[str]] = [
    ["O", "S", "Y", "E"],
    ["Y", "Y", "E", "M"],
    ["O", "T", "P", "E"],
    ["F", "U", "N", "D"],
]
