"""Screenshot to board: detect the board, cut 16 cells, recognize each letter."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from word_solver.vision.board_detector import BoardRegion, Cell, detect_board, split_cells
from word_solver.vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from word_solver.vision.image_processor import IsolatedCharacter, extract_glyph
from word_solver.vision.letter_recognizer import LetterRecognizer, Recognition, TemplateMatchRecognizer
from word_solver.vision.templates import TemplateLibrary


@dataclass
class CellImage:
    """Images for one cell, before recognition."""

    cell: Cell
    tile: np.ndarray  # full cell crop (BGR)
    letter_crop: np.ndarray  # letter region of the cell (BGR)
    isolated: IsolatedCharacter
    glyph: np.ndarray  # normalized glyph fed to the recognizer

    @property
    def row(self) -> int:
        return self.cell.row

    @property
    def col(self) -> int:
        return self.cell.col


@dataclass
class CellExtraction:
    image: np.ndarray
    region: BoardRegion
    cells: list[CellImage]


@dataclass
class CellReading:
    image: CellImage
    recognition: Recognition

    @property
    def row(self) -> int:
        return self.image.row

    @property
    def col(self) -> int:
        return self.image.col


@dataclass
class BoardReading:
    image: np.ndarray
    region: BoardRegion
    cells: list[CellReading]
    rows: int
    cols: int
    flagged: list[CellReading] = field(default_factory=list)

    @property
    def board(self) -> list[list[str]]:
        grid = [["?"] * self.cols for _ in range(self.rows)]
        for cell in self.cells:
            grid[cell.row][cell.col] = cell.recognition.letter
        return grid

    @property
    def confidences(self) -> list[list[float]]:
        grid = [[0.0] * self.cols for _ in range(self.rows)]
        for cell in self.cells:
            grid[cell.row][cell.col] = cell.recognition.confidence
        return grid

    @property
    def is_reliable(self) -> bool:
        return not self.flagged


def extract_cells(image: np.ndarray, config: VisionConfig = DEFAULT_VISION_CONFIG) -> CellExtraction:
    """Detect the board and produce the processed images for all cells."""
    region = detect_board(image, config)
    cells = []
    for cell in split_cells(region.rect, config):
        letter_crop = cell.letter_rect.crop(image)
        glyph, isolated = extract_glyph(letter_crop, config)
        cells.append(CellImage(cell, cell.rect.crop(image), letter_crop, isolated, glyph))
    return CellExtraction(image, region, cells)


def read_board(
    image: np.ndarray,
    config: VisionConfig = DEFAULT_VISION_CONFIG,
    recognizer: LetterRecognizer | None = None,
) -> BoardReading:
    """Recognize every cell of the board in ``image``."""
    if recognizer is None:
        recognizer = TemplateMatchRecognizer(TemplateLibrary.load(config), config)
    extraction = extract_cells(image, config)
    readings = [
        CellReading(c, recognizer.recognize(c.glyph, c.isolated.component_count))
        for c in extraction.cells
    ]
    return BoardReading(
        image=image,
        region=extraction.region,
        cells=readings,
        rows=config.grid_rows,
        cols=config.grid_cols,
        flagged=[r for r in readings if r.recognition.flagged],
    )
