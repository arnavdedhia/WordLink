"""Application state for the web interface, independent of FastAPI.

``BoardService`` owns the loaded dictionary and the current board. Each
cell keeps where its letter came from (screenshot recognition or a manual
correction) and how confident recognition was, so the UI can warn about
uncertain tiles. Every change re-runs the solver and returns a fresh
snapshot via ``state()``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from word_solver.config import DEFAULT_CONFIG, SolverConfig
from word_solver.dictionary import WordDictionary, normalize_word
from word_solver.solver import SolveResult, Solver

UNKNOWN = "?"
MANUAL = "manual"


@dataclass
class CellState:
    letter: str
    confidence: float | None = None  # None for letters typed by hand
    flagged: bool = False
    reason: str | None = None
    source: str = MANUAL  # "sample", "synthetic", or "manual"
    runner_up: str | None = None

    def to_dict(self) -> dict:
        return {
            "letter": self.letter,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "flagged": self.flagged,
            "reason": self.reason,
            "source": self.source,
            "runner_up": self.runner_up,
        }


@dataclass
class RecognitionInfo:
    image_name: str
    board_method: str
    debug_dir: str | None
    elapsed_ms: float


@dataclass
class BoardState:
    cells: list[list[CellState]]
    origin: str  # "screenshot", "iphone", or "manual"
    updated_at: datetime
    recognition: RecognitionInfo | None = None
    solve: SolveResult | None = None
    solve_error: str | None = None
    timings: dict | None = None

    @property
    def board(self) -> list[list[str]]:
        return [[c.letter for c in row] for row in self.cells]

    @property
    def flagged(self) -> list[list[int]]:
        return [[r, c] for r, row in enumerate(self.cells) for c, cell in enumerate(row) if cell.flagged]


class BoardService:
    def __init__(self, solver_config: SolverConfig = DEFAULT_CONFIG, top_n: int = 11, vision_config=None) -> None:
        self.config = solver_config
        self.vision_config = vision_config  # None means the vision package defaults
        self.top_n = top_n
        load_start = time.perf_counter()
        trie = WordDictionary.from_config(solver_config).build_trie(
            solver_config.min_word_length, solver_config.max_word_length
        )
        self.dictionary_size = len(trie)
        self.dictionary_load_ms = (time.perf_counter() - load_start) * 1000
        self.solver = Solver(trie, solver_config)
        self._state: BoardState | None = None
        self._lock = threading.Lock()

    # ----- updates -------------------------------------------------------

    def load_screenshot(self, data: bytes, image_name: str, debug: bool = False, origin: str = "screenshot") -> dict:
        """Recognize the board in an image (upload, file, or iPhone PNG bytes) and solve it."""
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The uploaded file is not a readable image.")
        return self.ingest_bgr(image, image_name, origin=origin, debug=debug)

    def ingest_bgr(self, image, image_name: str, origin: str = "screenshot", debug: bool = False, capture_ms: float = 0.0) -> dict:
        """Recognize and solve a BGR image already in memory. Used by live capture."""
        from word_solver.vision.board_reader import read_board
        from word_solver.vision.config import DEFAULT_VISION_CONFIG
        from word_solver.vision.debug import save_debug

        vision_config = self.vision_config or DEFAULT_VISION_CONFIG
        start = time.perf_counter()
        reading = read_board(image, vision_config)
        recognize_ms = (time.perf_counter() - start) * 1000
        debug_dir = None
        if debug:
            run_name = f"{Path(image_name).stem}_{datetime.now():%Y%m%d_%H%M%S}"
            debug_dir = str(save_debug(reading, vision_config, run_name=run_name))

        cells = [[CellState(UNKNOWN) for _ in range(reading.cols)] for _ in range(reading.rows)]
        for cell in reading.cells:
            rec = cell.recognition
            cells[cell.row][cell.col] = CellState(
                letter=rec.letter,
                confidence=rec.confidence,
                flagged=rec.flagged,
                reason=rec.reason,
                source=rec.source or "none",
                runner_up=rec.runner_up,
            )
        info = RecognitionInfo(image_name, reading.region.method, debug_dir, recognize_ms)
        state = BoardState(cells, origin, datetime.now(), info)
        snapshot = self._replace(state)
        solve_ms = state.solve.elapsed_seconds * 1000 if state.solve else 0.0
        timings = {
            "capture_ms": round(capture_ms, 1),
            "recognize_ms": round(recognize_ms, 1),
            "solve_ms": round(solve_ms, 3),
            "total_ms": round(capture_ms + recognize_ms + solve_ms, 1),
        }
        with self._lock:
            if self._state is state:
                self._state.timings = timings
        snapshot["timings"] = timings
        snapshot["recognize_ms"] = timings["recognize_ms"]
        snapshot["solve_ms"] = timings["solve_ms"]
        return snapshot

    def set_board(self, rows: list[str] | list[list[str]]) -> dict:
        """Replace the whole board with letters typed by hand."""
        grid = [list(row) for row in rows]
        if len(grid) != self.config.rows or any(len(r) != self.config.cols for r in grid):
            raise ValueError(f"Board must be {self.config.rows} rows of {self.config.cols} letters.")
        cells = [[CellState(self._clean_letter(ch)) for ch in row] for row in grid]
        return self._replace(BoardState(cells, "manual", datetime.now()))

    def set_cell(self, row: int, col: int, letter: str) -> dict:
        """Correct (or confirm) one tile. The tile is no longer flagged afterwards."""
        with self._lock:
            state = self._require_state()
            if not (0 <= row < self.config.rows and 0 <= col < self.config.cols):
                raise ValueError(f"Cell [{row}, {col}] is outside the board.")
            old = state.cells[row][col]
            state.cells[row][col] = CellState(
                letter=self._clean_letter(letter),
                confidence=old.confidence,
                source=MANUAL,
            )
            state.updated_at = datetime.now()
            self._solve(state)
            return self._snapshot(state)

    # ----- reads ---------------------------------------------------------

    def state(self) -> dict:
        with self._lock:
            return self._snapshot(self._state) if self._state else self._empty()

    # ----- internals -----------------------------------------------------

    def _clean_letter(self, letter: str) -> str:
        word = normalize_word(letter)
        if word is None or len(word) != 1:
            raise ValueError(f"A tile must be a single letter A-Z, got {letter!r}.")
        return word

    def _require_state(self) -> BoardState:
        if self._state is None:
            raise LookupError("No board loaded yet.")
        return self._state

    def _replace(self, state: BoardState) -> dict:
        with self._lock:
            self._solve(state)
            self._state = state
            return self._snapshot(state)

    def _solve(self, state: BoardState) -> None:
        unknown = [[r, c] for r, row in enumerate(state.cells) for c, cell in enumerate(row) if cell.letter == UNKNOWN]
        if unknown:
            state.solve = None
            state.solve_error = "Some tiles could not be read. Set their letters to solve the board."
            return
        state.solve = self.solver.solve(state.board)
        state.solve_error = None

    def _empty(self) -> dict:
        return {
            "loaded": False,
            "dictionary_size": self.dictionary_size,
        }

    def _snapshot(self, state: BoardState) -> dict:
        recognized = [c.confidence for row in state.cells for c in row if c.confidence is not None and c.source != MANUAL]
        solve = None
        if state.solve is not None:
            solve = {
                "words": [w.to_dict() for w in state.solve.top(self.top_n)],
                "best_length": state.solve.best_words[0].length if state.solve.best_words else 0,
                "best_count": len(state.solve.best_words),
                "total": state.solve.total,
                "elapsed_ms": round(state.solve.elapsed_seconds * 1000, 3),
            }
        recognition = None
        if state.recognition is not None:
            r = state.recognition
            recognition = {
                "image_name": r.image_name,
                "board_method": r.board_method,
                "debug_dir": r.debug_dir,
                "elapsed_ms": round(r.elapsed_ms, 1),
            }
        return {
            "loaded": True,
            "origin": state.origin,
            "board": state.board,
            "cells": [[c.to_dict() for c in row] for row in state.cells],
            "flagged": state.flagged,
            "reliable": not state.flagged and state.solve_error is None,
            "min_confidence": round(min(recognized), 4) if recognized else None,
            "updated_at": state.updated_at.isoformat(timespec="seconds"),
            "recognition": recognition,
            "solve": solve,
            "solve_error": state.solve_error,
            "dictionary_size": self.dictionary_size,
            "timings": state.timings,
        }
