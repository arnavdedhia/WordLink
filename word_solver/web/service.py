"""Application state for the web interface, independent of FastAPI.

``BoardService`` owns the loaded dictionary and the current board. Ranked
suggestions for a board are computed once and cached. Rejecting or skipping
a word only moves an index in that cache — it never recaptures, rereads, or
resolves the board.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from word_solver.config import DEFAULT_CONFIG, SolverConfig
from word_solver.dictionary import BlockedWordList, WordDictionary, normalize_word
from word_solver.solver import SolveResult, Solver, WordResult, select_best

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
    """Cached board + ranked suggestions. This is the current-board cache."""

    cells: list[list[CellState]]
    origin: str  # "screenshot", "iphone", or "manual"
    updated_at: datetime
    board_id: int = 0
    suggestion_index: int = 0
    recognized_at: datetime | None = None
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

    @property
    def recognition_confidence(self) -> float | None:
        values = [c.confidence for row in self.cells for c in row if c.confidence is not None and c.source != MANUAL]
        return min(values) if values else None


class BoardService:
    def __init__(self, solver_config: SolverConfig = DEFAULT_CONFIG, top_n: int = 11, vision_config=None) -> None:
        self.config = solver_config
        self.vision_config = vision_config  # None means the vision package defaults
        self.top_n = top_n
        load_start = time.perf_counter()
        dictionary = WordDictionary.from_config(solver_config)
        self.dictionary_load_ms = (time.perf_counter() - load_start) * 1000
        trie_start = time.perf_counter()
        trie = dictionary.build_trie(solver_config.min_word_length, solver_config.max_word_length)
        self.trie_build_ms = (time.perf_counter() - trie_start) * 1000
        self.dictionary_size = len(trie)
        self.solver = Solver(trie, solver_config)
        self.blocked = BlockedWordList(solver_config.blocked_words_path)
        self._rejection_stack: list[str] = []
        self._generation = 0
        self._update_seq = 0
        self._next_board_id = 1
        self._state: BoardState | None = None
        self._lock = threading.Lock()

    def _bump_generation(self) -> int:
        with self._lock:
            self._generation += 1
            return self._generation

    # ----- updates -------------------------------------------------------

    def load_screenshot(self, data: bytes, image_name: str, debug: bool = False, origin: str = "screenshot") -> dict:
        """Recognize the board in an image (upload, file, or iPhone PNG bytes) and solve it."""
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The uploaded file is not a readable image.")
        return self.ingest_bgr(image, image_name, origin=origin, debug=debug)

    def ingest_bgr(
        self,
        image,
        image_name: str,
        origin: str = "screenshot",
        debug: bool = False,
        capture_ms: float = 0.0,
        stabilize_ms: float = 0.0,
    ) -> dict:
        """Recognize and solve a BGR image already in memory. Used by live capture."""
        from word_solver.vision.board_reader import read_board
        from word_solver.vision.config import DEFAULT_VISION_CONFIG
        from word_solver.vision.debug import save_debug

        gen = self._bump_generation()
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
        now = datetime.now()
        info = RecognitionInfo(image_name, reading.region.method, debug_dir, recognize_ms)
        state = BoardState(cells, origin, now, recognized_at=now, recognition=info)
        solve_start = time.perf_counter()
        self._solve(state, debug=debug)
        solve_ms = (time.perf_counter() - solve_start) * 1000
        if state.solve is not None:
            solve_ms = state.solve.elapsed_seconds * 1000
        timings = {
            "capture_ms": round(capture_ms, 1),
            "stabilize_ms": round(stabilize_ms, 1),
            "recognize_ms": round(recognize_ms, 1),
            "solve_ms": round(solve_ms, 3),
            "total_ms": round(capture_ms + stabilize_ms + recognize_ms + solve_ms, 1),
        }
        state.timings = timings
        snapshot = self._apply_recognized(gen, state)
        _log_board_timings(timings)
        snapshot["timings"] = timings
        snapshot["recognize_ms"] = timings["recognize_ms"]
        snapshot["solve_ms"] = timings["solve_ms"]
        return snapshot

    def set_board(self, rows: list[str] | list[list[str]]) -> dict:
        """Replace the whole board with letters typed by hand."""
        gen = self._bump_generation()
        grid = [list(row) for row in rows]
        if len(grid) != self.config.rows or any(len(r) != self.config.cols for r in grid):
            raise ValueError(f"Board must be {self.config.rows} rows of {self.config.cols} letters.")
        cells = [[CellState(self._clean_letter(ch)) for ch in row] for row in grid]
        now = datetime.now()
        state = BoardState(cells, "manual", now, recognized_at=now)
        self._solve(state)
        return self._apply_recognized(gen, state)

    def set_cell(self, row: int, col: int, letter: str) -> dict:
        """Correct (or confirm) one tile. The tile is no longer flagged afterwards."""
        gen = self._bump_generation()
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
            state.suggestion_index = 0
            state.board_id = self._next_board_id
            self._next_board_id += 1
            self._solve(state)
            self._update_seq += 1
            return self._snapshot(state)

    def reject_current(self) -> dict:
        """Permanently block the current suggestion and show the next one. No capture/CV/solve."""
        started = time.perf_counter()
        with self._lock:
            state = self._require_state()
            current = self._current_word(state)
            if current is None:
                raise LookupError("No suggestion to reject.")
            word = current.word
            _, _, persist_ms = self.blocked.add(word)
            self._rejection_stack.append(word)
            state.suggestion_index = self._advance_index(state, state.suggestion_index + 1)
            self._update_seq += 1
            switch_ms = (time.perf_counter() - started) * 1000
            nxt = self._current_word(state)
            print(f"Rejected: {word}", flush=True)
            print(f"Next suggestion: {nxt.word if nxt else '-'}", flush=True)
            print(f"Persist rejection: {persist_ms:.1f} ms", flush=True)
            print(f"Suggestion switch: {switch_ms:.1f} ms", flush=True)
            snap = self._snapshot(state)
            snap["rejected_word"] = word
            snap["persist_ms"] = round(persist_ms, 2)
            snap["switch_ms"] = round(switch_ms, 2)
            return snap

    def next_word(self) -> dict:
        """Skip the current suggestion for this board only. Does not block it."""
        started = time.perf_counter()
        with self._lock:
            state = self._require_state()
            if not state.solve or not state.solve.words:
                raise LookupError("No suggestions on this board.")
            state.suggestion_index = self._advance_index(state, state.suggestion_index + 1)
            self._update_seq += 1
            switch_ms = (time.perf_counter() - started) * 1000
            snap = self._snapshot(state)
            snap["switch_ms"] = round(switch_ms, 2)
            return snap

    def undo_rejection(self) -> dict:
        """Unblock the most recently rejected word. Does not change the current suggestion."""
        with self._lock:
            if not self._rejection_stack:
                raise LookupError("Nothing to undo.")
            word = self._rejection_stack.pop()
            self.blocked.remove(word)
            self._update_seq += 1
            snap = self._snapshot(self._state) if self._state else self._empty()
            snap["undone_word"] = word
            return snap

    # ----- reads ---------------------------------------------------------

    def state(self) -> dict:
        with self._lock:
            return self._snapshot(self._state) if self._state else self._empty()

    # ----- internals -----------------------------------------------------

    def _clean_letter(self, letter: str) -> str:
        from word_solver.vision.templates import normalize_tile_label

        tile = normalize_tile_label(letter)
        if tile is None:
            raise ValueError(f"A tile must be a letter A-Z or Qu, got {letter!r}.")
        return tile

    def _require_state(self) -> BoardState:
        if self._state is None:
            raise LookupError("No board loaded yet.")
        return self._state

    def _apply_recognized(self, gen: int, state: BoardState) -> dict:
        """Install a newly solved board, ignoring stale async jobs and identical re-reads."""
        with self._lock:
            if gen != self._generation:
                return self._snapshot(self._state) if self._state else self._empty()
            if self._state is not None and self._state.board == state.board and self._state.solve is not None:
                if state.timings:
                    self._state.timings = state.timings
                if state.recognition:
                    self._state.recognition = state.recognition
                return self._snapshot(self._state)
            state.board_id = self._next_board_id
            self._next_board_id += 1
            state.suggestion_index = 0
            self._state = state
            self._update_seq += 1
            return self._snapshot(state)

    def _solve(self, state: BoardState, debug: bool = False) -> None:
        unknown = [[r, c] for r, row in enumerate(state.cells) for c, cell in enumerate(row) if cell.letter == UNKNOWN]
        if unknown:
            state.solve = None
            state.solve_error = "Some tiles could not be read. Set their letters to solve the board."
            return
        result = self.solver.solve(state.board)
        result.stats.dictionary_load_ms = self.dictionary_load_ms
        result.stats.trie_build_ms = self.trie_build_ms
        state.solve = self._filter_blocked(result)
        state.solve_error = None
        if debug:
            result.stats.print()

    def _filter_blocked(self, result: SolveResult) -> SolveResult:
        words = [w for w in result.words if w.word not in self.blocked.words]
        return SolveResult(
            words=words,
            elapsed_seconds=result.elapsed_seconds,
            best_words=select_best(words),
            stats=result.stats,
        )

    def _advance_index(self, state: BoardState, start: int) -> int:
        """First index at or after ``start`` whose word is not blocked. Clamps at last item."""
        words = state.solve.words if state.solve else []
        if not words:
            return 0
        idx = start
        while idx < len(words) and words[idx].word in self.blocked.words:
            idx += 1
        if idx >= len(words):
            idx = len(words) - 1
            while idx > 0 and words[idx].word in self.blocked.words:
                idx -= 1
        return idx

    def _current_word(self, state: BoardState) -> WordResult | None:
        if state.solve is None or not state.solve.words:
            return None
        idx = self._advance_index(state, state.suggestion_index)
        word = state.solve.words[idx]
        if word.word in self.blocked.words:
            return None
        return word

    def _empty(self) -> dict:
        return {
            "loaded": False,
            "dictionary_size": self.dictionary_size,
            "board_id": None,
            "update_seq": self._update_seq,
            "current_suggestion": None,
            "suggestion_index": 0,
            "total_suggestions": 0,
            "next_suggestions": [],
            "can_undo": bool(self._rejection_stack),
        }

    def _snapshot(self, state: BoardState | None) -> dict:
        if state is None:
            return self._empty()
        recognized = [c.confidence for row in state.cells for c in row if c.confidence is not None and c.source != MANUAL]
        words = state.solve.words if state.solve else []
        idx = self._advance_index(state, state.suggestion_index) if words else 0
        current = self._current_word(state)
        remaining = [w for w in words[idx:] if w.word not in self.blocked.words] if words else []
        next_three = remaining[1:4]
        solve = None
        if state.solve is not None:
            solve = {
                "words": [w.to_dict() for w in remaining[: self.top_n]],
                "best_length": remaining[0].length if remaining else 0,
                "best_score": remaining[0].score if remaining else 0,
                "best_count": sum(1 for w in remaining if remaining and w.score == remaining[0].score),
                "total": len(remaining),
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
            "board_id": state.board_id,
            "update_seq": self._update_seq,
            "cells": [[c.to_dict() for c in row] for row in state.cells],
            "flagged": state.flagged,
            "reliable": not state.flagged and state.solve_error is None,
            "min_confidence": round(min(recognized), 4) if recognized else None,
            "recognition_confidence": round(state.recognition_confidence, 4) if state.recognition_confidence is not None else None,
            "updated_at": state.updated_at.isoformat(timespec="seconds"),
            "recognized_at": (state.recognized_at or state.updated_at).isoformat(timespec="seconds"),
            "recognition": recognition,
            "solve": solve,
            "solve_error": state.solve_error,
            "dictionary_size": self.dictionary_size,
            "timings": state.timings,
            "current_suggestion": current.to_dict() if current else None,
            "suggestion_index": (idx + 1) if remaining else 0,
            "total_suggestions": len(words),
            "next_suggestions": [w.to_dict() for w in next_three],
            "can_undo": bool(self._rejection_stack),
        }


def _log_board_timings(timings: dict) -> None:
    print(f"Screen capture: {timings.get('capture_ms', 0):.0f} ms", flush=True)
    print(f"Board stabilization: {timings.get('stabilize_ms', 0):.0f} ms", flush=True)
    print(f"Letter recognition: {timings.get('recognize_ms', 0):.0f} ms", flush=True)
    print(f"Word solving: {timings.get('solve_ms', 0):.0f} ms", flush=True)
    print(f"Total new board processing: {timings.get('total_ms', 0):.0f} ms", flush=True)
