"""Shared test helpers."""

from __future__ import annotations

from word_solver.config import DEFAULT_CONFIG, SolverConfig
from word_solver.solver import SolveResult, Solver
from word_solver.trie import Trie

FILLER = "Z"


def make_board(placements: dict[tuple[int, int], str], size: int = 4) -> list[list[str]]:
    """Build a board of filler tiles with letters placed at the given (row, col) cells."""
    board = [[FILLER] * size for _ in range(size)]
    for (r, c), letter in placements.items():
        board[r][c] = letter
    return board


def solve(board, words, config: SolverConfig = DEFAULT_CONFIG) -> SolveResult:
    return Solver(Trie(words), config).solve(board)


def found_words(result: SolveResult) -> dict[str, list[list[int]]]:
    return {r.word: r.path for r in result.words}
