"""Board search: depth-first search from every tile, pruned by a Trie.

The search never builds a letter sequence unless the Trie confirms that some
dictionary word starts with it, so dead-end paths are abandoned after a
single failed child lookup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from word_solver.config import DEFAULT_CONFIG, SolverConfig
from word_solver.trie import Trie, TrieNode

Board = Sequence[Sequence[str]]
Coordinate = list[int]  # [row, col], zero-based


@dataclass
class WordResult:
    """A valid word and one board path that spells it."""

    word: str
    length: int
    path: list[Coordinate]

    def to_dict(self) -> dict:
        return {"word": self.word, "length": self.length, "path": [list(p) for p in self.path]}


@dataclass
class SolveResult:
    """Everything found on a board, ranked best first."""

    words: list[WordResult]
    elapsed_seconds: float
    best_words: list[WordResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.words)

    def top(self, n: int) -> list[WordResult]:
        return self.words[:n]


def rank_key(result: WordResult) -> tuple:
    """Sort key for results: longest first, then alphabetical.

    Change this (and ``select_best``) to switch the game's objective,
    e.g. to a point-based score.
    """
    return (-result.length, result.word)


def select_best(ranked: list[WordResult]) -> list[WordResult]:
    """Return every result tied with the top-ranked one (all longest words)."""
    if not ranked:
        return []
    best_length = ranked[0].length
    return [r for r in ranked if r.length == best_length]


def normalize_board(board: Board, config: SolverConfig = DEFAULT_CONFIG) -> list[list[str]]:
    """Validate the board shape and uppercase every tile.

    A tile is normally one letter, but multi-letter tiles such as "QU" are
    accepted and matched letter by letter.
    """
    if len(board) != config.rows:
        raise ValueError(f"Board must have {config.rows} rows, got {len(board)}.")
    normalized: list[list[str]] = []
    for r, row in enumerate(board):
        if len(row) != config.cols:
            raise ValueError(f"Row {r} must have {config.cols} tiles, got {len(row)}.")
        new_row = []
        for c, tile in enumerate(row):
            text = str(tile).strip().upper()
            if not text.isalpha() or not text.isascii():
                raise ValueError(f"Tile at [{r}, {c}] must contain only letters A-Z, got {tile!r}.")
            new_row.append(text)
        normalized.append(new_row)
    return normalized


def build_neighbors(rows: int, cols: int, offsets: Sequence[tuple[int, int]]) -> list[list[int]]:
    """For each cell index (row * cols + col), list the indexes of its in-bounds neighbors."""
    neighbors: list[list[int]] = []
    for r in range(rows):
        for c in range(cols):
            cell_neighbors = []
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    cell_neighbors.append(nr * cols + nc)
            neighbors.append(cell_neighbors)
    return neighbors


def _advance(node: TrieNode, tile: str) -> TrieNode | None:
    """Follow every letter of ``tile`` down the Trie; None if the prefix dies."""
    for letter in tile:
        node = node.children.get(letter)
        if node is None:
            return None
    return node


class Solver:
    """Finds every dictionary word that can be traced on a board."""

    def __init__(self, trie: Trie, config: SolverConfig = DEFAULT_CONFIG) -> None:
        self.trie = trie
        self.config = config
        self._neighbors = build_neighbors(config.rows, config.cols, config.neighbor_offsets)

    def solve(self, board: Board) -> SolveResult:
        start = time.perf_counter()
        grid = normalize_board(board, self.config)
        cols = self.config.cols
        tiles = [tile for row in grid for tile in row]
        found: dict[str, WordResult] = {}

        min_len = self.config.min_word_length
        max_len = self.config.max_word_length
        allow_reuse = self.config.allow_tile_reuse
        neighbors = self._neighbors

        # path holds cell indexes; `visited` is a bitmask of cells used in the path.
        path: list[int] = []

        def dfs(cell: int, node: TrieNode, word: str, visited: int) -> None:
            path.append(cell)
            if node.is_word and len(word) >= min_len and word not in found:
                found[word] = WordResult(
                    word=word,
                    length=len(word),
                    path=[[i // cols, i % cols] for i in path],
                )
            if max_len is None or len(word) < max_len:
                for nxt in neighbors[cell]:
                    bit = 1 << nxt
                    if not allow_reuse and visited & bit:
                        continue
                    child = _advance(node, tiles[nxt])
                    if child is None:
                        continue  # Trie pruning: no word has this prefix.
                    dfs(nxt, child, word + tiles[nxt], visited | bit)
            path.pop()

        for cell, tile in enumerate(tiles):
            node = _advance(self.trie.root, tile)
            if node is not None:
                dfs(cell, node, tile, 1 << cell)

        ranked = sorted(found.values(), key=rank_key)
        elapsed = time.perf_counter() - start
        return SolveResult(words=ranked, elapsed_seconds=elapsed, best_words=select_best(ranked))
