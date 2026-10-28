"""Board search: depth-first search from every tile, pruned by a Trie.

The search never builds a letter sequence unless the Trie confirms that some
dictionary word starts with it, so dead-end paths are abandoned after a
single failed child lookup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from word_solver.config import DEFAULT_CONFIG, SolverConfig, word_score
from word_solver.trie import Trie, TrieNode

Board = Sequence[Sequence[str]]
Coordinate = list[int]  # [row, col], zero-based


@dataclass
class WordResult:
    """A valid word, one board path that spells it, and its game score."""

    word: str
    length: int
    path: list[Coordinate]
    letter_score: int = 0
    length_bonus: int = 0
    score: int = 0

    @classmethod
    def make(cls, word: str, path: list[Coordinate]) -> WordResult:
        letters, bonus, total = word_score(word)
        return cls(word, len(word), path, letters, bonus, total)

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "length": self.length,
            "path": [list(p) for p in self.path],
            "letter_score": self.letter_score,
            "length_bonus": self.length_bonus,
            "score": self.score,
        }


@dataclass
class SearchStats:
    """Timings and counters for one solve. Dictionary/Trie times are filled by the caller."""

    dictionary_load_ms: float = 0.0
    trie_build_ms: float = 0.0
    search_ms: float = 0.0
    rank_ms: float = 0.0
    total_ms: float = 0.0
    states_visited: int = 0
    trie_pruned: int = 0
    words_found: int = 0

    def print(self) -> None:
        print("Solver measurements", flush=True)
        print(f"  Dictionary loading:     {self.dictionary_load_ms:.1f} ms", flush=True)
        print(f"  Trie construction:      {self.trie_build_ms:.1f} ms", flush=True)
        print(f"  Board search:           {self.search_ms:.1f} ms", flush=True)
        print(f"  Candidate ranking:      {self.rank_ms:.1f} ms", flush=True)
        print(f"  Total solver execution: {self.total_ms:.1f} ms", flush=True)
        print(f"  DFS states visited:     {self.states_visited:,}", flush=True)
        print(f"  Trie-pruned branches:   {self.trie_pruned:,}", flush=True)
        print(f"  Valid words discovered: {self.words_found:,}", flush=True)


@dataclass
class SolveResult:
    """Everything found on a board, ranked best first."""

    words: list[WordResult]
    elapsed_seconds: float
    best_words: list[WordResult] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)

    @property
    def total(self) -> int:
        return len(self.words)

    def top(self, n: int) -> list[WordResult]:
        return self.words[:n]


def rank_key(result: WordResult) -> tuple:
    """Sort key: highest score first, then longer words, then alphabetical."""
    return (-result.score, -result.length, result.word)


def select_best(ranked: list[WordResult]) -> list[WordResult]:
    """Return every result tied for the highest score."""
    if not ranked:
        return []
    best_score = ranked[0].score
    return [r for r in ranked if r.score == best_score]


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

    def solve(self, board: Board, debug: bool = False) -> SolveResult:
        total_start = time.perf_counter()
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
        states_visited = 0
        trie_pruned = 0

        search_start = time.perf_counter()

        def dfs(cell: int, node: TrieNode, word: str, visited: int) -> None:
            nonlocal states_visited, trie_pruned
            states_visited += 1
            path.append(cell)
            if node.is_word and len(word) >= min_len and word not in found:
                found[word] = WordResult.make(word, [[i // cols, i % cols] for i in path])
            if max_len is None or len(word) < max_len:
                for nxt in neighbors[cell]:
                    bit = 1 << nxt
                    if not allow_reuse and visited & bit:
                        continue
                    child = _advance(node, tiles[nxt])
                    if child is None:
                        trie_pruned += 1
                        continue  # Trie pruning: no word has this prefix.
                    dfs(nxt, child, word + tiles[nxt], visited | bit)
            path.pop()

        for cell, tile in enumerate(tiles):
            node = _advance(self.trie.root, tile)
            if node is None:
                trie_pruned += 1
                continue
            dfs(cell, node, tile, 1 << cell)

        search_ms = (time.perf_counter() - search_start) * 1000
        rank_start = time.perf_counter()
        ranked = sorted(found.values(), key=rank_key)
        rank_ms = (time.perf_counter() - rank_start) * 1000
        elapsed = time.perf_counter() - total_start
        stats = SearchStats(
            search_ms=search_ms,
            rank_ms=rank_ms,
            total_ms=elapsed * 1000,
            states_visited=states_visited,
            trie_pruned=trie_pruned,
            words_found=len(found),
        )
        if debug:
            stats.print()
        return SolveResult(words=ranked, elapsed_seconds=elapsed, best_words=select_best(ranked), stats=stats)
