"""One-off solver measurements. Run: python -m word_solver.tests.bench_solver"""

from __future__ import annotations

import random
import statistics
import time

from word_solver.config import DEFAULT_BOARD, DEFAULT_CONFIG
from word_solver.dictionary import WordDictionary
from word_solver.solver import Solver

LETTERS = "EEEEAAAARRRIIIOOOTTTNNSSLLUUUDDGGCCMMBBPPFFHHVVWWYYKJXQZ"


def random_board(rng: random.Random) -> list[list[str]]:
    tiles = [rng.choice(LETTERS) for _ in range(16)]
    return [tiles[i : i + 4] for i in range(0, 16, 4)]


def median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def run(solver: Solver, name: str, board: list[list[str]], repeats: int = 25) -> float:
    solver.solve(board)
    searches: list[float] = []
    ranks: list[float] = []
    totals: list[float] = []
    last = None
    for _ in range(repeats):
        last = solver.solve(board)
        searches.append(last.stats.search_ms)
        ranks.append(last.stats.rank_ms)
        totals.append(last.stats.total_ms)
    assert last is not None
    top = last.words[0].word if last.words else "-"
    top_score = last.words[0].score if last.words else 0
    print(name)
    print(f"  Board search:           {median(searches):.2f} ms  (min {min(searches):.2f}, max {max(searches):.2f})")
    print(f"  Candidate ranking:      {median(ranks):.3f} ms")
    print(f"  Total solver execution: {median(totals):.2f} ms")
    print(f"  DFS states visited:     {last.stats.states_visited:,}")
    print(f"  Trie-pruned branches:   {last.stats.trie_pruned:,}")
    print(f"  Valid words discovered: {last.stats.words_found:,}")
    print(f"  Top word: {top} ({top_score} pts)")
    print()
    return median(searches)


def main() -> None:
    t0 = time.perf_counter()
    dictionary = WordDictionary.from_config(DEFAULT_CONFIG)
    dict_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    trie = dictionary.build_trie(DEFAULT_CONFIG.min_word_length, DEFAULT_CONFIG.max_word_length)
    trie_ms = (time.perf_counter() - t1) * 1000
    print(f"Dictionary words: {len(trie):,}")
    print(f"Dictionary loading: {dict_ms:.1f} ms")
    print(f"Trie construction: {trie_ms:.1f} ms")
    print()
    solver = Solver(trie, DEFAULT_CONFIG)
    run(solver, "Sample board OSYE/YYEM/OTPE/FUND", DEFAULT_BOARD)
    rng = random.Random(42)
    for i in range(5):
        run(solver, f"Random board {i + 1}", random_board(rng), repeats=15)


if __name__ == "__main__":
    main()
