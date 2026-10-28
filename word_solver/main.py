"""Command-line entry point.

Examples (run from the repository root)::

    python -m word_solver
    python -m word_solver --board OSYE YYEM OTPE FUND
    python -m word_solver --allow FOOBAR --block YOYO
    python -m word_solver --image word_solver/vision/samples/board_osye.jpg --debug
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from word_solver.config import DEFAULT_BOARD, DEFAULT_CONFIG, SolverConfig
from word_solver.dictionary import WordDictionary, append_word
from word_solver.solver import Board, SolveResult, Solver, WordResult


def format_board(board: Board) -> str:
    """Render the board as a bordered grid with row/column indexes."""
    cols = len(board[0])
    width = max(len(t) for row in board for t in row) + 2
    header = "    " + " ".join(str(c).center(width) for c in range(cols))
    border = "   +" + "+".join("-" * width for _ in range(cols)) + "+"
    lines = [header, border]
    for r, row in enumerate(board):
        cells = "|".join(str(t).upper().center(width) for t in row)
        lines.append(f" {r} |{cells}|")
        lines.append(border)
    return "\n".join(lines)


def format_path(result: WordResult) -> str:
    return " -> ".join(f"[{r}, {c}]" for r, c in result.path)


def print_report(board: Board, result: SolveResult, config: SolverConfig) -> None:
    print("\nBoard:")
    print(format_board(board))
    print()

    if not result.words:
        print("No valid words found.")
        print(f"Time to solve: {result.elapsed_seconds * 1000:.2f} ms")
        return

    label = "Best word" if len(result.best_words) == 1 else "Best words (tied)"
    print(f"{label}:")
    for best in result.best_words:
        print(f"  {best.word}")
        print(f"    Length: {best.length}")
        print(f"    Points: {best.score}  (letters {best.letter_score} + length bonus {best.length_bonus})")
        print(f"    Path:   {format_path(best)}")
    print()

    top = result.top(config.top_n)
    print(f"Top {len(top)} highest-scoring words:")
    for i, item in enumerate(top, start=1):
        print(f"  {i:>2}. {item.word:<16} {item.score:>5} pts  ({item.length})  {format_path(item)}")
    print()

    print(f"Total valid words found: {result.total}")
    print(f"Time to solve: {result.elapsed_seconds * 1000:.2f} ms")


def board_from_screenshot(image: Path, debug: bool, accept_uncertain: bool) -> Board:
    """Recognize the board in a screenshot; stop if any letter is uncertain unless told otherwise."""
    # Imported here so the manual-board path works without OpenCV installed.
    from word_solver.vision.debug import format_reading
    from word_solver.vision.pipeline import recognize_screenshot

    result = recognize_screenshot(image, debug=debug)
    reading = result.reading
    print(f"Board region ({reading.region.method}): {vars(reading.region.rect)}")
    print(format_reading(reading))
    if result.debug_dir:
        print(f"\nDebug output saved to: {result.debug_dir}")

    if not reading.is_reliable:
        cells = ", ".join(f"[{c.row}, {c.col}]" for c in reading.flagged)
        message = f"\n{len(reading.flagged)} uncertain cell(s): {cells}."
        if not accept_uncertain:
            raise SystemExit(
                message + " Not solving. Check the debug images, add labeled samples with\n"
                "  python -m word_solver.vision.templates add --image <shot> --labels \"ROW1 ROW2 ROW3 ROW4\"\n"
                "or pass the board manually with --board, or rerun with --accept-uncertain."
            )
        print(message + " Solving anyway because --accept-uncertain was given.")
    return reading.board


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the longest words on a 4x4 letter board.")
    parser.add_argument(
        "--board",
        nargs="+",
        metavar="ROW",
        help="Board rows as strings, e.g. --board OSYE YYEM OTPE FUND. Defaults to the sample board.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Read the board from a game screenshot instead of --board.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print solver timings and DFS counters. With --image, also save vision debug images.",
    )
    parser.add_argument(
        "--accept-uncertain",
        action="store_true",
        help="With --image: solve even if some letters were flagged as uncertain.",
    )
    parser.add_argument(
        "--allow",
        nargs="+",
        default=[],
        metavar="WORD",
        help="Add words to data/allowed_words.txt before solving.",
    )
    parser.add_argument(
        "--block",
        nargs="+",
        default=[],
        metavar="WORD",
        help="Add words to data/blocked_words.txt before solving.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, config: SolverConfig = DEFAULT_CONFIG) -> None:
    args = parse_args(argv)

    for word in args.allow:
        print(f"Allowed: {append_word(config.allowed_words_path, word)}")
    for word in args.block:
        print(f"Blocked: {append_word(config.blocked_words_path, word)}")

    if args.image and args.board:
        raise SystemExit("Use either --image or --board, not both.")
    if args.image:
        board = board_from_screenshot(args.image, args.debug, args.accept_uncertain)
    else:
        board = [list(row) for row in args.board] if args.board else DEFAULT_BOARD

    load_start = time.perf_counter()
    dictionary = WordDictionary.from_config(config)
    dict_ms = (time.perf_counter() - load_start) * 1000
    trie_start = time.perf_counter()
    trie = dictionary.build_trie(config.min_word_length, config.max_word_length)
    trie_ms = (time.perf_counter() - trie_start) * 1000
    print(f"Loaded {len(trie):,} dictionary words in {dict_ms + trie_ms:.0f} ms.")

    result = Solver(trie, config).solve(board)
    result.stats.dictionary_load_ms = dict_ms
    result.stats.trie_build_ms = trie_ms
    if args.debug:
        result.stats.print()
    print_report(board, result, config)


if __name__ == "__main__":
    main()
