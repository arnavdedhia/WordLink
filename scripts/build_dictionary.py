"""Build a frequency-filtered Triumph dictionary from ENABLE1.

Run from the repository root::

    python scripts/build_dictionary.py
    python scripts/build_dictionary.py --size 100000

This script is the only place ENABLE1 and frequency data are loaded.
The running app reads ``word_solver/data/triumph_words.txt`` only.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from word_solver.config import (  # noqa: E402
    DATA_DIR,
    DEFAULT_BOARD,
    DEFAULT_CONFIG,
    DEFAULT_DICTIONARY_SIZE,
    ENABLE_WORDS_PATH,
    MAX_BOARD_LETTERS,
    TRIUMPH_WORDS_PATH,
)
from word_solver.dictionary import WordDictionary, normalize_word  # noqa: E402
from word_solver.solver import Solver  # noqa: E402

FREQUENCY_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_full.txt"
)
FREQUENCY_CACHE = DATA_DIR / ".cache" / "opensubtitles_en_full.txt"
VOWELS = set("AEIOU")

INFLECTION_EXAMPLES = (
    "PLAY",
    "PLAYS",
    "PLAYED",
    "PLAYING",
    "HOUSE",
    "HOUSES",
    "QUICK",
    "QUICKER",
    "QUICKEST",
)

BENCH_LETTERS = "EEEEAAAARRRIIIOOOTTTNNSSLLUUUDDGGCCMMBBPPFFHHVVWWYYKJXQZ"
TARGET_SIZES = (50_000, 75_000, 85_000, 100_000, 125_000)


@dataclass(frozen=True)
class FilterStats:
    original_enable: int
    invalid_characters: int
    too_short: int
    too_long: int
    eligible: int
    with_frequency: int
    without_frequency: int
    core_size: int
    inflection_extras: int
    final_size: int


def related_forms(word: str) -> set[str]:
    """Regular English inflections of ``word``. Not a full stemmer."""
    forms = {word}

    def add(candidate: str) -> None:
        if candidate.isalpha() and len(candidate) >= 2:
            forms.add(candidate)

    add(word + "S")
    add(word + "ES")
    add(word + "ED")
    add(word + "D")
    add(word + "ING")
    add(word + "ER")
    add(word + "EST")
    if word.endswith("E"):
        add(word + "D")
        add(word + "R")
        add(word + "ST")
        if not word.endswith("EE"):
            add(word[:-1] + "ING")
    if word.endswith("Y") and len(word) > 1 and word[-2] not in VOWELS:
        add(word[:-1] + "IES")
        add(word[:-1] + "IED")
        add(word[:-1] + "IER")
        add(word[:-1] + "IEST")
    if word.endswith("S") and not word.endswith("SS") and len(word) > 1:
        add(word[:-1])
    if word.endswith("ES") and len(word) > 2:
        add(word[:-2])
    if word.endswith("IES") and len(word) > 3:
        add(word[:-3] + "Y")
    if word.endswith("ED") and len(word) > 3:
        add(word[:-2])
        add(word[:-1])
    if word.endswith("ING") and len(word) > 4:
        add(word[:-3])
        add(word[:-3] + "E")
    if word.endswith("ER") and len(word) > 3:
        add(word[:-2])
        add(word[:-1])
    if word.endswith("EST") and len(word) > 4:
        add(word[:-3])
        add(word[:-2])
    return forms


def download_frequency_list(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached frequency list: {dest}")
        return dest
    print(f"Downloading frequency list:\n  {url}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "WordLink-dictionary-builder"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(dest)
    print(f"Saved {dest.stat().st_size:,} bytes to {dest}")
    return dest


def load_enable(path: Path) -> tuple[set[str], int, int]:
    """Return (normalized words, raw non-comment lines, invalid-character count)."""
    words: set[str] = set()
    raw = 0
    invalid = 0
    if not path.exists():
        raise FileNotFoundError(f"ENABLE1 word list not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            raw += 1
            word = normalize_word(line)
            if word is None:
                invalid += 1
                continue
            words.add(word)
    return words, raw, invalid


def load_frequency_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                count = int(parts[-1])
            except ValueError:
                continue
            word = normalize_word(parts[0])
            if word is None:
                continue
            previous = counts.get(word)
            if previous is None or count > previous:
                counts[word] = count
    return counts


def filter_eligible(enable: set[str], min_length: int, max_length: int) -> tuple[set[str], int, int]:
    too_short = 0
    too_long = 0
    eligible: set[str] = set()
    for word in enable:
        if len(word) < min_length:
            too_short += 1
        elif len(word) > max_length:
            too_long += 1
        else:
            eligible.add(word)
    return eligible, too_short, too_long


def rank_and_select(
    eligible: set[str],
    frequencies: dict[str, int],
    target: int,
    keep_inflections: bool,
) -> tuple[list[str], FilterStats, int, int]:
    with_frequency = 0
    without_frequency = 0
    ranked: list[tuple[int, str]] = []
    for word in eligible:
        freq = frequencies.get(word, 0)
        if freq:
            with_frequency += 1
        else:
            without_frequency += 1
        ranked.append((freq, word))
    # Highest frequency first; alphabetical tie-break for a stable file.
    ranked.sort(key=lambda item: (-item[0], item[1]))

    core = [word for _freq, word in ranked[:target]]
    selected = set(core)
    inflection_extras = 0
    if keep_inflections:
        for word in core:
            family = related_forms(word) & eligible
            for form in family:
                if form not in selected:
                    selected.add(form)
                    inflection_extras += 1

    chosen = sorted(selected)
    stats = FilterStats(
        original_enable=0,
        invalid_characters=0,
        too_short=0,
        too_long=0,
        eligible=len(eligible),
        with_frequency=with_frequency,
        without_frequency=without_frequency,
        core_size=len(core),
        inflection_extras=inflection_extras,
        final_size=len(chosen),
    )
    return chosen, stats, with_frequency, without_frequency


def write_dictionary(path: Path, words: list[str], size: int, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Frequency-filtered ENABLE1 subset for Triumph Word Link.\n"
        f"# Generated by scripts/build_dictionary.py --size {size}.\n"
        f"# Ranking source: {source}.\n"
        f"# Do not edit this file by hand. Add exceptions to allowed_words.txt\n"
        f"# or blocked_words.txt instead.\n"
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(header)
        for word in words:
            handle.write(word + "\n")


def random_board(rng: random.Random) -> list[list[str]]:
    tiles = [rng.choice(BENCH_LETTERS) for _ in range(16)]
    return [tiles[i : i + 4] for i in range(0, 16, 4)]


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def bench_trie_and_solver(label: str, words_path: Path, boards: list[tuple[str, list[list[str]]]]) -> None:
    config = replace(DEFAULT_CONFIG, words_path=words_path)
    load_start = time.perf_counter()
    dictionary = WordDictionary.from_config(config)
    load_ms = (time.perf_counter() - load_start) * 1000
    trie_start = time.perf_counter()
    trie = dictionary.build_trie(config.min_word_length, config.max_word_length)
    trie_ms = (time.perf_counter() - trie_start) * 1000
    print(f"{label}")
    print(f"  Words in trie:          {len(trie):,}")
    print(f"  Dictionary loading:     {load_ms:.1f} ms")
    print(f"  Trie construction:      {trie_ms:.1f} ms")
    solver = Solver(trie, config)
    for name, board in boards:
        solver.solve(board)
        searches: list[float] = []
        last = None
        repeats = 25 if name.startswith("Sample") else 15
        for _ in range(repeats):
            last = solver.solve(board)
            searches.append(last.stats.search_ms)
        assert last is not None
        top = last.words[0].word if last.words else "-"
        print(
            f"  {name}: search median {median(searches):.2f} ms "
            f"(min {min(searches):.2f}, max {max(searches):.2f}); "
            f"{last.stats.words_found:,} words; top {top}"
        )
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build triumph_words.txt from ENABLE1 + word frequency.")
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_DICTIONARY_SIZE,
        help=f"Target ranked size before inflection extras (default {DEFAULT_DICTIONARY_SIZE}). "
        f"Useful values: {', '.join(str(n) for n in TARGET_SIZES)}.",
    )
    parser.add_argument("--enable", type=Path, default=ENABLE_WORDS_PATH)
    parser.add_argument("--output", type=Path, default=TRIUMPH_WORDS_PATH)
    parser.add_argument("--min-length", type=int, default=DEFAULT_CONFIG.min_word_length)
    parser.add_argument("--max-length", type=int, default=MAX_BOARD_LETTERS)
    parser.add_argument("--frequency-url", default=FREQUENCY_URL)
    parser.add_argument("--frequency-file", type=Path, default=FREQUENCY_CACHE)
    parser.add_argument(
        "--keep-inflections",
        action="store_true",
        help="Also keep regular ENABLE1 inflections of the ranked words. "
        "Off by default because word-form frequency already retains PLAY/PLAYS/PLAYED/PLAYING "
        "and this expander adds tens of thousands of extra words.",
    )
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.size <= 0:
        raise SystemExit("--size must be a positive integer.")

    enable, raw_count, invalid = load_enable(args.enable)
    eligible, too_short, too_long = filter_eligible(enable, args.min_length, args.max_length)

    download_frequency_list(args.frequency_url, args.frequency_file)
    frequencies = load_frequency_counts(args.frequency_file)

    chosen, stats, with_frequency, without_frequency = rank_and_select(
        eligible,
        frequencies,
        target=args.size,
        keep_inflections=args.keep_inflections,
    )
    stats = FilterStats(
        original_enable=raw_count,
        invalid_characters=invalid,
        too_short=too_short,
        too_long=too_long,
        eligible=len(eligible),
        with_frequency=with_frequency,
        without_frequency=without_frequency,
        core_size=stats.core_size,
        inflection_extras=stats.inflection_extras,
        final_size=len(chosen),
    )

    source = "OpenSubtitles 2018 FrequencyWords (hermitdave/FrequencyWords en_full.txt)"
    write_dictionary(args.output, chosen, args.size, source)

    chosen_set = set(chosen)
    print("Dictionary generation")
    print(f"  Original ENABLE1 words:              {stats.original_enable:,}")
    print(f"  Words removed (invalid characters):  {stats.invalid_characters:,}")
    print(f"  Words removed (< {args.min_length} letters):           {stats.too_short:,}")
    print(f"  Words removed (> {args.max_length} letters):          {stats.too_long:,}")
    print(f"  Eligible after filters:              {stats.eligible:,}")
    print(f"  Words matched with frequency data:   {stats.with_frequency:,}")
    print(f"  Words without frequency information: {stats.without_frequency:,}")
    print(f"  Ranked core (top {args.size:,}):            {stats.core_size:,}")
    print(f"  Extra inflected forms added:         {stats.inflection_extras:,}")
    print(f"  Final dictionary size:               {stats.final_size:,}")
    print(f"  Wrote {args.output}")
    print()
    print("Inflection check (must be present if they exist in ENABLE1):")
    for word in INFLECTION_EXAMPLES:
        in_enable = word in eligible
        kept = word in chosen_set
        status = "kept" if kept else ("missing from ENABLE1" if not in_enable else "DROPPED")
        print(f"  {word:<10} {status}")
    print()

    if args.skip_benchmark:
        return

    rng = random.Random(42)
    boards = [("Sample board OSYE/YYEM/OTPE/FUND", DEFAULT_BOARD)]
    for i in range(5):
        boards.append((f"Random board {i + 1}", random_board(rng)))

    print("Benchmarks use the same boards for both dictionaries.")
    print("Allowed/blocked lists are applied in both runs.\n")
    bench_trie_and_solver("Old dictionary (ENABLE1 / words.txt)", args.enable, boards)
    bench_trie_and_solver("New dictionary (triumph_words.txt)", args.output, boards)


if __name__ == "__main__":
    main()
