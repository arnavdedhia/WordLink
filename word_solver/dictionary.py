"""Loading and normalizing the word list.

The final word set is::

    (triumph_words.txt + allowed_words.txt) - blocked_words.txt

``triumph_words.txt`` is a frequency-filtered ENABLE1 subset built offline by
``scripts/build_dictionary.py``. The app never loads ENABLE1 or frequency data.

Words confirmed in-game go in ``allowed_words.txt`` and are always accepted.
Words the game rejects go in ``blocked_words.txt`` and are never recommended.

To use a different base dictionary, point ``SolverConfig.words_path`` at
another file with one word per line, or build a ``WordDictionary`` directly
from any iterable of strings.

``allowed_words.txt`` and ``blocked_words.txt`` hold one word per line.
Blank lines and lines starting with ``#`` are ignored.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable

from word_solver.config import DEFAULT_CONFIG, SolverConfig
from word_solver.trie import Trie

_VALID_WORD = re.compile(r"^[A-Z]+$")


def normalize_word(raw: str) -> str | None:
    """Uppercase and trim ``raw``; return None if it contains anything but letters A-Z."""
    word = raw.strip().upper()
    if not word or not _VALID_WORD.match(word):
        return None
    return word


def read_word_file(path: Path) -> set[str]:
    """Read a one-word-per-line file into a set of normalized words.

    Comment lines (``#``), blank lines, and invalid entries are skipped.
    A missing file yields an empty set.
    """
    if not path.exists():
        return set()
    words: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.lstrip().startswith("#"):
                continue
            word = normalize_word(line)
            if word:
                words.add(word)
    return words


def append_word(path: Path, raw: str, known: set[str] | None = None) -> str:
    """Append a normalized word to ``path`` (used for the allow/block lists).

    Returns the normalized word. Raises ValueError for invalid input.
    Pass ``known`` to skip rereading the file (O(1) duplicate check).
    """
    word = normalize_word(raw)
    if word is None:
        raise ValueError(f"Invalid word {raw!r}: only letters A-Z are allowed.")
    if known is not None:
        if word in known:
            return word
    elif word in read_word_file(path):
        return word
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(word + "\n")
    if known is not None:
        known.add(word)
    return word


def _append_line(path: Path, word: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(word + "\n")


class BlockedWordList:
    """In-memory blocked-word set with append-only persistence.

    ``contains`` and duplicate checks are O(1). The file is not reread on add.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.words: set[str] = set()
        self._order: list[str] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.lstrip().startswith("#"):
                        continue
                    word = normalize_word(line)
                    if word and word not in self.words:
                        self.words.add(word)
                        self._order.append(word)

    def __contains__(self, word: str) -> bool:
        normalized = normalize_word(word)
        return normalized is not None and normalized in self.words

    def add(self, raw: str) -> tuple[str, bool, float]:
        """Block ``raw``. Returns (word, newly_added, persist_ms). Does not reread the file."""
        word = normalize_word(raw)
        if word is None:
            raise ValueError(f"Invalid word {raw!r}: only letters A-Z are allowed.")
        if word in self.words:
            return word, False, 0.0
        self.words.add(word)
        self._order.append(word)
        start = time.perf_counter()
        _append_line(self.path, word)
        persist_ms = (time.perf_counter() - start) * 1000
        return word, True, persist_ms

    def remove(self, raw: str) -> str | None:
        """Unblock ``raw`` and rewrite the file from the in-memory list. None if it was not blocked."""
        word = normalize_word(raw)
        if word is None or word not in self.words:
            return None
        self.words.discard(word)
        self._order = [w for w in self._order if w != word]
        self._rewrite()
        return word

    def _rewrite(self) -> None:
        comments: list[str] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    comments.append(line)
                else:
                    break
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for line in comments:
                handle.write(line + "\n")
            for word in self._order:
                handle.write(word + "\n")


class WordDictionary:
    """The set of words the solver accepts."""

    def __init__(
        self,
        base_words: Iterable[str],
        allowed_words: Iterable[str] = (),
        blocked_words: Iterable[str] = (),
    ) -> None:
        normalized = {w for w in map(normalize_word, base_words) if w}
        normalized |= {w for w in map(normalize_word, allowed_words) if w}
        normalized -= {w for w in map(normalize_word, blocked_words) if w}
        self.words: frozenset[str] = frozenset(normalized)

    @classmethod
    def from_config(cls, config: SolverConfig = DEFAULT_CONFIG) -> WordDictionary:
        """Load the base, allowed, and blocked lists from the paths in ``config``."""
        if not config.words_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {config.words_path}")
        return cls(
            base_words=read_word_file(config.words_path),
            allowed_words=read_word_file(config.allowed_words_path),
            blocked_words=read_word_file(config.blocked_words_path),
        )

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        normalized = normalize_word(word)
        return normalized is not None and normalized in self.words

    def build_trie(self, min_length: int = 1, max_length: int | None = None) -> Trie:
        """Build a Trie of the words whose length is within the given bounds."""
        return Trie(
            w
            for w in self.words
            if len(w) >= min_length and (max_length is None or len(w) <= max_length)
        )
