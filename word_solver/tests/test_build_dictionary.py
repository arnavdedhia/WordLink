"""Offline dictionary builder: filters, ranking, and generated triumph_words.txt."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_dictionary as builder  # noqa: E402

from word_solver.config import ENABLE_WORDS_PATH, MAX_BOARD_LETTERS, TRIUMPH_WORDS_PATH  # noqa: E402
from word_solver.dictionary import normalize_word, read_word_file  # noqa: E402


class TestRelatedForms(unittest.TestCase):
    def test_play_family(self):
        forms = builder.related_forms("PLAY")
        for word in ("PLAY", "PLAYS", "PLAYED", "PLAYING"):
            self.assertIn(word, forms)

    def test_house_plural(self):
        self.assertIn("HOUSES", builder.related_forms("HOUSE"))
        self.assertIn("HOUSE", builder.related_forms("HOUSES"))

    def test_quick_comparatives(self):
        forms = builder.related_forms("QUICK")
        self.assertIn("QUICKER", forms)
        self.assertIn("QUICKEST", forms)


class TestRankAndSelect(unittest.TestCase):
    def test_keeps_highest_frequency_words(self):
        eligible = {"APPLE", "ZZZZZ", "PLAY", "PLAYS", "BANANA"}
        frequencies = {"APPLE": 100, "PLAY": 90, "PLAYS": 80, "BANANA": 10, "ZZZZZ": 0}
        chosen, stats, matched, missing = builder.rank_and_select(
            eligible, frequencies, target=3, keep_inflections=False
        )
        self.assertEqual(chosen, ["APPLE", "PLAY", "PLAYS"])
        self.assertEqual(matched, 4)
        self.assertEqual(missing, 1)
        self.assertEqual(stats.final_size, 3)

    def test_adds_inflections_of_kept_words(self):
        eligible = {"PLAY", "PLAYS", "PLAYED", "PLAYING", "OBSCURE"}
        frequencies = {"PLAY": 100, "OBSCURE": 1}
        chosen, stats, _matched, _missing = builder.rank_and_select(
            eligible, frequencies, target=1, keep_inflections=True
        )
        self.assertIn("PLAY", chosen)
        self.assertIn("PLAYS", chosen)
        self.assertIn("PLAYED", chosen)
        self.assertIn("PLAYING", chosen)
        self.assertNotIn("OBSCURE", chosen)
        self.assertEqual(stats.inflection_extras, 3)


@unittest.skipUnless(TRIUMPH_WORDS_PATH.exists(), "triumph_words.txt has not been generated yet")
class TestGeneratedTriumphWords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.words = read_word_file(TRIUMPH_WORDS_PATH)
        cls.enable = read_word_file(ENABLE_WORDS_PATH)

    def test_size_is_in_the_requested_band(self):
        self.assertGreaterEqual(len(self.words), 50_000)
        self.assertLessEqual(len(self.words), 125_000)

    def test_every_word_is_a_normalized_enable_word(self):
        for word in self.words:
            self.assertEqual(normalize_word(word), word)
            self.assertTrue(word.isalpha())
            self.assertEqual(word, word.upper())
            self.assertGreaterEqual(len(word), 3)
            self.assertLessEqual(len(word), MAX_BOARD_LETTERS)
            self.assertIn(word, self.enable)

    def test_common_inflections_are_kept(self):
        for word in builder.INFLECTION_EXAMPLES:
            if word in self.enable:
                self.assertIn(word, self.words, f"{word} exists in ENABLE1 but was dropped")


if __name__ == "__main__":
    unittest.main()
