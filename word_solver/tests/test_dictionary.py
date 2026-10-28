"""Dictionary normalization, allow/block lists, and validation during solving."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from word_solver.config import DEFAULT_CONFIG
from word_solver.dictionary import WordDictionary, append_word, normalize_word, read_word_file
from word_solver.tests.helpers import found_words, make_board, solve


class TestNormalizeWord(unittest.TestCase):
    def test_uppercases_and_trims(self):
        self.assertEqual(normalize_word("  found\n"), "FOUND")

    def test_rejects_punctuation_spaces_and_digits(self):
        for raw in ["don't", "ice cream", "abc1", "co-op", "", "   ", "café"]:
            with self.subTest(raw=raw):
                self.assertIsNone(normalize_word(raw))


class TestWordDictionary(unittest.TestCase):
    def test_filters_and_normalizes(self):
        d = WordDictionary(["cat", "Dog", "it's", "2nd", "two words"])
        self.assertEqual(d.words, frozenset({"CAT", "DOG"}))

    def test_allowed_words_are_added(self):
        d = WordDictionary(["cat"], allowed_words=["zyx"])
        self.assertIn("ZYX", d)

    def test_blocked_words_are_removed(self):
        d = WordDictionary(["cat", "dog"], blocked_words=["DOG"])
        self.assertNotIn("DOG", d)
        self.assertIn("CAT", d)

    def test_block_wins_over_allow(self):
        d = WordDictionary(["cat"], allowed_words=["dog"], blocked_words=["dog"])
        self.assertNotIn("DOG", d)

    def test_build_trie_respects_length_bounds(self):
        trie = WordDictionary(["at", "cat", "cats"]).build_trie(min_length=3, max_length=3)
        self.assertEqual(len(trie), 1)
        self.assertTrue(trie.contains("CAT"))

    def test_from_config_reads_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "words.txt").write_text("cat\ndog\nbird\n", encoding="utf-8")
            (tmp / "allow.txt").write_text("# comment\nzyx\n\n", encoding="utf-8")
            (tmp / "block.txt").write_text("dog\n", encoding="utf-8")
            config = replace(
                DEFAULT_CONFIG,
                words_path=tmp / "words.txt",
                allowed_words_path=tmp / "allow.txt",
                blocked_words_path=tmp / "block.txt",
            )
            d = WordDictionary.from_config(config)
            self.assertEqual(d.words, frozenset({"CAT", "BIRD", "ZYX"}))

    def test_missing_allow_and_block_files_are_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.assertEqual(read_word_file(tmp / "missing.txt"), set())

    def test_append_word(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blocked.txt"
            path.write_text("# header", encoding="utf-8")
            self.assertEqual(append_word(path, " yoyo "), "YOYO")
            append_word(path, "YOYO")  # duplicate is ignored
            self.assertEqual(read_word_file(path), {"YOYO"})
            self.assertEqual(path.read_text(encoding="utf-8").count("YOYO"), 1)
            with self.assertRaises(ValueError):
                append_word(path, "bad word")


class TestSolverDictionaryValidation(unittest.TestCase):
    def test_only_dictionary_words_are_returned(self):
        board = make_board({(0, 0): "C", (0, 1): "A", (0, 2): "T", (0, 3): "S"})
        found = found_words(solve(board, ["CAT", "CATS"]))
        self.assertEqual(set(found), {"CAT", "CATS"})
        self.assertNotIn("ACT", found)  # traceable on the board but not in the dictionary

    def test_blocked_word_is_not_found(self):
        board = make_board({(0, 0): "C", (0, 1): "A", (0, 2): "T", (0, 3): "S"})
        trie = WordDictionary(["cat", "cats"], blocked_words=["cats"]).build_trie(3)
        from word_solver.solver import Solver

        self.assertEqual({r.word for r in Solver(trie).solve(board).words}, {"CAT"})


if __name__ == "__main__":
    unittest.main()
