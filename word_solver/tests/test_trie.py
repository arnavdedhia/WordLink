"""Trie behavior and prefix pruning during the search."""

import unittest
from unittest import mock

from word_solver import solver as solver_module
from word_solver.tests.helpers import make_board, solve
from word_solver.trie import Trie


class TestTrie(unittest.TestCase):
    def setUp(self):
        self.trie = Trie(["CAT", "CATS", "DOG"])

    def test_contains_complete_words_only(self):
        self.assertTrue(self.trie.contains("CAT"))
        self.assertTrue(self.trie.contains("CATS"))
        self.assertFalse(self.trie.contains("CA"))
        self.assertFalse(self.trie.contains("COW"))

    def test_has_prefix(self):
        self.assertTrue(self.trie.has_prefix("C"))
        self.assertTrue(self.trie.has_prefix("CA"))
        self.assertTrue(self.trie.has_prefix("CATS"))
        self.assertFalse(self.trie.has_prefix("CATSS"))
        self.assertFalse(self.trie.has_prefix("X"))

    def test_size_ignores_duplicates(self):
        trie = Trie(["CAT", "CAT", "CATS"])
        self.assertEqual(len(trie), 2)


class TestPrefixPruning(unittest.TestCase):
    def _count_lookups(self, board, words) -> int:
        with mock.patch.object(solver_module, "_advance", wraps=solver_module._advance) as spy:
            solve(board, words)
        return spy.call_count

    def test_empty_trie_never_extends_a_path(self):
        board = make_board({})
        # One lookup per starting tile, and every one fails immediately.
        self.assertEqual(self._count_lookups(board, []), 16)

    def test_search_stops_at_dead_prefixes(self):
        # Only C-A-T can grow; every Z neighbor is rejected after one lookup.
        board = make_board({(1, 1): "C", (1, 2): "A", (2, 2): "T"})
        lookups = self._count_lookups(board, ["CAT"])
        # 16 start lookups + C's 8 neighbors + A's 7 unvisited neighbors + T's 6 unvisited neighbors.
        self.assertEqual(lookups, 16 + 8 + 7 + 6)

    def test_unknown_starting_letter_is_never_expanded(self):
        board = make_board({(0, 0): "Q"})
        self.assertEqual(self._count_lookups(board, ["CAT"]), 16)


if __name__ == "__main__":
    unittest.main()
