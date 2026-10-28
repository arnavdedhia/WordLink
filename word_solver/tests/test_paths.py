"""Coordinate paths, result ranking, and an end-to-end run on the sample board."""

import unittest

from word_solver.config import ALL_OFFSETS, DEFAULT_BOARD, DEFAULT_CONFIG
from word_solver.dictionary import WordDictionary
from word_solver.solver import Solver, normalize_board
from word_solver.tests.helpers import found_words, solve


def assert_valid_path(test: unittest.TestCase, board, word: str, path) -> None:
    """The path must spell the word, stay in bounds, step to neighbors, and not repeat tiles."""
    test.assertEqual("".join(board[r][c] for r, c in path), word)
    test.assertEqual(len(path), len({tuple(p) for p in path}), f"{word} reuses a tile")
    for r, c in path:
        test.assertTrue(0 <= r < 4 and 0 <= c < 4)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        test.assertIn((r2 - r1, c2 - c1), ALL_OFFSETS, f"{word} has a non-adjacent step")


class TestCoordinatePaths(unittest.TestCase):
    def test_found_example_path(self):
        result = solve(DEFAULT_BOARD, ["FOUND"])
        self.assertEqual(
            result.words[0].to_dict(),
            {
                "word": "FOUND",
                "length": 5,
                "path": [[3, 0], [2, 0], [3, 1], [3, 2], [3, 3]],
            },
        )

    def test_paths_are_zero_based_row_col(self):
        board = [list("ZZZZ"), list("ZZZZ"), list("ZZZZ"), list("ZCAT")]
        self.assertEqual(found_words(solve(board, ["CAT"]))["CAT"], [[3, 1], [3, 2], [3, 3]])

    def test_lowercase_board_is_accepted(self):
        board = [[t.lower() for t in row] for row in DEFAULT_BOARD]
        self.assertIn("FOUND", found_words(solve(board, ["FOUND"])))

    def test_invalid_board_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_board([["A", "B"], ["C", "D"]], DEFAULT_CONFIG)
        with self.assertRaises(ValueError):
            normalize_board([list("ABCD"), list("ABCD"), list("ABCD"), list("AB1D")], DEFAULT_CONFIG)


class TestRanking(unittest.TestCase):
    def test_longest_first_and_ties_all_returned(self):
        board = [list("CATS"), list("DOGZ"), list("ZZZZ"), list("ZZZZ")]
        result = solve(board, ["CAT", "CATS", "DOGS", "DOG"])
        self.assertEqual([r.word for r in result.words], ["CATS", "DOGS", "CAT", "DOG"])
        self.assertEqual([r.word for r in result.best_words], ["CATS", "DOGS"])


class TestSampleBoardWithRealDictionary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trie = WordDictionary.from_config(DEFAULT_CONFIG).build_trie(DEFAULT_CONFIG.min_word_length)
        cls.board = normalize_board(DEFAULT_BOARD)
        cls.result = Solver(trie).solve(DEFAULT_BOARD)

    def test_finds_words(self):
        self.assertGreater(self.result.total, 0)
        self.assertIn("FOUND", found_words(self.result))

    def test_every_result_path_is_valid(self):
        for item in self.result.words:
            with self.subTest(word=item.word):
                self.assertEqual(item.length, len(item.word))
                self.assertGreaterEqual(item.length, 3)
                assert_valid_path(self, self.board, item.word, item.path)

    def test_solves_quickly(self):
        self.assertLess(self.result.elapsed_seconds, 1.0)


if __name__ == "__main__":
    unittest.main()
