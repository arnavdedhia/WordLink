"""Horizontal, vertical, and diagonal movement; tile reuse; minimum length."""

import unittest
from dataclasses import replace

from word_solver.config import DEFAULT_CONFIG, ORTHOGONAL_OFFSETS
from word_solver.tests.helpers import found_words, make_board, solve


class TestMovement(unittest.TestCase):
    def test_horizontal_left_to_right(self):
        board = make_board({(1, 0): "C", (1, 1): "A", (1, 2): "T"})
        self.assertEqual(found_words(solve(board, ["CAT"]))["CAT"], [[1, 0], [1, 1], [1, 2]])

    def test_horizontal_right_to_left(self):
        board = make_board({(2, 3): "C", (2, 2): "A", (2, 1): "T"})
        self.assertEqual(found_words(solve(board, ["CAT"]))["CAT"], [[2, 3], [2, 2], [2, 1]])

    def test_vertical_top_to_bottom(self):
        board = make_board({(0, 2): "C", (1, 2): "A", (2, 2): "T"})
        self.assertEqual(found_words(solve(board, ["CAT"]))["CAT"], [[0, 2], [1, 2], [2, 2]])

    def test_vertical_bottom_to_top(self):
        board = make_board({(3, 0): "C", (2, 0): "A", (1, 0): "T"})
        self.assertEqual(found_words(solve(board, ["CAT"]))["CAT"], [[3, 0], [2, 0], [1, 0]])

    def test_diagonal_main(self):
        board = make_board({(0, 0): "C", (1, 1): "A", (2, 2): "T", (3, 3): "S"})
        self.assertEqual(
            found_words(solve(board, ["CATS"]))["CATS"],
            [[0, 0], [1, 1], [2, 2], [3, 3]],
        )

    def test_diagonal_anti(self):
        board = make_board({(0, 3): "C", (1, 2): "A", (2, 1): "T"})
        self.assertEqual(found_words(solve(board, ["CAT"]))["CAT"], [[0, 3], [1, 2], [2, 1]])

    def test_diagonal_can_be_disabled(self):
        board = make_board({(0, 0): "C", (1, 1): "A", (2, 2): "T"})
        config = replace(DEFAULT_CONFIG, neighbor_offsets=ORTHOGONAL_OFFSETS)
        self.assertNotIn("CAT", found_words(solve(board, ["CAT"], config)))

    def test_non_adjacent_tiles_do_not_connect(self):
        board = make_board({(0, 0): "C", (0, 2): "A", (0, 3): "T"})
        self.assertNotIn("CAT", found_words(solve(board, ["CAT"])))

    def test_mixed_directions(self):
        # Right, then down, then diagonally down-left.
        board = make_board({(0, 0): "S", (0, 1): "T", (1, 1): "O", (2, 0): "P"})
        self.assertEqual(
            found_words(solve(board, ["STOP"]))["STOP"],
            [[0, 0], [0, 1], [1, 1], [2, 0]],
        )


class TestTileReuse(unittest.TestCase):
    def test_tile_cannot_be_reused(self):
        # Only one A exists, so ANA would need to revisit [0, 0].
        board = make_board({(0, 0): "A", (0, 1): "N"})
        self.assertNotIn("ANA", found_words(solve(board, ["ANA"])))

    def test_word_found_when_letter_appears_twice(self):
        board = make_board({(0, 0): "A", (0, 1): "N", (1, 1): "A"})
        self.assertEqual(found_words(solve(board, ["ANA"]))["ANA"], [[0, 0], [0, 1], [1, 1]])

    def test_reuse_rule_is_configurable(self):
        board = make_board({(0, 0): "A", (0, 1): "N"})
        config = replace(DEFAULT_CONFIG, allow_tile_reuse=True)
        self.assertEqual(found_words(solve(board, ["ANA"], config))["ANA"], [[0, 0], [0, 1], [0, 0]])

    def test_every_path_uses_unique_tiles(self):
        board = [list("OSYE"), list("YYEM"), list("OTPE"), list("FUND")]
        words = ["YOYO", "EYE", "SEEM", "TOYS", "FOUND", "MEME", "DEEP"]
        for item in solve(board, words).words:
            cells = [tuple(p) for p in item.path]
            self.assertEqual(len(cells), len(set(cells)), item.word)


class TestMinimumLength(unittest.TestCase):
    def setUp(self):
        self.board = make_board({(0, 0): "C", (0, 1): "A", (0, 2): "T"})
        self.words = ["A", "AT", "CAT"]

    def test_short_words_excluded_by_default(self):
        found = found_words(solve(self.board, self.words))
        self.assertEqual(set(found), {"CAT"})

    def test_minimum_length_is_configurable(self):
        config = replace(DEFAULT_CONFIG, min_word_length=2)
        found = found_words(solve(self.board, self.words, config))
        self.assertEqual(set(found), {"AT", "CAT"})

    def test_higher_minimum_excludes_more(self):
        config = replace(DEFAULT_CONFIG, min_word_length=4)
        self.assertEqual(found_words(solve(self.board, self.words, config)), {})


if __name__ == "__main__":
    unittest.main()
