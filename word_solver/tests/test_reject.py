"""Reject / next / undo use the cached suggestion list. They must not recapture or resolve."""

import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from word_solver.config import DEFAULT_CONFIG
from word_solver.web.service import BoardService, BoardState, CellState

BOARD = ["OSYE", "YYEM", "OTPE", "FUND"]
OTHER = ["CATS", "OOOO", "OOOO", "OOOO"]


def service_with_temp_block():
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "blocked_words.txt"
    path.write_text("# test block list\n", encoding="utf-8")
    config = replace(DEFAULT_CONFIG, blocked_words_path=path)
    return BoardService(solver_config=config), path, tmp


def manual_state(rows: list[str]) -> BoardState:
    cells = [[CellState(ch) for ch in row] for row in rows]
    now = datetime.now()
    return BoardState(cells, "manual", now, recognized_at=now)


class TestRejectAndNext(unittest.TestCase):
    def setUp(self):
        self.service, self.block_path, self._tmp = service_with_temp_block()
        self.addCleanup(self._tmp.cleanup)

    def test_reject_returns_next_suggestion_without_resolving(self):
        self.service.set_board(BOARD)
        first = self.service.state()["current_suggestion"]["word"]
        solve_calls = {"n": 0}
        original = self.service.solver.solve

        def wrapped(board):
            solve_calls["n"] += 1
            return original(board)

        self.service.solver.solve = wrapped
        before = solve_calls["n"]
        started = time.perf_counter()
        out = self.service.reject_current()
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.assertEqual(solve_calls["n"], before)
        self.assertEqual(out["rejected_word"], first)
        self.assertNotEqual(out["current_suggestion"]["word"], first)
        self.assertEqual(out["suggestion_index"], 2)
        self.assertLess(elapsed_ms, 50)
        self.assertLess(out["switch_ms"], 50)

    def test_reject_does_not_call_ingest_or_recognition(self):
        self.service.set_board(BOARD)

        def boom(*_args, **_kwargs):
            self.fail("ingest_bgr must not run when rejecting a word")

        self.service.ingest_bgr = boom
        self.service.reject_current()

    def test_rejected_word_is_appended_without_duplicates(self):
        self.service.set_board(BOARD)
        word = self.service.state()["current_suggestion"]["word"]
        self.service.reject_current()
        text = self.block_path.read_text(encoding="utf-8")
        self.assertEqual(text.count(word), 1)
        self.service.blocked.add(word)
        self.assertEqual(self.block_path.read_text(encoding="utf-8").count(word), 1)

    def test_rejected_words_remain_blocked_after_restart(self):
        self.service.set_board(BOARD)
        word = self.service.state()["current_suggestion"]["word"]
        self.service.reject_current()
        restarted = BoardService(solver_config=self.service.config)
        restarted.set_board(BOARD)
        found = {item.word for item in restarted._state.solve.words}
        self.assertNotIn(word, found)
        self.assertNotEqual(restarted.state()["current_suggestion"]["word"], word)

    def test_next_word_does_not_block(self):
        self.service.set_board(BOARD)
        first = self.service.state()["current_suggestion"]["word"]
        out = self.service.next_word()
        self.assertNotEqual(out["current_suggestion"]["word"], first)
        self.assertEqual(out["suggestion_index"], 2)
        self.assertNotIn(first, self.service.blocked)
        self.assertNotIn(first, self.block_path.read_text(encoding="utf-8"))
        self.service.set_board(OTHER)
        self.service.set_board(BOARD)
        self.assertEqual(self.service.state()["current_suggestion"]["word"], first)

    def test_undo_removes_latest_rejection(self):
        self.service.set_board(BOARD)
        word = self.service.state()["current_suggestion"]["word"]
        self.service.reject_current()
        out = self.service.undo_rejection()
        self.assertEqual(out["undone_word"], word)
        self.assertNotIn(word, self.service.blocked)
        self.assertNotIn(word, self.block_path.read_text(encoding="utf-8"))
        restarted = BoardService(solver_config=self.service.config)
        restarted.set_board(BOARD)
        found = {item.word for item in restarted._state.solve.words}
        self.assertIn(word, found)

    def test_new_board_resets_suggestion_index(self):
        self.service.set_board(BOARD)
        self.service.next_word()
        self.service.next_word()
        self.assertEqual(self.service.state()["suggestion_index"], 3)
        self.service.set_board(OTHER)
        self.assertEqual(self.service.state()["suggestion_index"], 1)

    def test_identical_reread_does_not_reset_index(self):
        self.service.set_board(BOARD)
        self.service.next_word()
        idx = self.service.state()["suggestion_index"]
        board_id = self.service.state()["board_id"]
        self.service.set_board(BOARD)
        self.assertEqual(self.service.state()["suggestion_index"], idx)
        self.assertEqual(self.service.state()["board_id"], board_id)

    def test_stale_async_result_cannot_overwrite_newer_board(self):
        g1 = self.service._bump_generation()
        g2 = self.service._bump_generation()
        older = manual_state(BOARD)
        newer = manual_state(OTHER)
        self.service._solve(older)
        self.service._solve(newer)
        self.service._apply_recognized(g2, newer)
        self.assertEqual(self.service._state.board[0], list("CATS"))
        newer_id = self.service._state.board_id
        self.service._apply_recognized(g1, older)
        self.assertEqual(self.service._state.board[0], list("CATS"))
        self.assertEqual(self.service._state.board_id, newer_id)


try:
    from fastapi.testclient import TestClient

    from word_solver.web.app import create_app
except ImportError:  # pragma: no cover
    TestClient = None


@unittest.skipIf(TestClient is None, "FastAPI not installed")
class TestRejectApi(unittest.TestCase):
    def setUp(self):
        self.service, self.block_path, self._tmp = service_with_temp_block()
        self.addCleanup(self._tmp.cleanup)
        self.client = TestClient(create_app(self.service))

    def test_reject_and_next_endpoints(self):
        first = self.client.post("/api/board", json={"rows": BOARD}).json()["current_suggestion"]["word"]
        rejected = self.client.post("/api/reject-current-word").json()
        self.assertEqual(rejected["rejected_word"], first)
        self.assertEqual(self.client.get("/api/current-board").json()["current_suggestion"]["word"], rejected["current_suggestion"]["word"])
        skipped = self.client.post("/api/next-word").json()
        self.assertNotEqual(skipped["current_suggestion"]["word"], rejected["current_suggestion"]["word"])
        blocked_lines = {line.strip() for line in self.block_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
        self.assertNotIn(skipped["current_suggestion"]["word"], blocked_lines)

    def test_keyboard_shortcuts_are_wired(self):
        html = self.client.get("/").text
        js = self.client.get("/static/app.js").text
        self.assertIn("WORD NOT ACCEPTED", html)
        self.assertIn(">X</span>", html)
        self.assertIn(">N</span>", html)
        self.assertIn(">Z</span>", html)
        self.assertIn('key === "x"', js)
        self.assertIn('key === "n"', js)
        self.assertIn('key === "z"', js)
        self.assertIn("typingInField", js)


if __name__ == "__main__":
    unittest.main()
