"""Web API: screenshot upload, manual boards, tile corrections, and warnings."""

import unittest
from dataclasses import replace
from pathlib import Path

try:
    from fastapi.testclient import TestClient

    from word_solver.web.app import create_app
    from word_solver.web.service import BoardService
except ImportError:  # pragma: no cover
    TestClient = None

SCREENSHOT = Path(__file__).resolve().parents[1] / "vision" / "samples" / "board_osye.jpg"
EXPECTED = [list("OSYE"), list("YYEM"), list("OTPE"), list("FUND")]


@unittest.skipIf(TestClient is None, "FastAPI not installed")
class TestWebApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = BoardService()

    def setUp(self):
        self.service._state = None
        self.client = TestClient(create_app(self.service))

    def upload(self):
        with SCREENSHOT.open("rb") as handle:
            return self.client.post("/api/screenshot", files={"file": ("board.jpg", handle, "image/jpeg")})

    def test_index_and_static_files_are_served(self):
        self.assertIn("Word Link Solver", self.client.get("/").text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/app.css").status_code, 200)

    def test_state_before_anything_is_loaded(self):
        body = self.client.get("/api/state").json()
        self.assertFalse(body["loaded"])
        self.assertIn("capture", body)
        self.assertFalse(body["capture"]["enabled"])

    def test_screenshot_is_recognized_and_solved(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        state = response.json()
        self.assertEqual(state["board"], EXPECTED)
        self.assertTrue(state["reliable"])
        self.assertEqual(state["flagged"], [])
        self.assertGreater(state["min_confidence"], 0.9)
        self.assertEqual(state["origin"], "screenshot")
        self.assertEqual(state["current_suggestion"]["word"], "DEPUTES")
        self.assertEqual(state["solve"]["words"][0]["word"], "DEPUTES")
        self.assertEqual(state["solve"]["words"][1]["word"], "TOUPEES")
        self.assertEqual(state["solve"]["best_length"], 7)
        self.assertEqual(state["current_suggestion"]["score"], 1010)
        self.assertEqual(state["solve"]["best_score"], 1010)
        self.assertEqual(state["solve"]["best_count"], 1)
        self.assertEqual(state["suggestion_index"], 1)
        self.assertGreater(state["total_suggestions"], 10)
        self.assertEqual(len(state["solve"]["words"]), 11)  # current word + next 10
        self.assertEqual(len(state["next_suggestions"]), 3)
        self.assertIn("updated_at", state)
        self.assertGreaterEqual(state["solve"]["elapsed_ms"], 0)
        self.assertEqual(self.client.get("/api/state").json()["board"], EXPECTED)

    def test_paths_match_solver_format(self):
        state = self.client.post("/api/board", json={"rows": ["OSYE", "YYEM", "OTPE", "FUND"]}).json()
        for word in state["solve"]["words"]:
            self.assertEqual(len(word["path"]), word["length"])
            for r, c in word["path"]:
                self.assertEqual(state["board"][r][c], word["word"][word["path"].index([r, c])])

    def test_manual_board(self):
        state = self.client.post("/api/board", json={"rows": ["osye", "yyem", "otpe", "fund"]}).json()
        self.assertEqual(state["board"], EXPECTED)
        self.assertEqual(state["origin"], "manual")
        self.assertIsNone(state["min_confidence"])
        self.assertTrue(state["reliable"])

    def test_cell_correction_reruns_solver(self):
        self.upload()
        state = self.client.post("/api/cell", json={"row": 3, "col": 3, "letter": "x"}).json()
        self.assertEqual(state["board"][3][3], "X")
        self.assertEqual(state["cells"][3][3]["source"], "manual")
        words = [w["word"] for w in state["solve"]["words"]]
        self.assertNotIn("DEPUTES", words)  # DEPUTES started on the D that was replaced

    def test_flagged_cells_make_state_unreliable_until_confirmed(self):
        from word_solver.vision.config import DEFAULT_VISION_CONFIG

        self.service.vision_config = replace(DEFAULT_VISION_CONFIG, min_confidence=0.99999)
        try:
            state = self.upload().json()
        finally:
            self.service.vision_config = None
        client = self.client
        self.assertFalse(state["reliable"])
        self.assertTrue(state["flagged"])
        for r, c in state["flagged"]:
            state = client.post("/api/cell", json={"row": r, "col": c, "letter": state["cells"][r][c]["letter"]}).json()
        self.assertTrue(state["reliable"])
        self.assertEqual(state["board"], EXPECTED)

    def test_rejects_bad_input(self):
        self.assertEqual(self.client.post("/api/cell", json={"row": 0, "col": 0, "letter": "A"}).status_code, 409)
        self.upload()
        self.assertEqual(self.client.post("/api/cell", json={"row": 0, "col": 0, "letter": "1"}).status_code, 400)
        self.assertEqual(self.client.post("/api/cell", json={"row": 0, "col": 0, "letter": "AB"}).status_code, 400)
        self.assertEqual(self.client.post("/api/cell", json={"row": 9, "col": 0, "letter": "A"}).status_code, 400)
        self.assertEqual(self.client.post("/api/board", json={"rows": ["ABC"]}).status_code, 400)
        bad = self.client.post("/api/screenshot", files={"file": ("x.png", b"not an image", "image/png")})
        self.assertEqual(bad.status_code, 400)


if __name__ == "__main__":
    unittest.main()
