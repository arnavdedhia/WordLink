"""Screenshot recognition: board detection, cells, letter isolation, confidence, debug output."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None

if cv2 is not None:
    from word_solver.vision.board_detector import Rect, configured_rect, detect_board, split_cells
    from word_solver.vision.board_reader import read_board
    from word_solver.vision.config import DEFAULT_VISION_CONFIG
    from word_solver.vision.debug import save_debug
    from word_solver.vision.image_processor import isolate_character, load_image, normalize_glyph
    from word_solver.vision.letter_recognizer import TemplateMatchRecognizer
    from word_solver.vision.templates import (
        SAMPLE,
        SYNTHETIC,
        Template,
        TemplateLibrary,
        parse_labels,
        save_sample,
    )

SCREENSHOT = Path(__file__).resolve().parents[1] / "vision" / "samples" / "board_osye.jpg"
EXPECTED = [
    ["O", "S", "Y", "E"],
    ["Y", "Y", "E", "M"],
    ["O", "T", "P", "E"],
    ["F", "U", "N", "D"],
]


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class TestReferenceScreenshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = load_image(SCREENSHOT)
        cls.library = TemplateLibrary.load()

    def test_recognizes_expected_board(self):
        reading = read_board(self.image)
        self.assertEqual(reading.board, EXPECTED)
        self.assertTrue(reading.is_reliable, [c.recognition.reason for c in reading.flagged])

    def test_recognizes_board_with_synthetic_templates_only(self):
        # Guards against the test above passing only because the samples came from this screenshot.
        synthetic = TemplateLibrary([t for ts in self.library.by_letter.values() for t in ts if t.source == SYNTHETIC])
        reading = read_board(self.image, recognizer=TemplateMatchRecognizer(synthetic))
        self.assertEqual(reading.board, EXPECTED)

    def test_other_resolutions(self):
        for scale in (0.75, 1.5, 2.5):
            with self.subTest(scale=scale):
                resized = cv2.resize(self.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                self.assertEqual(read_board(resized).board, EXPECTED)

    def test_without_refinement(self):
        config = replace(DEFAULT_VISION_CONFIG, refine_board=False)
        self.assertEqual(read_board(self.image, config).board, EXPECTED)

    def test_every_cell_has_confidence(self):
        reading = read_board(self.image)
        self.assertEqual(len(reading.cells), 16)
        for row in reading.confidences:
            for value in row:
                self.assertGreaterEqual(value, DEFAULT_VISION_CONFIG.min_confidence)
                self.assertLessEqual(value, 1.0)


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class TestBoardDetection(unittest.TestCase):
    def test_configured_rect_scales_with_resolution(self):
        small = configured_rect((1000, 500, 3))
        large = configured_rect((2000, 1000, 3))
        for a, b in zip((large.left, large.top, large.right, large.bottom), (small.left, small.top, small.right, small.bottom)):
            self.assertLessEqual(abs(a - 2 * b), 1)

    def test_refinement_snaps_to_tiles(self):
        region = detect_board(load_image(SCREENSHOT))
        self.assertEqual(region.method, "refined")
        self.assertEqual((region.rect.left, region.rect.top, region.rect.right, region.rect.bottom), (46, 373, 425, 753))

    def test_falls_back_to_configured_region(self):
        blank = np.zeros((1000, 500, 3), np.uint8)
        region = detect_board(blank)
        self.assertEqual(region.method, "configured")
        self.assertEqual(region.rect, configured_rect(blank.shape))

    def test_split_cells(self):
        board = Rect(100, 200, 500, 600)
        cells = split_cells(board)
        self.assertEqual(len(cells), 16)
        self.assertEqual([(c.row, c.col) for c in cells[:5]], [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)])
        self.assertEqual(cells[0].rect, Rect(100, 200, 200, 300))
        self.assertEqual(cells[15].rect, Rect(400, 500, 500, 600))
        for cell in cells:
            lr, r = cell.letter_rect, cell.rect
            self.assertTrue(r.left <= lr.left < lr.right <= r.right)
            self.assertTrue(r.top <= lr.top < lr.bottom <= r.bottom)
        # The letter region ends above the dot row (dots sit at ~73% of the cell height).
        self.assertLessEqual(cells[0].letter_rect.bottom - cells[0].rect.top, 0.72 * 100)


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class TestLetterIsolation(unittest.TestCase):
    def test_drops_dots_and_border_debris(self):
        mask = np.zeros((60, 60), np.uint8)
        mask[10:40, 25:32] = 255  # the letter (an "I" bar)
        mask[50:54, 27:31] = 255  # a dot beneath it
        mask[0:8, 0:8] = 255  # debris touching the border
        isolated = isolate_character(mask, min_blob_fraction=0.15)
        self.assertEqual(isolated.bbox, (25, 10, 7, 30))

    def test_empty_crop(self):
        isolated = isolate_character(np.zeros((40, 40), np.uint8))
        self.assertIsNone(isolated.bbox)
        self.assertFalse(normalize_glyph(isolated.mask).any())

    def test_normalize_keeps_aspect_ratio(self):
        glyph = normalize_glyph(np.full((30, 10), 255, np.uint8), 32)
        ys, xs = np.nonzero(glyph)
        self.assertEqual(glyph.shape, (32, 32))
        self.assertGreater(ys.max() - ys.min(), 2.5 * (xs.max() - xs.min()))


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class TestConfidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = TemplateLibrary.load()
        cls.reading = read_board(load_image(SCREENSHOT))

    def test_empty_glyph_is_flagged_unknown(self):
        rec = TemplateMatchRecognizer(self.library).recognize(np.zeros((32, 32), np.uint8))
        self.assertEqual(rec.letter, "?")
        self.assertTrue(rec.flagged)

    def test_noise_is_flagged(self):
        rng = np.random.default_rng(0)
        noise = (rng.random((32, 32)) > 0.5).astype(np.uint8) * 255
        self.assertTrue(TemplateMatchRecognizer(self.library).recognize(noise).flagged)

    def test_threshold_is_configurable(self):
        strict = replace(DEFAULT_VISION_CONFIG, min_confidence=1.01)
        reading = read_board(self.reading.image, strict)
        self.assertEqual(len(reading.flagged), 16)
        self.assertFalse(reading.is_reliable)

    def test_close_runner_up_is_flagged(self):
        strict = replace(DEFAULT_VISION_CONFIG, min_margin=0.99)
        reading = read_board(self.reading.image, strict)
        self.assertTrue(all("too close" in c.recognition.reason for c in reading.flagged))
        self.assertEqual(len(reading.flagged), 16)


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class TestTemplateLibrary(unittest.TestCase):
    def test_samples_override_synthetic(self):
        a = np.zeros((32, 32), np.uint8)
        lib = TemplateLibrary([Template("A", a, SYNTHETIC), Template("A", a, SAMPLE), Template("B", a, SYNTHETIC)])
        active = lib.active_templates()
        self.assertEqual([t.source for t in active["A"]], [SAMPLE])
        self.assertEqual([t.source for t in active["B"]], [SYNTHETIC])

    def test_save_and_load_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(DEFAULT_VISION_CONFIG, templates_dir=Path(tmp))
            glyph = np.zeros((32, 32), np.uint8)
            glyph[8:24, 14:18] = 255
            path = save_sample("q", glyph, "test", config)
            self.assertEqual(path, Path(tmp) / "samples" / "Q" / "test.png")
            loaded = TemplateLibrary.load(config)
            self.assertEqual(loaded.letters_with_samples(), {"Q"})
            np.testing.assert_array_equal(loaded.by_letter["Q"][0].image, glyph)

    def test_refuses_empty_or_invalid_samples(self):
        with self.assertRaises(ValueError):
            save_sample("A", np.zeros((32, 32), np.uint8), "x")
        with self.assertRaises(ValueError):
            save_sample("1", np.full((32, 32), 255, np.uint8), "x")

    def test_parse_labels(self):
        self.assertEqual(parse_labels("osye,yyem otpe FUND", 4, 4), EXPECTED)
        self.assertEqual(parse_labels("A... .... .... ....", 4, 4)[0], ["A", ".", ".", "."])
        with self.assertRaises(ValueError):
            parse_labels("OSY YYEM OTPE FUND", 4, 4)


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class TestDebugOutput(unittest.TestCase):
    def test_writes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(DEFAULT_VISION_CONFIG, debug_dir=Path(tmp))
            run = save_debug(read_board(load_image(SCREENSHOT)), config, run_name="t")
            for name in ["00_original.png", "01_board_crop.png", "02_annotated.png", "recognition.txt", "recognition.json"]:
                self.assertTrue((run / name).exists(), name)
            self.assertEqual(len(list((run / "tiles").glob("*_tile.png"))), 16)
            self.assertEqual(len(list((run / "letters").glob("*_glyph.png"))), 16)
            self.assertIn("O S Y E", (run / "recognition.txt").read_text(encoding="utf-8"))


@unittest.skipIf(cv2 is None, "OpenCV not installed")
class TestSolverIntegration(unittest.TestCase):
    def test_screenshot_to_best_words(self):
        from word_solver.main import board_from_screenshot
        from word_solver.dictionary import WordDictionary
        from word_solver.solver import Solver
        from contextlib import redirect_stdout
        import io

        with redirect_stdout(io.StringIO()):
            board = board_from_screenshot(SCREENSHOT, debug=False, accept_uncertain=False)
        self.assertEqual(board, EXPECTED)
        result = Solver(WordDictionary(["FOUND", "TOUPEES", "DEPUTES"]).build_trie(3)).solve(board)
        self.assertEqual([w.word for w in result.best_words], ["DEPUTES", "TOUPEES"])


if __name__ == "__main__":
    unittest.main()
