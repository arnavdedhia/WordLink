"""Board-change detection and capture-manager behaviour, without an iPhone."""

import time
import unittest

import numpy as np

from word_solver.capture.capture_manager import CaptureManager
from word_solver.capture.change_detector import BoardPhase, ChangeDetector, crop_board
from word_solver.capture.config import CaptureConfig
from word_solver.capture.sources import CaptureError, DisconnectedError, SourceStatus
from word_solver.vision.config import DEFAULT_VISION_CONFIG


def board_image(value: int, noise: int = 0, seed: int = 0) -> np.ndarray:
    """A screenshot-sized BGR image whose board rectangle is filled with ``value``."""
    h, w = 800, 400
    image = np.full((h, w, 3), 30, np.uint8)
    rect = (
        round(DEFAULT_VISION_CONFIG.board_left * w),
        round(DEFAULT_VISION_CONFIG.board_top * h),
        round(DEFAULT_VISION_CONFIG.board_right * w),
        round(DEFAULT_VISION_CONFIG.board_bottom * h),
    )
    image[rect[1] : rect[3], rect[0] : rect[2]] = value
    if noise:
        rng = np.random.default_rng(seed)
        patch = image[rect[1] : rect[3], rect[0] : rect[2]]
        delta = rng.integers(-noise, noise + 1, patch.shape, dtype=np.int16)
        image[rect[1] : rect[3], rect[0] : rect[2]] = np.clip(patch.astype(np.int16) + delta, 0, 255).astype(np.uint8)
    return image


FAST = CaptureConfig(
    poll_interval_seconds=0.02,
    reconnect_interval_seconds=0.05,
    change_threshold=0.04,
    settle_threshold=0.015,
    settle_frames=3,
    settle_timeout_seconds=2.0,
)


class SequenceSource:
    def __init__(self, frames, fail_after: int | None = None, disconnect_after: int | None = None):
        self.frames = list(frames)
        self.i = 0
        self.fail_after = fail_after
        self.disconnect_after = disconnect_after
        self.closed = False

    def status(self) -> SourceStatus:
        if self.disconnect_after is not None and self.i >= self.disconnect_after:
            return SourceStatus("iphone", False, "unplugged")
        return SourceStatus("iphone", True, "connected", name="FakePhone")

    def capture(self):
        from word_solver.capture.sources import CapturedFrame, encode_png

        if self.disconnect_after is not None and self.i >= self.disconnect_after:
            raise DisconnectedError("unplugged")
        if self.fail_after is not None and self.i >= self.fail_after:
            self.i += 1
            raise CaptureError("capture failed")
        image = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return CapturedFrame(image, encode_png(image), 1.0, "FakePhone")

    def close(self) -> None:
        self.closed = True


class TestChangeDetector(unittest.TestCase):
    def test_first_stable_board_is_reported_after_settle_frames(self):
        detector = ChangeDetector(FAST)
        still = board_image(40)
        phases = [detector.observe(still).phase for _ in range(6)]
        self.assertIn(BoardPhase.STABLE, phases)
        self.assertEqual(phases[-1], BoardPhase.UNCHANGED)

    def test_falling_letters_do_not_count_as_stable(self):
        detector = ChangeDetector(FAST)
        still = board_image(40)
        for _ in range(5):
            detector.observe(still)
        noisy = [detector.observe(board_image(40, noise=80, seed=i)).phase for i in range(6)]
        self.assertTrue(all(p is BoardPhase.SETTLING for p in noisy))

    def test_new_board_after_animation(self):
        detector = ChangeDetector(FAST)
        old = board_image(40)
        for _ in range(5):
            detector.observe(old)
        for i in range(4):
            self.assertEqual(detector.observe(board_image(40, noise=90, seed=i)).phase, BoardPhase.SETTLING)
        new = board_image(200)
        phases = [detector.observe(new).phase for _ in range(6)]
        self.assertIn(BoardPhase.STABLE, phases)

    def test_score_area_outside_board_is_ignored(self):
        detector = ChangeDetector(FAST)
        a = board_image(40)
        for _ in range(5):
            detector.observe(a)
        b = a.copy()
        b[0:40, :, :] = 255
        self.assertEqual(detector.observe(b).phase, BoardPhase.UNCHANGED)

    def test_crop_matches_configured_region(self):
        image = board_image(77)
        crop = crop_board(image)
        self.assertTrue(np.all(crop == 77))


class TestCaptureManager(unittest.TestCase):
    def test_solves_only_after_stable_frames(self):
        frames = [board_image(40, noise=70, seed=i) for i in range(4)] + [board_image(90)] * 8
        solved = []
        source = SequenceSource(frames)
        manager = CaptureManager(source, lambda img, name, ms, dbg: solved.append(name) or {"recognize_ms": 1, "solve_ms": 1}, FAST)
        manager.start()
        deadline = time.time() + 3
        while time.time() < deadline and len(solved) < 1:
            time.sleep(0.05)
        manager.stop()
        self.assertEqual(len(solved), 1)

    def test_disconnect_keeps_last_board_stale(self):
        frames = [board_image(80)] * 10
        solved = []
        source = SequenceSource(frames, disconnect_after=8)
        manager = CaptureManager(source, lambda img, name, ms, dbg: solved.append(1) or {"recognize_ms": 1, "solve_ms": 1}, FAST)
        manager.start()
        deadline = time.time() + 3
        while time.time() < deadline and not manager.snapshot().stale:
            time.sleep(0.05)
        snap = manager.snapshot()
        manager.stop()
        self.assertTrue(solved)
        self.assertTrue(snap.stale)
        self.assertEqual(snap.phase, "disconnected")
        self.assertFalse(snap.source["connected"])

    def test_failed_capture_is_stale_not_a_new_board(self):
        frames = [board_image(80)] * 10
        solved = []
        source = SequenceSource(frames, fail_after=8)
        manager = CaptureManager(source, lambda img, name, ms, dbg: solved.append(1) or {"recognize_ms": 1, "solve_ms": 1}, FAST)
        manager.start()
        deadline = time.time() + 3
        while time.time() < deadline and manager.snapshot().phase != "error":
            time.sleep(0.05)
        snap = manager.snapshot()
        manager.stop()
        self.assertTrue(solved)
        self.assertTrue(snap.stale)
        self.assertEqual(len(solved), 1)


if __name__ == "__main__":
    unittest.main()
