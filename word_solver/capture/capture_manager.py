"""Watch an image source, wait for a stable board, then recognize and solve.

The manager never drives the phone. It only reads frames. Failed captures leave
the last solved board in place and mark it stale.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from word_solver.capture.change_detector import BoardPhase, ChangeDetector
from word_solver.capture.config import DEFAULT_CAPTURE_CONFIG, CaptureConfig
from word_solver.capture.sources import CaptureError, DisconnectedError, ImageSource, SourceStatus
from word_solver.vision.config import DEFAULT_VISION_CONFIG, VisionConfig


@dataclass
class Timings:
    capture_ms: float = 0.0
    recognize_ms: float = 0.0
    solve_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "capture_ms": round(self.capture_ms, 1),
            "recognize_ms": round(self.recognize_ms, 1),
            "solve_ms": round(self.solve_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


@dataclass
class CaptureSnapshot:
    enabled: bool
    phase: str
    stale: bool
    source: dict
    timings: dict | None = None
    last_success_at: str | None = None
    debug: bool = False

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "phase": self.phase,
            "stale": self.stale,
            "source": self.source,
            "timings": self.timings,
            "last_success_at": self.last_success_at,
            "debug": self.debug,
        }


class CaptureManager:
    def __init__(
        self,
        source: ImageSource,
        on_frame: Callable,  # (image_bgr, name, timings, debug) -> None
        capture_config: CaptureConfig = DEFAULT_CAPTURE_CONFIG,
        vision_config: VisionConfig = DEFAULT_VISION_CONFIG,
    ) -> None:
        self.source = source
        self.on_frame = on_frame
        self.config = capture_config
        self.detector = ChangeDetector(capture_config, vision_config)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._enabled = False
        self._phase = "idle"
        self._stale = False
        self._had_board = False
        self._timings: Timings | None = None
        self._last_success_at: datetime | None = None
        self._debug = False
        self._source_status: SourceStatus | None = None
        self._settle_started: float | None = None
        self.last_stabilize_ms = 0.0

    # ----- control -------------------------------------------------------

    def start(self, debug: bool = False) -> None:
        with self._lock:
            self._debug = debug
            if self._thread and self._thread.is_alive():
                self._enabled = True
                return
            self._enabled = True
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="capture-manager", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._enabled = False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self._set_phase("idle")

    def set_debug(self, debug: bool) -> None:
        self._debug = debug

    def snapshot(self) -> CaptureSnapshot:
        with self._lock:
            status = self._source_status or self._safe_status()
            return CaptureSnapshot(
                enabled=self._enabled,
                phase=self._phase,
                stale=self._stale,
                source=status.to_dict(),
                timings=self._timings.to_dict() if self._timings else None,
                last_success_at=self._last_success_at.isoformat(timespec="seconds") if self._last_success_at else None,
                debug=self._debug,
            )

    # ----- loop ----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._enabled:
                time.sleep(0.2)
                continue
            try:
                self._tick()
                time.sleep(self.config.poll_interval_seconds)
            except DisconnectedError as exc:
                self._on_disconnect(str(exc))
                time.sleep(self.config.reconnect_interval_seconds)
            except CaptureError as exc:
                self._on_capture_fail(str(exc))
                time.sleep(self.config.reconnect_interval_seconds)
            except Exception as exc:
                self._on_capture_fail(str(exc))
                time.sleep(self.config.reconnect_interval_seconds)

    def _tick(self) -> None:
        status = self.source.status()
        with self._lock:
            self._source_status = status
        if not status.connected:
            raise DisconnectedError(status.message)

        frame = self.source.capture()
        observed = self.detector.observe(frame.image_bgr)

        if observed.phase is BoardPhase.SETTLING:
            self._set_phase("settling")
            now = time.monotonic()
            if self._settle_started is None:
                self._settle_started = now
            elif now - self._settle_started > self.config.settle_timeout_seconds:
                # Animation took too long; keep waiting rather than solving a mid-drop frame.
                self._settle_started = now
            return

        stabilize_ms = 0.0
        if self._settle_started is not None:
            stabilize_ms = (time.monotonic() - self._settle_started) * 1000
        self._settle_started = None
        self.last_stabilize_ms = stabilize_ms
        if observed.phase is BoardPhase.UNCHANGED:
            self._set_phase("watching")
            with self._lock:
                self._stale = False
            return

        # STABLE: a new board that has stopped moving.
        self._set_phase("solving")
        total_start = time.perf_counter()
        try:
            result = self.on_frame(frame.image_bgr, frame.source_name, frame.capture_ms, self._debug)
        except Exception as exc:
            self._on_capture_fail(f"Recognition failed: {exc}")
            return
        timings = Timings(
            capture_ms=frame.capture_ms,
            recognize_ms=float((result or {}).get("recognize_ms") or 0),
            solve_ms=float((result or {}).get("solve_ms") or 0),
        )
        timings.total_ms = (time.perf_counter() - total_start) * 1000 + frame.capture_ms
        with self._lock:
            self._timings = timings
            self._stale = False
            self._had_board = True
            self._last_success_at = datetime.now()
            self._phase = "watching"

    def _on_disconnect(self, message: str) -> None:
        self.detector.reset()
        try:
            self.source.close()
        except Exception:
            pass
        with self._lock:
            self._phase = "disconnected"
            self._stale = self._had_board
            self._source_status = SourceStatus(kind=getattr(self._source_status, "kind", "iphone"), connected=False, message=message)

    def _on_capture_fail(self, message: str) -> None:
        with self._lock:
            self._phase = "error"
            self._stale = self._had_board
            kind = self._source_status.kind if self._source_status else "iphone"
            connected = self._source_status.connected if self._source_status else False
            self._source_status = SourceStatus(kind=kind, connected=connected, message=message)

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def _safe_status(self) -> SourceStatus:
        try:
            return self.source.status()
        except Exception as exc:
            return SourceStatus(kind="iphone", connected=False, message=str(exc))
