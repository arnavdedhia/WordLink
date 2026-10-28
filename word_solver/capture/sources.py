"""Common image source interface.

The rest of the app (solver, vision, web UI) only sees ``CapturedFrame``
objects. It does not care whether the pixels came from an iPhone, a file,
or an upload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


class CaptureError(Exception):
    """A screenshot could not be taken. The last known board should stay visible."""


class DisconnectedError(CaptureError):
    """No usable device is attached."""


@dataclass
class SourceStatus:
    kind: str  # "iphone", "file", "memory"
    connected: bool
    message: str
    udid: str | None = None
    name: str | None = None
    ios: str | None = None
    paired: bool | None = None
    developer_mode: bool | None = None
    ddi_mounted: bool | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "connected": self.connected,
            "message": self.message,
            "udid": self.udid,
            "name": self.name,
            "ios": self.ios,
            "paired": self.paired,
            "developer_mode": self.developer_mode,
            "ddi_mounted": self.ddi_mounted,
        }


@dataclass
class CapturedFrame:
    image_bgr: np.ndarray
    png_bytes: bytes
    capture_ms: float
    source_name: str
    extra: dict = field(default_factory=dict)


class ImageSource(Protocol):
    """Anything that can produce a BGR screenshot."""

    def status(self) -> SourceStatus:
        ...

    def capture(self) -> CapturedFrame:
        ...

    def close(self) -> None:
        ...


def decode_png(data: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise CaptureError("The captured bytes are not a readable image.")
    return image


def encode_png(image: np.ndarray) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise CaptureError("Could not encode the screenshot as PNG.")
    return encoded.tobytes()


class FileImageSource:
    """Replay one image file. Used by tests and as a stand-in for a live phone."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._image: np.ndarray | None = None

    def status(self) -> SourceStatus:
        return SourceStatus(
            kind="file",
            connected=self.path.exists(),
            message=str(self.path) if self.path.exists() else f"File not found: {self.path}",
            name=self.path.name,
        )

    def capture(self) -> CapturedFrame:
        import time

        start = time.perf_counter()
        if self._image is None:
            from word_solver.vision.image_processor import load_image

            self._image = load_image(self.path)
        elapsed = (time.perf_counter() - start) * 1000
        return CapturedFrame(self._image, encode_png(self._image), elapsed, self.path.name)

    def close(self) -> None:
        self._image = None
