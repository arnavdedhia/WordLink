"""Image helpers: loading, thresholding, isolating a letter, normalizing it.

A "glyph" throughout the vision package is a square ``uint8`` image of size
``glyph_size`` with the letter in white (255) on black (0), scaled to fit and
centered. Templates and screenshot letters go through the same
``normalize_glyph`` step so they can be compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from word_solver.vision.config import DEFAULT_VISION_CONFIG, VisionConfig


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR. Raises FileNotFoundError if it cannot be read."""
    path = Path(path)
    # imdecode handles non-ASCII Windows paths that cv2.imread can choke on.
    data = np.fromfile(str(path), dtype=np.uint8) if path.exists() else None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data is not None and data.size else None
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image for {path}")
    encoded.tofile(str(path))


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize_dark_ink(gray: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """Return a mask where dark pixels (the letter) are 255 and the light tile is 0."""
    if threshold is None:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


@dataclass
class IsolatedCharacter:
    """The letter's pixels, cut out of a thresholded crop."""

    mask: np.ndarray  # letter pixels only, cropped to their bounding box
    bbox: tuple[int, int, int, int] | None  # (x, y, w, h) within the input crop
    component_count: int = 0  # 2 usually means a Qu tile (Q plus a small u)


def isolate_character(mask: np.ndarray, min_blob_fraction: float = 0.15) -> IsolatedCharacter:
    """Keep the letter's connected components and drop dots, noise, and edge debris.

    Components touching the crop border are treated as background (tile edges,
    the gap between tiles). Of the rest, anything much smaller than the
    largest component is discarded, which removes the small dots under the
    letter if any part of them falls inside the crop.

    The only two-letter tile is Qu: a smaller blob sitting to the right of the
    main blob is kept even when it is below ``min_blob_fraction``.
    """
    h, w = mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches_border:
            candidates.append((label, int(area)))

    if not candidates:
        return IsolatedCharacter(mask=np.zeros((0, 0), np.uint8), bbox=None, component_count=0)

    largest_label, largest_area = max(candidates, key=lambda item: item[1])
    keep = [label for label, area in candidates if area >= largest_area * min_blob_fraction]
    if largest_label not in keep:
        keep.append(largest_label)

    companion = _qu_companion_label(candidates, stats, largest_label, largest_area)
    if companion is not None and companion not in keep:
        keep.append(companion)

    letter = np.isin(labels, keep).astype(np.uint8) * 255
    ys, xs = np.nonzero(letter)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return IsolatedCharacter(
        mask=letter[y0:y1, x0:x1],
        bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
        component_count=len(keep),
    )


def _qu_companion_label(candidates, stats, largest_label: int, largest_area: int) -> int | None:
    """Return the label of a small blob to the right of the main letter (the u in Qu)."""
    lx, ly, lw, lh, _ = stats[largest_label]
    main_cx = lx + lw / 2.0
    best = None
    best_area = 0
    for label, area in candidates:
        if label == largest_label or area < largest_area * 0.05:
            continue
        x, y, bw, bh, _ = stats[label]
        cx = x + bw / 2.0
        to_the_right = cx > main_cx and (x + bw) > lx + lw * 0.4
        overlaps_vertically = y < ly + lh and y + bh > ly
        below_the_letter = y >= ly + lh * 0.85
        if to_the_right and overlaps_vertically and not below_the_letter and area > best_area:
            best, best_area = label, area
    return best


def normalize_glyph(letter_mask: np.ndarray, size: int = DEFAULT_VISION_CONFIG.glyph_size) -> np.ndarray:
    """Scale a tight letter mask to fit a ``size`` x ``size`` square, keeping its aspect ratio.

    A small border is left around the letter so template matching can try
    slight shifts. Returns an all-black image for an empty mask.
    """
    canvas = np.zeros((size, size), np.uint8)
    if letter_mask.size == 0:
        return canvas
    border = max(1, size // 16)
    inner = size - 2 * border
    h, w = letter_mask.shape
    scale = inner / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(letter_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def extract_glyph(letter_crop: np.ndarray, config: VisionConfig = DEFAULT_VISION_CONFIG) -> tuple[np.ndarray, IsolatedCharacter]:
    """Full letter-processing chain for one crop: gray, threshold, isolate, normalize."""
    mask = binarize_dark_ink(to_gray(letter_crop), config.ink_threshold)
    isolated = isolate_character(mask, config.min_blob_fraction)
    return normalize_glyph(isolated.mask, config.glyph_size), isolated
