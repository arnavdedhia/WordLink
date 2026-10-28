"""Letter recognition behind a small interface.

``LetterRecognizer`` is the interface; ``TemplateMatchRecognizer`` is the
deterministic implementation. A different approach (for example a trained
classifier) can be swapped in by implementing ``recognize``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from word_solver.vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from word_solver.vision.templates import QU_TILE, TemplateLibrary

UNKNOWN = "?"


@dataclass(frozen=True)
class Recognition:
    letter: str  # predicted letter, or "?" if nothing could be matched
    confidence: float  # best match score, 0.0 to 1.0
    runner_up: str | None  # second most likely letter
    margin: float  # confidence minus the runner-up's score
    source: str | None  # "sample" or "synthetic": which kind of template matched
    flagged: bool
    reason: str | None  # why the cell is flagged


class LetterRecognizer(ABC):
    @abstractmethod
    def recognize(self, glyph: np.ndarray, component_count: int = 1) -> Recognition:
        """Classify one normalized glyph (see ``image_processor.normalize_glyph``).

        ``component_count`` is the number of ink blobs in the tile. Two blobs
        means this is the Qu tile (the only two-letter square).
        """


class TemplateMatchRecognizer(LetterRecognizer):
    """Compares a glyph against every letter's templates with normalized cross-correlation.

    For each letter the score is its best template match. The glyph is padded
    by a few pixels so matching can absorb small shifts.
    """

    def __init__(self, library: TemplateLibrary, config: VisionConfig = DEFAULT_VISION_CONFIG) -> None:
        self.config = config
        active = library.active_templates()
        if not active:
            raise ValueError(
                f"No letter templates found in {config.templates_dir}. "
                "Run: python -m word_solver.vision.templates synthetic"
            )
        self.shift = max(1, config.glyph_size // 16)
        self.templates = {
            letter: [(self._prepare(t.image), t.source) for t in templates]
            for letter, templates in active.items()
        }

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        if self.config.match_blur_sigma > 0:
            image = cv2.GaussianBlur(image, (0, 0), self.config.match_blur_sigma)
        return image

    def score_letters(self, glyph: np.ndarray) -> dict[str, tuple[float, str]]:
        """Best score and template source for every letter."""
        padded = cv2.copyMakeBorder(glyph, *[self.shift] * 4, cv2.BORDER_CONSTANT, value=0)
        padded = self._prepare(padded)
        scores = {}
        for letter, templates in self.templates.items():
            best, best_source = -1.0, templates[0][1]
            for image, source in templates:
                score = float(cv2.matchTemplate(padded, image, cv2.TM_CCOEFF_NORMED).max())
                if score > best:
                    best, best_source = score, source
            scores[letter] = (best, best_source)
        return scores

    def recognize(self, glyph: np.ndarray, component_count: int = 1) -> Recognition:
        if not glyph.any():
            return Recognition(UNKNOWN, 0.0, None, 0.0, None, True, "no letter found in tile")

        ranked = sorted(self.score_letters(glyph).items(), key=lambda kv: kv[1][0], reverse=True)
        letter, (score, source) = ranked[0]
        letter, score, source, ranked = self._prefer_qu(ranked, component_count)
        confidence = max(0.0, min(1.0, score))
        runner_up, runner_score = (ranked[1][0], max(0.0, ranked[1][1][0])) if len(ranked) > 1 else (None, 0.0)
        margin = confidence - runner_score

        reasons = []
        if confidence < self.config.min_confidence:
            reasons.append(f"confidence {confidence:.2f} < {self.config.min_confidence:.2f}")
        if runner_up is not None and margin < self.config.min_margin:
            reasons.append(f"too close to {runner_up} (margin {margin:.2f})")
        return Recognition(
            letter=letter,
            confidence=confidence,
            runner_up=runner_up,
            margin=margin,
            source=source,
            flagged=bool(reasons),
            reason="; ".join(reasons) or None,
        )

    def _prefer_qu(self, ranked, component_count: int):
        """Qu is the only two-letter tile and is often confused with W."""
        scores = {letter: (score, source) for letter, (score, source) in ranked}
        if QU_TILE not in scores:
            letter, (score, source) = ranked[0]
            return letter, score, source, ranked

        qu_score, qu_source = scores[QU_TILE]
        best_letter, (best_score, best_source) = ranked[0]
        choose_qu = False
        if component_count >= 2 and qu_score >= 0.35:
            choose_qu = True
        elif best_letter == "W" and qu_score >= best_score - 0.12 and qu_score >= 0.50:
            choose_qu = True
        if not choose_qu:
            return best_letter, best_score, best_source, ranked

        rest = [(letter, val) for letter, val in ranked if letter != QU_TILE]
        new_ranked = [(QU_TILE, (qu_score, qu_source)), *rest]
        return QU_TILE, qu_score, qu_source, new_ranked
