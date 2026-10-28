"""Measure recognition accuracy on screenshots whose boards you know.

Two checks per screenshot:

* ``current``: recognize the board with the template library as it is now.
* ``leave-out``: for each letter on the board, hide that letter's real
  samples and recognize it from the synthetic fallback alone. This estimates
  how the recognizer behaves when the game shows a letter you have not
  sampled yet.

Usage (repository root)::

    python -m word_solver.vision.evaluate --image shot.png --labels "OSYE YYEM OTPE FUND"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from word_solver.vision.board_reader import extract_cells
from word_solver.vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from word_solver.vision.image_processor import load_image
from word_solver.vision.letter_recognizer import Recognition, TemplateMatchRecognizer
from word_solver.vision.templates import SAMPLE, TemplateLibrary, parse_labels


def _line(label: str, truth: str, rec: Recognition) -> str:
    verdict = "correct" if rec.letter == truth else "WRONG"
    flag = f"FLAGGED ({rec.reason})" if rec.flagged else "not flagged"
    return (
        f"  {label:<8} truth {truth}  got {rec.letter}  conf {rec.confidence:.3f}  "
        f"runner-up {rec.runner_up} margin {rec.margin:.3f}  [{rec.source}]  {verdict}, {flag}"
    )


def evaluate(image_path: Path, labels: str, config: VisionConfig = DEFAULT_VISION_CONFIG) -> bool:
    """Print both checks. Returns True if no cell was wrong without being flagged."""
    grid = parse_labels(labels, config.grid_rows, config.grid_cols)
    cells = extract_cells(load_image(image_path), config).cells
    library = TemplateLibrary.load(config)
    safe = True

    print("Current library:")
    recognizer = TemplateMatchRecognizer(library, config)
    for cell in cells:
        truth = grid[cell.row][cell.col]
        rec = recognizer.recognize(cell.glyph)
        safe &= rec.letter == truth or rec.flagged
        print(_line(f"[{cell.row},{cell.col}]", truth, rec))

    print("\nLeave-out (letter's own samples hidden):")
    for held in sorted({ch for row in grid for ch in row} - {"."}):
        reduced = TemplateLibrary(
            [t for ts in library.by_letter.values() for t in ts if not (t.letter == held and t.source == SAMPLE)]
        )
        recognizer = TemplateMatchRecognizer(reduced, config)
        for cell in cells:
            if grid[cell.row][cell.col] == held:
                rec = recognizer.recognize(cell.glyph)
                safe &= rec.letter == held or rec.flagged
                print(_line(f"[{cell.row},{cell.col}]", held, rec))

    print("\nNo silent errors." if safe else "\nWARNING: at least one wrong letter was NOT flagged.")
    return safe


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate letter recognition on a labeled screenshot.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--labels", required=True, help='Board rows, e.g. "OSYE YYEM OTPE FUND".')
    args = parser.parse_args(argv)
    evaluate(args.image, args.labels)


if __name__ == "__main__":
    main()
