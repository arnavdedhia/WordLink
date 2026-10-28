"""Template library for letter recognition, plus a CLI to grow it.

Layout on disk (under ``VisionConfig.templates_dir``)::

    samples/<LETTER>/<name>.png     real glyphs cut from labeled screenshots
    synthetic/<LETTER>/<font>.png   glyphs rendered from system fonts

Real samples are authoritative. A letter's synthetic template is only used
while that letter has no real samples, so recognition improves as samples
are added.

CLI (run from the repository root)::

    # Save all 16 letters from a screenshot whose board you know.
    python -m word_solver.vision.templates add --image shot.png --labels "OSYE YYEM OTPE FUND"

    # Use "." for cells you do not want to save.
    python -m word_solver.vision.templates add --image shot.png --labels "..A. .... .... ...."

    # Rebuild the font-rendered fallbacks and show what the library holds.
    python -m word_solver.vision.templates synthetic
    python -m word_solver.vision.templates synthetic --font C:/Windows/Fonts/arial.ttf
    python -m word_solver.vision.templates list
"""

from __future__ import annotations

import argparse
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from word_solver.vision.config import DEFAULT_VISION_CONFIG, VisionConfig
from word_solver.vision.image_processor import (
    binarize_dark_ink,
    isolate_character,
    load_image,
    normalize_glyph,
    save_image,
    to_gray,
)

LETTERS = string.ascii_uppercase
SAMPLE = "sample"
SYNTHETIC = "synthetic"
# Windows fonts whose combined shapes best matched the game's letters on the
# reference screenshot. Missing fonts are skipped.
DEFAULT_SYNTHETIC_FONTS = (
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("C:/Windows/Fonts/corbel.ttf"),
    Path("C:/Windows/Fonts/framd.ttf"),
)


@dataclass
class Template:
    letter: str
    image: np.ndarray  # normalized glyph, uint8, white letter on black
    source: str  # SAMPLE or SYNTHETIC
    path: Path | None = None


class TemplateLibrary:
    """All templates, grouped by letter."""

    def __init__(self, templates: list[Template] | None = None) -> None:
        self.by_letter: dict[str, list[Template]] = {}
        for template in templates or []:
            self.add(template)

    def add(self, template: Template) -> None:
        self.by_letter.setdefault(template.letter, []).append(template)

    def active_templates(self) -> dict[str, list[Template]]:
        """Templates used for matching: real samples when present, otherwise synthetic."""
        active = {}
        for letter, templates in self.by_letter.items():
            samples = [t for t in templates if t.source == SAMPLE]
            active[letter] = samples or templates
        return active

    def letters_with_samples(self) -> set[str]:
        return {l for l, ts in self.by_letter.items() if any(t.source == SAMPLE for t in ts)}

    def __len__(self) -> int:
        return sum(len(ts) for ts in self.by_letter.values())

    @classmethod
    def load(cls, config: VisionConfig = DEFAULT_VISION_CONFIG, include_synthetic: bool = True) -> TemplateLibrary:
        library = cls()
        root = config.templates_dir
        for path in sorted((root / "samples").glob("*/*.png")):
            letter = path.parent.name.upper()
            if letter in LETTERS:
                library.add(Template(letter, _load_glyph(path, config), SAMPLE, path))
        if include_synthetic:
            for path in sorted((root / "synthetic").glob("*/*.png")):
                letter = path.parent.name.upper()
                if letter in LETTERS:
                    library.add(Template(letter, _load_glyph(path, config), SYNTHETIC, path))
        return library


def _load_glyph(path: Path, config: VisionConfig) -> np.ndarray:
    glyph = to_gray(load_image(path))
    if glyph.shape != (config.glyph_size, config.glyph_size):
        raise ValueError(f"Template {path} is {glyph.shape}, expected {config.glyph_size}x{config.glyph_size}.")
    return glyph


def save_sample(letter: str, glyph: np.ndarray, name: str, config: VisionConfig = DEFAULT_VISION_CONFIG) -> Path:
    """Store a labeled glyph as a real sample. Returns the file path."""
    letter = letter.upper()
    if letter not in LETTERS:
        raise ValueError(f"Not a letter A-Z: {letter!r}")
    if not glyph.any():
        raise ValueError(f"Refusing to save an empty glyph for {letter}.")
    path = config.templates_dir / "samples" / letter / f"{name}.png"
    save_image(path, glyph)
    return path


def render_font_glyph(letter: str, font_path: Path, config: VisionConfig = DEFAULT_VISION_CONFIG) -> np.ndarray:
    """Render ``letter`` with a TrueType font and normalize it like a screenshot glyph."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path), 160)
    canvas = Image.new("L", (240, 240), 255)
    ImageDraw.Draw(canvas).text((40, 20), letter, font=font, fill=0)
    mask = binarize_dark_ink(np.array(canvas), threshold=128)
    isolated = isolate_character(mask, min_blob_fraction=0.0)
    return normalize_glyph(isolated.mask, config.glyph_size)


def generate_synthetic(
    font_paths: tuple[Path, ...] | list[Path] = DEFAULT_SYNTHETIC_FONTS,
    config: VisionConfig = DEFAULT_VISION_CONFIG,
) -> list[Path]:
    """Replace ``templates/synthetic`` with one rendered template per letter per available font."""
    fonts = [Path(f) for f in font_paths if Path(f).exists()]
    if not fonts:
        raise FileNotFoundError(f"None of these fonts exist: {[str(f) for f in font_paths]}")
    synthetic_dir = config.templates_dir / "synthetic"
    for old in synthetic_dir.glob("*/*.png"):
        old.unlink()
    paths = []
    for font in fonts:
        for letter in LETTERS:
            path = synthetic_dir / letter / f"{font.stem.lower()}.png"
            save_image(path, render_font_glyph(letter, font, config))
            paths.append(path)
    return paths


def parse_labels(text: str, rows: int, cols: int) -> list[list[str]]:
    """Parse "OSYE YYEM OTPE FUND" (or comma separated) into a grid. "." marks a cell to skip."""
    parts = text.replace(",", " ").split()
    if len(parts) != rows or any(len(p) != cols for p in parts):
        raise ValueError(f"Labels must be {rows} rows of {cols} characters, e.g. 'OSYE YYEM OTPE FUND'.")
    grid = [list(p.upper()) for p in parts]
    for row in grid:
        for ch in row:
            if ch != "." and ch not in LETTERS:
                raise ValueError(f"Invalid label {ch!r}; use A-Z or '.'.")
    return grid


def add_samples_from_screenshot(image_path: Path, labels: str, config: VisionConfig = DEFAULT_VISION_CONFIG, name: str | None = None) -> list[Path]:
    """Cut every labeled cell out of a screenshot and store it as a sample."""
    from word_solver.vision.board_reader import extract_cells

    grid = parse_labels(labels, config.grid_rows, config.grid_cols)
    image = load_image(image_path)
    prefix = name or Path(image_path).stem
    saved = []
    for cell in extract_cells(image, config).cells:
        letter = grid[cell.row][cell.col]
        if letter == ".":
            continue
        saved.append(save_sample(letter, cell.glyph, f"{prefix}_r{cell.row}c{cell.col}", config))
    return saved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage letter templates.")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Save labeled letters from a screenshot as samples.")
    add.add_argument("--image", required=True, type=Path)
    add.add_argument("--labels", required=True, help='Board rows, e.g. "OSYE YYEM OTPE FUND"; "." skips a cell.')
    add.add_argument("--name", help="File name prefix (defaults to the image file name).")

    syn = sub.add_parser("synthetic", help="Regenerate font-rendered fallback templates.")
    syn.add_argument("--font", type=Path, action="append", help="TrueType font; repeat for several. Defaults to Arial Bold, Corbel, Franklin Gothic Medium.")

    sub.add_parser("list", help="Show how many templates each letter has.")

    args = parser.parse_args(argv)
    if args.command == "add":
        saved = add_samples_from_screenshot(args.image, args.labels, name=args.name)
        print(f"Saved {len(saved)} samples to {DEFAULT_VISION_CONFIG.templates_dir / 'samples'}")
    elif args.command == "synthetic":
        paths = generate_synthetic(args.font or DEFAULT_SYNTHETIC_FONTS)
        print(f"Wrote {len(paths)} synthetic templates to {DEFAULT_VISION_CONFIG.templates_dir / 'synthetic'}")
    else:
        library = TemplateLibrary.load()
        sampled = library.letters_with_samples()
        for letter in LETTERS:
            templates = library.by_letter.get(letter, [])
            n_samples = sum(t.source == SAMPLE for t in templates)
            status = f"{n_samples} sample(s)" if letter in sampled else ("synthetic only" if templates else "MISSING")
            print(f"  {letter}: {status}")


if __name__ == "__main__":
    main()
