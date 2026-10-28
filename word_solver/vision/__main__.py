"""Recognize a screenshot without solving: python -m word_solver.vision --image shot.png [--debug]"""

import argparse
import sys
from pathlib import Path

from word_solver.vision.debug import format_reading
from word_solver.vision.pipeline import recognize_screenshot

parser = argparse.ArgumentParser(description="Read the letter board from a game screenshot.")
parser.add_argument("--image", required=True, type=Path)
parser.add_argument("--debug", action="store_true", help="Save debug images to word_solver/vision/debug/.")
args = parser.parse_args()

result = recognize_screenshot(args.image, debug=args.debug)
print(format_reading(result.reading))
if result.debug_dir:
    print(f"\nDebug output: {result.debug_dir}")
sys.exit(0 if result.reading.is_reliable else 2)
