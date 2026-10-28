"""Independent iPhone screenshot checks.

Run these before relying on the web app::

    python -m word_solver.capture list
    python -m word_solver.capture pair
    python -m word_solver.capture prepare
    python -m word_solver.capture screenshot --out word_solver/capture/captures/iphone.png
    python -m word_solver.capture doctor
"""

from __future__ import annotations

import argparse
from pathlib import Path

from word_solver.capture.config import CAPTURES_DIR
from word_solver.capture.iphone_capture import IPhoneCapture


def _print_status(status) -> None:
    print(f"connected:       {status.connected}")
    print(f"message:         {status.message}")
    if status.name:
        print(f"name:            {status.name}")
    if status.udid:
        print(f"udid:            {status.udid}")
    if status.ios:
        print(f"ios:             {status.ios}")
    if status.paired is not None:
        print(f"paired:          {status.paired}")
    if status.developer_mode is not None:
        print(f"developer_mode:  {status.developer_mode}")
    if status.ddi_mounted is not None:
        print(f"ddi_mounted:     {status.ddi_mounted}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test iPhone USB screenshot capture (read-only).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="See whether Windows can see an iPhone over USB.")
    sub.add_parser("pair", help="Pair with the iPhone. Unlock it and tap Trust when asked.")
    sub.add_parser("prepare", help="Enable Developer Mode if needed and mount the Developer Disk Image.")
    shot = sub.add_parser("screenshot", help="Capture one screenshot and save it locally.")
    shot.add_argument("--out", type=Path, default=CAPTURES_DIR / "iphone.png")
    sub.add_parser("doctor", help="Run the five connectivity checks and print a report.")
    args = parser.parse_args(argv)

    capture = IPhoneCapture()
    try:
        if args.command == "list":
            status = capture.status()
            _print_status(status)
            return 0 if status.connected else 2

        if args.command == "pair":
            print("Unlock the iPhone. If a Trust dialog appears, tap Trust.")
            status = capture.pair()
            _print_status(status)
            return 0 if status.paired else 2

        if args.command == "prepare":
            print(capture.prepare_developer())
            _print_status(capture.status())
            return 0

        if args.command == "screenshot":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            frame = capture.capture()
            args.out.write_bytes(frame.png_bytes)
            print(f"Saved {args.out}  ({frame.image_bgr.shape[1]}x{frame.image_bgr.shape[0]}, {frame.capture_ms:.0f} ms)")
            print("This file never left this computer.")
            return 0

        # doctor
        print("1. Detect iPhone over USB")
        status = capture.status()
        _print_status(status)
        if not status.connected:
            print("\nFAILED: Windows does not see an iPhone. Plug in a USB data cable, unlock the phone,")
            print("and confirm Apple Mobile Device Service is running (services.msc).")
            return 2
        print("\n2. Pair / Trust")
        print("   ", "paired" if status.paired else "NOT paired — run: python -m word_solver.capture pair")
        print("\n3. Developer services")
        print("   Developer Mode:", status.developer_mode)
        print("   DDI mounted:   ", status.ddi_mounted)
        if status.developer_mode is False or status.ddi_mounted is False:
            print("   Run: python -m word_solver.capture prepare")
        print("\n4-5. Capture a screenshot and save it locally")
        out = CAPTURES_DIR / "doctor.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = capture.capture()
        out.write_bytes(frame.png_bytes)
        print(f"   Saved {out} ({frame.image_bgr.shape[1]}x{frame.image_bgr.shape[0]}, {frame.capture_ms:.0f} ms)")
        print("\nOK. Live capture can be used in the web app.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        capture.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
