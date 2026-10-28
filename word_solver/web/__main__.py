"""Start the local web interface: python -m word_solver.web [--port 8000] [--open]"""

import argparse
import threading
import webbrowser

import uvicorn

from word_solver.web.app import attach_iphone_capture, create_app

parser = argparse.ArgumentParser(description="Run the Word Link solver web interface.")
parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to reach it from other devices on your network.")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--open", action="store_true", help="Open the page in your default browser.")
parser.add_argument("--no-iphone", action="store_true", help="Do not watch USB for an iPhone.")
args = parser.parse_args()

print("Loading dictionary...")
app = create_app()
if not args.no_iphone:
    print("Starting iPhone watcher (read-only; safe if no phone is plugged in)...")
    attach_iphone_capture(app)
url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
print(f"Ready: {url}   (Ctrl+C to stop)")
if args.open:
    threading.Timer(1.0, webbrowser.open, args=(url,)).start()
uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
