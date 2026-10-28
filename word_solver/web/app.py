"""FastAPI backend. Serves the API under /api and the single-page UI from /static.

All endpoints return the same "state" object (see ``BoardService._snapshot``),
so the page can simply re-render whatever comes back.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from word_solver.capture.capture_manager import CaptureManager, CaptureSnapshot
from word_solver.web.service import BoardService

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class CellUpdate(BaseModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    letter: str


class BoardUpdate(BaseModel):
    rows: list[str]


class CaptureToggle(BaseModel):
    debug: bool = False


def _idle_capture() -> dict:
    return CaptureSnapshot(
        enabled=False,
        phase="idle",
        stale=False,
        source={"kind": "iphone", "connected": False, "message": "Live iPhone capture is not running."},
    ).to_dict()


def merge_state(app: FastAPI) -> dict:
    data = app.state.service.state()
    manager: CaptureManager | None = getattr(app.state, "capture", None)
    if manager is None:
        data["capture"] = _idle_capture()
        data["stale"] = False
        return data
    snap = manager.snapshot()
    data["capture"] = snap.to_dict()
    data["stale"] = snap.stale
    return data


def attach_iphone_capture(app: FastAPI, debug: bool = False) -> CaptureManager:
    """Start watching a USB iPhone. Safe to call when none is plugged in."""
    from word_solver.capture.iphone_capture import IPhoneCapture

    def on_frame(image, name, capture_ms, debug_flag):
        manager = getattr(app.state, "capture", None)
        stabilize_ms = float(getattr(manager, "last_stabilize_ms", 0.0) or 0.0)
        return app.state.service.ingest_bgr(
            image,
            name,
            origin="iphone",
            debug=debug_flag,
            capture_ms=capture_ms,
            stabilize_ms=stabilize_ms,
        )

    source = IPhoneCapture()
    manager = CaptureManager(source, on_frame)
    app.state.capture = manager
    manager.start(debug=debug)
    return manager


def create_app(service: BoardService | None = None, capture: CaptureManager | None = None) -> FastAPI:
    app = FastAPI(title="Word Link Solver")
    app.state.service = service or BoardService()
    app.state.capture = capture

    def svc() -> BoardService:
        return app.state.service

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    def get_state() -> dict:
        return merge_state(app)

    @app.post("/api/screenshot")
    async def upload_screenshot(file: UploadFile = File(...), debug: bool = Form(False)) -> dict:
        data = await file.read()
        if not data:
            raise HTTPException(400, "The uploaded file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Screenshot is larger than 25 MB.")
        try:
            svc().load_screenshot(data, file.filename or "upload", debug=debug, origin="screenshot")
            return merge_state(app)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/board")
    def set_board(update: BoardUpdate) -> dict:
        try:
            svc().set_board(update.rows)
            return merge_state(app)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/cell")
    def set_cell(update: CellUpdate) -> dict:
        try:
            svc().set_cell(update.row, update.col, update.letter)
            return merge_state(app)
        except LookupError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/current-board")
    def current_board() -> dict:
        return merge_state(app)

    @app.post("/api/reject-current-word")
    def reject_current_word() -> dict:
        try:
            extra = svc().reject_current()
            data = merge_state(app)
            data.update({k: extra[k] for k in ("rejected_word", "persist_ms", "switch_ms") if k in extra})
            return data
        except LookupError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/next-word")
    def next_word() -> dict:
        try:
            extra = svc().next_word()
            data = merge_state(app)
            if "switch_ms" in extra:
                data["switch_ms"] = extra["switch_ms"]
            return data
        except LookupError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/undo-rejection")
    def undo_rejection() -> dict:
        try:
            extra = svc().undo_rejection()
            data = merge_state(app)
            if "undone_word" in extra:
                data["undone_word"] = extra["undone_word"]
            return data
        except LookupError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/capture/start")
    def capture_start(body: CaptureToggle | None = None) -> dict:
        debug = body.debug if body else False
        manager = app.state.capture
        if manager is None:
            attach_iphone_capture(app, debug=debug)
        else:
            manager.set_debug(debug)
            manager.start(debug=debug)
        return merge_state(app)

    @app.post("/api/capture/stop")
    def capture_stop() -> dict:
        manager = app.state.capture
        if manager is not None:
            manager.stop()
        return merge_state(app)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
