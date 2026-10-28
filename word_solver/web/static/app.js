"use strict";

// Every API call returns the full board state; render() redraws from it.
let state = null;
let editing = false;
let editingCell = null; // [row, col] open in the dialog

const $ = (id) => document.getElementById(id);
const SIZE = 4;
const CELL = 400 / SIZE; // SVG viewBox units per cell
const SVG_NS = "http://www.w3.org/2000/svg";

// ----- API -----------------------------------------------------------------

async function api(path, options = {}, { busy = true } = {}) {
  if (busy) showBusy(true);
  try {
    const response = await fetch(path, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(formatApiError(body, response.status));
    showError(null);
    return body;
  } catch (err) {
    showError(err.message);
    throw err;
  } finally {
    if (busy) showBusy(false);
  }
}

function apiLocal(path) {
  return api(path, { method: "POST" }, { busy: false });
}

async function uploadScreenshot(file) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file, file.name || "pasted.png");
  form.append("debug", $("debug-check").checked ? "true" : "false");
  applyServerState(await api("/api/screenshot", { method: "POST", body: form }));
}

async function submitBoard(text) {
  const rows = text.toUpperCase().replace(/,/g, " ").trim().split(/\s+/);
  applyServerState(await api("/api/board", jsonBody({ rows })));
}

async function setCell(row, col, letter) {
  applyServerState(await api("/api/cell", jsonBody({ row, col, letter })));
}

function jsonBody(data) {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) };
}

function applyServerState(next) {
  if (!next) return;
  if (state && typeof next.update_seq === "number" && typeof state.update_seq === "number" && next.update_seq < state.update_seq) {
    return;
  }
  if (state && next.board_id && state.board_id && next.board_id < state.board_id) {
    return;
  }
  state = next;
  render();
}

// ----- rendering -------------------------------------------------------------

function render() {
  renderCapture();
  const loaded = state && state.loaded;
  $("empty").hidden = loaded;
  $("content").hidden = !loaded;
  if (!loaded) return;

  const current = state.current_suggestion || null;

  renderWarning();
  renderHero(current);
  renderPlayActions(current);
  renderBoard(current);
  renderStatus();
}

function renderWarning() {
  const box = $("warning");
  const parts = [];
  if (state.flagged.length) {
    const details = state.flagged.map(([r, c]) => {
      const cell = state.cells[r][c];
      const conf = cell.confidence == null ? "" : ` ${pct(cell.confidence)}`;
      return `row ${r + 1}, col ${c + 1} read as "${cell.letter}"${conf}`;
    });
    parts.push(
      `WARNING: ${state.flagged.length} tile${state.flagged.length > 1 ? "s" : ""} may be misread. ` +
      `Results may be wrong.<small>${details.join("; ")}. Tap the red tile${state.flagged.length > 1 ? "s" : ""} ` +
      `to fix or confirm the letter.</small>`
    );
  }
  if (state.solve_error) parts.push(state.solve_error);
  if (state.stale) {
    parts.push("Showing the last known board. Live capture is stale until a new screenshot succeeds.");
  }
  box.innerHTML = parts.join("<br>");
  box.hidden = parts.length === 0;
}

function renderCapture() {
  const cap = (state && state.capture) || {};
  const source = cap.source || {};
  const watching = !!cap.enabled;
  $("iphone-toggle").textContent = watching ? "Stop watching" : "Watch iPhone";
  $("stale-tag").hidden = !cap.stale;
  const dot = $("capture-dot");
  dot.className = "dot " + (cap.stale ? "stale" : cap.phase === "settling" ? "settling" : source.connected ? "on" : "off");
  let text = source.message || "iPhone: not watching";
  if (watching && cap.phase === "settling") text = "Waiting for the board to stop animating...";
  else if (watching && source.connected && cap.phase === "watching") text = source.message || "Watching iPhone";
  else if (watching && !source.connected) text = source.message || "iPhone disconnected — retrying";
  $("capture-text").textContent = text;
  const metrics = $("capture-metrics");
  const show = $("debug-check").checked && cap.timings;
  metrics.hidden = !show;
  if (show) {
    const t = cap.timings;
    metrics.textContent = `capture ${t.capture_ms} ms · recognize ${t.recognize_ms} ms · solve ${t.solve_ms} ms · total ${t.total_ms} ms`;
  }
}

function renderHero(current) {
  const hero = $("hero");
  hero.classList.toggle("unverified", !state.reliable);
  $("unverified-tag").hidden = state.reliable || !current;
  $("hero-label").textContent = "BEST WORD";
  $("hero-word").textContent = current ? current.word : state.solve ? "No words found" : "-";
  $("hero-length").textContent = current ? current.length : "-";
  $("hero-score").textContent = current && current.score != null ? current.score : "-";
  const tied = state.solve && state.suggestion_index === 1 && state.solve.best_count > 1;
  $("hero-tied").hidden = !tied;
  $("hero-tied").textContent = tied ? `(tied with ${state.solve.best_count - 1} other word${state.solve.best_count === 2 ? "" : "s"})` : "";
}

function renderPlayActions(current) {
  const hasWords = !!(state.solve && state.total_suggestions);
  $("reject-btn").disabled = !current;
  $("next-btn").disabled = !current || !(state.next_suggestions && state.next_suggestions.length);
  $("undo-btn").disabled = !state.can_undo;
  $("suggestion-count").textContent = hasWords
    ? `Suggestion ${state.suggestion_index} of ${state.total_suggestions}`
    : "No suggestions";
  const preview = $("next-preview");
  const next = state.next_suggestions || [];
  if (!next.length) {
    preview.textContent = "";
    return;
  }
  preview.innerHTML = '<span class="label">Next:</span>' + next.map((w) => `<span class="alt"></span>`).join("");
  [...preview.querySelectorAll(".alt")].forEach((el, i) => {
    const w = next[i];
    el.textContent = w.score != null ? `${w.word} (${w.score})` : w.word;
  });
}

function renderBoard(current) {
  const board = $("board");
  board.classList.toggle("editing", editing);
  $("edit-hint").hidden = !editing;

  const order = new Map(); // "r,c" -> step index (0-based)
  if (current) current.path.forEach(([r, c], i) => order.set(`${r},${c}`, i));
  const last = current ? current.path.length - 1 : -1;
  const role = (r, c) => {
    const i = order.get(`${r},${c}`);
    if (i === undefined) return current ? "unused" : "";
    return i === 0 ? "start" : i === last ? "end" : "mid";
  };

  const tiles = $("tile-layer");
  const labels = $("label-layer");
  tiles.replaceChildren();
  labels.replaceChildren();

  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const cell = state.cells[r][c];
      const cls = [role(r, c), cell.flagged ? "flagged" : ""].filter(Boolean);
      const pos = { left: `${c * 25}%`, top: `${r * 25}%` };

      const tile = document.createElement("div");
      tile.className = ["tile", ...cls].join(" ");
      tile.setAttribute("role", "button");
      tile.setAttribute("aria-label", `Row ${r + 1}, column ${c + 1}: ${cell.letter}`);
      tile.tabIndex = 0;
      Object.assign(tile.style, pos);
      tile.innerHTML = '<div class="tile-face"></div>';
      tile.addEventListener("click", () => onTileClick(r, c));
      tile.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onTileClick(r, c); }
      });
      tiles.appendChild(tile);

      const label = document.createElement("div");
      label.className = ["label", ...cls].join(" ");
      Object.assign(label.style, pos);
      label.textContent = cell.letter;
      const step = order.get(`${r},${c}`);
      if (step !== undefined) {
        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = step + 1;
        label.appendChild(badge);
      }
      if (cell.flagged) {
        const mark = document.createElement("span");
        mark.className = "flag-mark";
        mark.textContent = "CHECK";
        label.appendChild(mark);
      }
      labels.appendChild(label);
    }
  }
  drawPath(current ? current.path : []);
}

function drawPath(path) {
  const svg = $("path-layer");
  svg.replaceChildren();
  if (path.length === 0) return;

  const pts = path.map(([r, c]) => [(c + 0.5) * CELL, (r + 0.5) * CELL]);
  const pointsAttr = pts.map((p) => p.join(",")).join(" ");
  svg.appendChild(svgEl("polyline", { points: pointsAttr, fill: "none", stroke: "#fff", "stroke-width": 24, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: 0.9 }));
  svg.appendChild(svgEl("polyline", { points: pointsAttr, fill: "none", stroke: "var(--path)", "stroke-width": 14, "stroke-linejoin": "round", "stroke-linecap": "round" }));

  // Arrowhead halfway along each segment shows direction.
  for (let i = 0; i < pts.length - 1; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[i + 1];
    const angle = (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI;
    svg.appendChild(svgEl("path", {
      d: "M -13 -14 L 15 0 L -13 14 Z",
      fill: "#fff",
      stroke: "var(--path)",
      "stroke-width": 3,
      transform: `translate(${(x1 + x2) / 2} ${(y1 + y2) / 2}) rotate(${angle})`,
    }));
  }
  const [sx, sy] = pts[0];
  const [ex, ey] = pts[pts.length - 1];
  svg.appendChild(svgEl("circle", { cx: sx, cy: sy, r: 16, fill: "var(--start)", stroke: "#fff", "stroke-width": 4 }));
  svg.appendChild(svgEl("rect", { x: ex - 14, y: ey - 14, width: 28, height: 28, rx: 4, fill: "var(--end)", stroke: "#fff", "stroke-width": 4 }));
}

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function renderStatus() {
  const mini = $("mini-board");
  mini.replaceChildren();
  state.cells.forEach((row) => row.forEach((cell) => {
    const div = document.createElement("div");
    div.className = "mini-cell" + (cell.flagged ? " flagged" : cell.source === "manual" ? " manual" : "");
    const conf = cell.source === "manual" ? "manual" : cell.confidence == null ? "-" : pct(cell.confidence);
    div.innerHTML = `<span class="l"></span><span class="c">${conf}</span>`;
    div.querySelector(".l").textContent = cell.letter;
    div.title = cell.reason || (cell.runner_up ? `runner-up: ${cell.runner_up}` : "");
    mini.appendChild(div);
  }));

  const conf = $("stat-confidence");
  conf.textContent = state.min_confidence == null ? "n/a (typed by hand)" : pct(state.min_confidence);
  conf.className = state.flagged.length ? "bad" : "";
  $("stat-updated").textContent = formatTime(state.updated_at);
  $("stat-solve").textContent = state.solve ? `${state.solve.elapsed_ms.toFixed(2)} ms` : "not solved";
  $("stat-recognize").textContent = state.recognition ? `${state.recognition.elapsed_ms.toFixed(0)} ms` : "n/a";
  const timings = state.timings || (state.capture && state.capture.timings);
  $("stat-capture").textContent = timings ? `${timings.capture_ms} ms` : "n/a";
  $("stat-total-ms").textContent = timings ? `${timings.total_ms} ms` : "n/a";
  $("stat-total").textContent = state.solve ? state.solve.total : "-";
  let source = state.origin === "iphone" ? "Live iPhone" : state.origin === "screenshot" ? `Screenshot: ${state.recognition && state.recognition.image_name}` : "Typed by hand";
  if (state.stale) source += " (stale)";
  if (state.recognition && state.recognition.debug_dir) source += ` (debug: ${state.recognition.debug_dir})`;
  $("stat-source").textContent = source;
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function formatTime(iso) {
  const date = new Date(iso);
  const seconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  const ago = seconds < 60 ? `${seconds}s ago` : seconds < 3600 ? `${Math.floor(seconds / 60)} min ago` : `${Math.floor(seconds / 3600)} h ago`;
  return `${date.toLocaleTimeString()} (${ago})`;
}

// ----- manual correction -----------------------------------------------------

function onTileClick(r, c) {
  const cell = state.cells[r][c];
  if (!editing && !cell.flagged) return;
  editingCell = [r, c];
  $("cell-title").textContent = `Row ${r + 1}, column ${c + 1}`;
  const detail = cell.source === "manual"
    ? `Currently "${cell.letter}" (set by hand)`
    : `Read as "${cell.letter}"${cell.confidence == null ? "" : ` with ${pct(cell.confidence)} confidence`}` +
      (cell.reason ? ` - ${cell.reason}` : "");
  $("cell-detail").textContent = detail;
  $("cell-keep").textContent = `Keep "${cell.letter}"`;
  $("cell-keep").hidden = cell.letter === "?";
  const input = $("cell-input");
  input.value = "";
  $("cell-dialog").showModal();
  input.focus();
}

$("cell-input").addEventListener("input", (e) => {
  const letter = e.target.value.replace(/[^a-z]/gi, "").toUpperCase();
  e.target.value = letter;
  if (letter && editingCell) {
    $("cell-dialog").close();
    setCell(editingCell[0], editingCell[1], letter);
  }
});

$("cell-keep").addEventListener("click", () => {
  if (!editingCell) return;
  const [r, c] = editingCell;
  $("cell-dialog").close();
  setCell(r, c, state.cells[r][c].letter);
});

$("edit-toggle").addEventListener("click", () => {
  editing = !editing;
  $("edit-toggle").setAttribute("aria-pressed", String(editing));
  $("edit-toggle").textContent = editing ? "Done editing" : "Edit letters";
  if (state && state.loaded) render();
});

// ----- inputs ----------------------------------------------------------------

$("iphone-toggle").addEventListener("click", () => {
  const watching = state && state.capture && state.capture.enabled;
  const path = watching ? "/api/capture/stop" : "/api/capture/start";
  const options = watching ? { method: "POST" } : jsonBody({ debug: $("debug-check").checked });
  api(path, options).then(applyServerState).catch(() => {});
});

$("debug-check").addEventListener("change", () => {
  if (state && state.capture) renderCapture();
});

$("file-input").addEventListener("change", (e) => {
  uploadScreenshot(e.target.files[0]);
  e.target.value = "";
});

$("type-toggle").addEventListener("click", () => {
  const form = $("type-form");
  form.hidden = !form.hidden;
  if (!form.hidden) {
    $("type-input").value = state && state.loaded ? state.board.map((r) => r.join("")).join(" ") : "";
    $("type-input").focus();
  }
});

$("type-form").addEventListener("submit", (e) => {
  e.preventDefault();
  submitBoard($("type-input").value).then(() => { $("type-form").hidden = true; }).catch(() => {});
});

$("reject-btn").addEventListener("click", () => {
  apiLocal("/api/reject-current-word").then(applyServerState).catch(() => {});
});
$("next-btn").addEventListener("click", () => {
  apiLocal("/api/next-word").then(applyServerState).catch(() => {});
});
$("undo-btn").addEventListener("click", () => {
  apiLocal("/api/undo-rejection").then(applyServerState).catch(() => {});
});

function typingInField() {
  const tag = (document.activeElement && document.activeElement.tagName) || "";
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

document.addEventListener("keydown", (e) => {
  if (typingInField() || $("cell-dialog").open) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key === "x") { e.preventDefault(); $("reject-btn").click(); }
  else if (key === "n") { e.preventDefault(); $("next-btn").click(); }
  else if (key === "z") { e.preventDefault(); $("undo-btn").click(); }
});

let dragDepth = 0;
document.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; $("drop-overlay").hidden = false; });
document.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop-overlay").hidden = true; } });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("drop-overlay").hidden = true;
  const file = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
  if (file) uploadScreenshot(file);
});

document.addEventListener("paste", (e) => {
  if (e.target.tagName === "INPUT") return;
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) uploadScreenshot(item.getAsFile());
});

function formatApiError(body, status) {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  }
  return `Request failed (${status})`;
}

function showError(message) {
  $("error").hidden = !message;
  $("error").textContent = message || "";
}

function showBusy(on) {
  $("busy").hidden = !on;
}

// Keep "x min ago" current.
setInterval(() => { if (state && state.loaded) $("stat-updated").textContent = formatTime(state.updated_at); }, 5000);

api("/api/state").then(applyServerState).catch(() => {});

setInterval(() => {
  if (!(state && state.capture && state.capture.enabled)) return;
  fetch("/api/state").then((r) => r.json()).then(applyServerState).catch(() => {});
}, 500);
