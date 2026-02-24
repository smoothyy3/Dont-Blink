"""
Lightweight local web UI for reviewing contribution frames.

Serves a single-page app from Python's built-in http.server.
Zero external dependencies — the HTML/CSS/JS is self-contained.
"""
import json
import logging
import os
import socket
import threading
import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML template (self-contained single-page app)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dont-Blink — Frame Review</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d37;
    --text: #e4e4e7; --muted: #71717a;
    --green: #22c55e; --red: #ef4444; --gray: #6b7280; --blue: #3b82f6;
    --amber: #f59e0b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 20px;
  }
  h1 { font-size: 1.3rem; font-weight: 600; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 16px; }

  .progress-bar {
    width: 100%; max-width: 700px; height: 6px;
    background: var(--border); border-radius: 3px; margin-bottom: 20px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: var(--blue); border-radius: 3px;
    transition: width 0.3s ease;
  }

  .frame-container {
    position: relative; max-width: 700px; width: 100%;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden; margin-bottom: 16px;
  }
  .frame-container img {
    display: block; width: 100%; height: auto;
  }
  .frame-info {
    padding: 10px 14px; display: flex; justify-content: space-between;
    font-size: 0.8rem; color: var(--muted); border-top: 1px solid var(--border);
  }

  .actions {
    display: flex; gap: 12px; max-width: 700px; width: 100%;
    margin-bottom: 8px;
  }
  .actions button {
    flex: 1; padding: 14px 0; border: none; border-radius: 10px;
    font-size: 1rem; font-weight: 600; cursor: pointer;
    transition: transform 0.1s, opacity 0.15s;
    color: #fff;
  }
  .actions button:hover { opacity: 0.9; }
  .actions button:active { transform: scale(0.97); }
  .btn-confirm { background: var(--green); }
  .btn-reject { background: var(--red); }
  .btn-no-head { background: var(--gray); }

  .nav-row {
    display: flex; justify-content: space-between; align-items: center;
    max-width: 700px; width: 100%; margin-bottom: 20px;
  }
  .btn-back {
    background: none; border: 1px solid var(--border); color: var(--muted);
    padding: 8px 18px; border-radius: 8px; font-size: 0.85rem; cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
  }
  .btn-back:hover { border-color: var(--text); color: var(--text); }
  .btn-back:disabled { opacity: 0.3; cursor: default; }

  .keyboard-hint {
    font-size: 0.75rem; color: var(--muted);
  }
  kbd {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 4px; padding: 2px 6px; font-family: inherit;
  }

  .screen { max-width: 700px; width: 100%; }
  .done-screen { text-align: center; padding: 40px 20px; }
  .done-screen h2 { font-size: 1.5rem; margin-bottom: 12px; }
  .done-screen .stats { color: var(--muted); margin-bottom: 24px; line-height: 1.8; }
  .done-screen button {
    background: var(--blue); color: #fff; border: none;
    padding: 12px 32px; border-radius: 8px; font-size: 1rem;
    font-weight: 600; cursor: pointer;
  }
  .done-screen button:hover { opacity: 0.9; }

  /* Consent screen */
  .consent-screen { max-width: 600px; width: 100%; }
  .consent-screen h2 { font-size: 1.4rem; margin-bottom: 16px; }
  .consent-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;
    line-height: 1.7; font-size: 0.9rem;
  }
  .consent-box ul { margin: 10px 0 10px 20px; }
  .consent-box li { margin-bottom: 4px; }
  .consent-box strong { color: var(--amber); }
  .consent-check {
    display: flex; align-items: flex-start; gap: 10px;
    margin-bottom: 20px; font-size: 0.9rem; cursor: pointer;
  }
  .consent-check input { margin-top: 4px; accent-color: var(--blue); width: 18px; height: 18px; }
  .btn-agree {
    background: var(--blue); color: #fff; border: none;
    padding: 14px 0; border-radius: 10px; font-size: 1rem;
    font-weight: 600; cursor: pointer; width: 100%;
    transition: opacity 0.15s;
  }
  .btn-agree:disabled { opacity: 0.35; cursor: default; }
  .btn-agree:not(:disabled):hover { opacity: 0.9; }

  .hidden { display: none; }
</style>
</head>
<body>

<h1>Dont-Blink — Frame Review</h1>
<p class="subtitle" id="counter">Data Contribution</p>

<div class="progress-bar"><div class="progress-fill" id="progress"></div></div>

<!-- Consent screen -->
<div id="consent-screen" class="consent-screen">
  <h2>Before you start</h2>
  <div class="consent-box">
    <p>You're about to review frames extracted from your video. Here's what happens with the data:</p>
    <ul>
      <li><strong>Only cropped frames</strong> (not the full video) will be included.</li>
      <li>Frames are labeled with bounding boxes and your printer/camera metadata.</li>
      <li>The contribution is licensed under <strong>CC0 (public domain)</strong>, so it can be used freely to improve the model for everyone.</li>
      <li><strong>No personal information</strong> is collected — only printer frames and the metadata you provide.</li>
    </ul>
    <p>You can reject any frame you don't want to share during the review.</p>
  </div>
  <label class="consent-check">
    <input type="checkbox" id="consent-cb" onchange="updateConsent()">
    <span>I understand and agree that my contribution will be released under CC0 (public domain) for open use.</span>
  </label>
  <button class="btn-agree" id="consent-btn" disabled onclick="startReview()">Start Review</button>
</div>

<!-- Review area -->
<div id="review-area" class="screen hidden">
  <div class="frame-container">
    <img id="frame-img" src="" alt="Frame preview">
    <div class="frame-info">
      <span id="info-time"></span>
      <span id="info-conf"></span>
    </div>
  </div>
  <div class="actions">
    <button class="btn-confirm" onclick="label('confirm')">&#10003; Confirm</button>
    <button class="btn-reject" onclick="label('reject')">&#10007; Reject</button>
    <button class="btn-no-head" onclick="label('no_printhead')">No Printhead</button>
  </div>
  <div class="nav-row">
    <button class="btn-back" id="back-btn" onclick="goBack()" disabled>&#8592; Back</button>
    <span class="keyboard-hint">
      <kbd>1</kbd> Confirm &nbsp; <kbd>2</kbd> Reject &nbsp; <kbd>3</kbd> No Printhead &nbsp; <kbd>&larr;</kbd> Back
    </span>
  </div>
</div>

<!-- Done screen -->
<div id="done-screen" class="done-screen hidden">
  <h2>Review Complete</h2>
  <div class="stats" id="done-stats"></div>
  <div style="display:flex; gap:12px; justify-content:center;">
    <button class="btn-back" onclick="goBack()" style="padding:12px 24px;">&#8592; Go Back</button>
    <button onclick="submitAll()">Save &amp; Finish</button>
  </div>
</div>

<script>
const items = /*ITEMS_JSON*/;
const labels = {};
let current = 0;
let consentGiven = false;

function updateConsent() {
  consentGiven = document.getElementById('consent-cb').checked;
  document.getElementById('consent-btn').disabled = !consentGiven;
}

function startReview() {
  if (!consentGiven) return;
  document.getElementById('consent-screen').classList.add('hidden');
  renderFrame();
}

function renderFrame() {
  const backBtn = document.getElementById('back-btn');

  if (current >= items.length) {
    document.getElementById('review-area').classList.add('hidden');
    document.getElementById('done-screen').classList.remove('hidden');
    const c = Object.values(labels).filter(v => v === 'confirm').length;
    const r = Object.values(labels).filter(v => v === 'reject').length;
    const n = Object.values(labels).filter(v => v === 'no_printhead').length;
    document.getElementById('done-stats').innerHTML =
      `${c} confirmed &middot; ${r} rejected &middot; ${n} no-printhead<br>` +
      `${c + n} frames will be included in the contribution bundle.`;
    document.getElementById('progress').style.width = '100%';
    document.getElementById('counter').textContent =
      `Review complete — ${items.length} frames`;
    return;
  }

  document.getElementById('review-area').classList.remove('hidden');
  document.getElementById('done-screen').classList.add('hidden');
  backBtn.disabled = (current === 0);

  const item = items[current];
  document.getElementById('counter').textContent =
    `Frame ${current + 1} of ${items.length}`;
  document.getElementById('progress').style.width =
    `${(current / items.length) * 100}%`;
  document.getElementById('frame-img').src =
    `/previews/${item.filename}?t=${Date.now()}`;
  document.getElementById('info-time').textContent =
    `t = ${item.timestamp_s.toFixed(1)}s  (frame #${item.frame_idx})`;
  document.getElementById('info-conf').textContent =
    `confidence: ${item.p_present.toFixed(3)}`;
}

function label(action) {
  labels[items[current].index] = action;
  current++;
  renderFrame();
}

function goBack() {
  if (current > 0) {
    current--;
    delete labels[items[current].index];
    renderFrame();
  }
}

async function submitAll() {
  try {
    const resp = await fetch('/api/labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(labels),
    });
    if (resp.ok) {
      document.getElementById('done-screen').innerHTML =
        '<h2>Saved!</h2><p class="stats">You can close this tab.<br>Return to the terminal to continue.</p>';
    }
  } catch (e) {
    alert('Failed to save: ' + e.message);
  }
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') { goBack(); return; }
  if (current >= items.length) return;
  if (e.key === '1') label('confirm');
  else if (e.key === '2') label('reject');
  else if (e.key === '3') label('no_printhead');
});

// Don't auto-start review; wait for consent
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class _LabelingHandler(SimpleHTTPRequestHandler):
    """Serves the labeling page, preview images, and the label-submission API."""

    def __init__(self, *args, work_dir: Path, review_items: list,
                 result_holder: dict, shutdown_event: threading.Event, **kwargs):
        self._work_dir = work_dir
        self._review_items = review_items
        self._result = result_holder
        self._shutdown_event = shutdown_event
        super().__init__(*args, directory=str(work_dir), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_html()
        elif parsed.path.startswith("/previews/"):
            self._serve_file(self._work_dir / "previews", parsed.path.split("/previews/")[1])
        elif parsed.path.startswith("/frames/"):
            self._serve_file(self._work_dir / "frames", parsed.path.split("/frames/")[1])
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/labels":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                self._result["labels"] = {int(k): v for k, v in data.items()}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
                # Signal the main thread that we're done
                self._shutdown_event.set()
            except Exception as e:
                self.send_error(400, str(e))
        else:
            self.send_error(404)

    def _serve_html(self):
        items_json = json.dumps(self._review_items)
        html = _HTML_TEMPLATE.replace("/*ITEMS_JSON*/", items_json)
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, base_dir: Path, filename: str):
        # Strip query string
        filename = filename.split("?")[0]
        filepath = base_dir / filename
        if not filepath.exists():
            self.send_error(404)
            return
        data = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass  # Silence request logging


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def launch_review_ui(
    work_dir: Path,
    review_items: List[Dict[str, Any]],
) -> Dict[int, str]:
    """
    Start local web server, open browser, wait for user to finish labeling.

    Returns:
        Dict mapping frame index -> "confirm" | "reject" | "no_printhead"
    """
    port = _find_free_port()
    result: Dict[str, Any] = {}
    shutdown_event = threading.Event()

    handler_cls = partial(
        _LabelingHandler,
        work_dir=work_dir,
        review_items=review_items,
        result_holder=result,
        shutdown_event=shutdown_event,
    )

    server = HTTPServer(("127.0.0.1", port), handler_cls)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://127.0.0.1:{port}"
    logger.info(f"Labeling UI running at {url}")
    print(f"\n  Opening labeling UI in your browser: {url}")
    print("  Label each frame, then click 'Save & Finish'.")
    print("  (Press Ctrl+C in this terminal to abort)\n")

    webbrowser.open(url)

    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=0.5)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        return {}
    finally:
        server.shutdown()

    return result.get("labels", {})
