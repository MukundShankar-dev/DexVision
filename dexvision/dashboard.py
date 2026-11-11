"""Live view of training runs for a browser (apps/dashboard.py serves it).

Everything shown is read from files the pipeline already writes: ``train_log.csv`` (loss),
``step_*.pt`` (checkpoints), ``val_step_*.json`` and ``selection.json`` (selection on val) in
``runs/<task>/<method>_seed<k>/``, and the logs being written in ``runs/logs/``. The only thing the
dashboard adds is one recorded val episode of each run's newest checkpoint, under
``runs/dashboard/``; it never writes into a run directory or ``results/``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from dexvision import REPO_ROOT
from dexvision.tasks.spec import load_task_spec

PAGE = Path(__file__).with_name("dashboard.html")
METHOD_CONFIG = REPO_ROOT / "configs" / "learning" / "method.yaml"


def read_run(run: Path, total_steps: int) -> dict:
    """One training run's progress. ``loss`` rows are [step, mean loss, seconds]."""
    with (run / "train_log.csv").open(encoding="utf-8") as fh:
        rows = [[int(r["step"]), float(r["loss"]), float(r["seconds"])]
                for r in csv.DictReader(fh) if r["seconds"]]
    recent = rows[-11:]
    rate = None
    if len(recent) > 1 and recent[-1][2] > recent[0][2]:
        rate = (recent[-1][0] - recent[0][0]) / (recent[-1][2] - recent[0][2])
    val = []
    for path in sorted(run.glob("val_step_*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:      # being written right now; it is complete on the next poll
            continue
        val.append({"step": int(path.stem.rsplit("_", 1)[1]), **report["overall"]})
    selection = run / "selection.json"
    selected = (json.loads(selection.read_text(encoding="utf-8"))["selected"]["step"]
                if selection.is_file() else None)
    method, _, seed = run.name.rpartition("_seed")
    return {"name": run.name, "method": method, "seed": seed, "total_steps": total_steps,
            "loss": rows, "steps_per_s": rate, "val": val, "selected": selected,
            "checkpoints": sorted(int(p.stem.rsplit("_", 1)[1]) for p in run.glob("step_*.pt")),
            "updated": (run / "train_log.csv").stat().st_mtime}


def recent_logs(logs: Path, lines: int = 14, within_s: float = 900.0, most: int = 4) -> list[dict]:
    """The tails of the log files still being written (the newest first), or of the newest one
    when none was written in the last ``within_s`` seconds."""
    files = sorted((p for p in logs.glob("*.log") if p.is_file()),
                   key=lambda p: p.stat().st_mtime, reverse=True) if logs.is_dir() else []
    live = [p for p in files if time.time() - p.stat().st_mtime < within_s][:most] or files[:1]
    tails = []
    for path in live:
        with path.open("rb") as fh:
            fh.seek(max(0, path.stat().st_size - 20000))
            text = fh.read().decode("utf-8", errors="replace").replace("\r", "\n")
        tails.append({"name": path.name,
                      "lines": [x for x in text.splitlines() if x.strip()][-lines:]})
    return tails


def collect(runs_root: Path, videos: dict | None = None) -> dict:
    """Everything the page shows: {"tasks": {task: [run, ...]}, "logs": [...], "time": ...}.
    ``videos``: the recorded episodes by (task, run), or None when recording is off."""
    total = int(yaml.safe_load(METHOD_CONFIG.read_text(encoding="utf-8"))["training"]["steps"])
    tasks: dict[str, list[dict]] = {}
    for log in sorted(runs_root.glob("*/*/train_log.csv")):
        run = read_run(log.parent, total)
        run["video"] = (videos or {}).get((log.parent.parent.name, run["name"]))
        tasks.setdefault(log.parent.parent.name, []).append(run)
    return {"tasks": tasks, "logs": recent_logs(runs_root / "logs"), "time": time.time(),
            "videos": videos is not None}


class EpisodeVideos:
    """Records one val episode of each run's newest checkpoint, in a background thread.

    The first val seed every time, so successive checkpoints are seen on the same episode. A
    camera task gets the policy's own camera frames with the encoder's keypoints drawn
    (apps/keypoint_video.py); a state task gets the viewer video (apps/evaluate.py --videos).
    """

    def __init__(self, runs_root: Path, poll_s: float = 20.0) -> None:
        self.runs_root, self.poll_s = runs_root, poll_s
        self.out = runs_root / "dashboard"
        self.latest: dict[tuple[str, str], dict] = {}
        self._failed: set[Path] = set()

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            for log in sorted(self.runs_root.glob("*/*/train_log.csv")):
                checkpoints = sorted(log.parent.glob("step_*.pt"))
                if checkpoints and checkpoints[-1] not in self._failed:
                    self.record(checkpoints[-1])
            time.sleep(self.poll_s)

    def record(self, checkpoint: Path) -> None:
        task, run = checkpoint.parent.parent.name, checkpoint.parent.name
        out = self.out / task / run / checkpoint.stem
        if not list(out.glob("*.mp4")):
            spec = load_task_spec(task)
            out.mkdir(parents=True, exist_ok=True)
            app = [sys.executable, "-m"]
            if spec.cameras:
                app += ["dexvision.apps.keypoint_video", "--task", task, "--checkpoint",
                        str(checkpoint), "--seeds", str(spec.splits["val"][0]),
                        "--out-dir", str(out)]
            else:
                app += ["dexvision.apps.evaluate", "--task", task, "--checkpoint",
                        str(checkpoint), "--episodes", "1", "--videos", "1", "--video-dir",
                        str(out), "--out", str(out / "report.json")]
            done = subprocess.run(app, cwd=REPO_ROOT, capture_output=True, text=True,
                                  env={**os.environ, "OMP_NUM_THREADS": "1"}, check=False)
            if done.returncode:
                self._failed.add(checkpoint)
                print(f"video of {checkpoint} failed:\n{done.stderr[-2000:]}", flush=True)
                return
        files = sorted(out.glob("*.mp4"))
        if files:
            self.latest[task, run] = {
                "step": int(checkpoint.stem.rsplit("_", 1)[1]), "outcome": outcome(files[0].stem),
                "url": f"/video/{task}/{run}/{checkpoint.stem}/{files[0].name}"}


def outcome(stem: str) -> str:
    """The outcome at the end of a video name: the words after the seed (the last all-digit
    field), e.g. ``keypoints_pick_vision_100000_no_contact`` -> "no contact"."""
    fields = stem.split("_")
    seed_at = max(i for i, f in enumerate(fields) if f.isdigit())
    return " ".join(fields[seed_at + 1:])


class Handler(BaseHTTPRequestHandler):
    runs_root: Path
    videos: EpisodeVideos | None

    def log_message(self, *args) -> None:   # quiet: the page polls every few seconds
        pass

    def _send(self, body: bytes, content_type: str, status=HTTPStatus.OK, extra=()) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            state = collect(self.runs_root, self.videos.latest if self.videos else None)
            self._send(json.dumps(state).encode(), "application/json")
        elif path.startswith("/video/") and self.videos:
            self._video(path.removeprefix("/video/"))
        else:
            self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

    def _video(self, relative: str) -> None:
        root = self.videos.out.resolve()
        file = (root / relative).resolve()
        if not file.is_relative_to(root) or file.suffix != ".mp4" or not file.is_file():
            self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            return
        data = file.read_bytes()
        span = self.headers.get("Range", "")
        if span.startswith("bytes="):       # browsers seek in a video with range requests
            first, _, last = span.removeprefix("bytes=").partition("-")
            start = int(first or 0)
            stop = min(int(last), len(data) - 1) if last else len(data) - 1
            self._send(data[start:stop + 1], "video/mp4", HTTPStatus.PARTIAL_CONTENT,
                       [("Content-Range", f"bytes {start}-{stop}/{len(data)}"),
                        ("Accept-Ranges", "bytes")])
        else:
            self._send(data, "video/mp4", extra=[("Accept-Ranges", "bytes")])


def make_server(runs_root: Path, host: str, port: int, videos: bool) -> ThreadingHTTPServer:
    recorder = EpisodeVideos(runs_root) if videos else None
    handler = type("DashboardHandler", (Handler,), {"runs_root": runs_root, "videos": recorder})
    server = ThreadingHTTPServer((host, port), handler)
    if recorder:
        recorder.start()
    return server
