"""Live view of training runs for a browser (apps/dashboard.py serves it).

Everything shown is read from files the pipeline already writes: ``train_log.csv`` (loss),
``step_*.pt`` (checkpoints), ``val_step_*.json`` and ``selection.json`` (selection on val) in
``runs/<task>/<method>_seed<k>/``, and the logs being written in ``runs/logs/``. Only the
current version of each skill is shown (``current_runs``). The only thing the
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
from urllib.parse import parse_qs, urlsplit

import yaml

from dexvision import REPO_ROOT
from dexvision.tasks import TASKS
from dexvision.tasks.spec import load_task_spec

PAGE = Path(__file__).with_name("dashboard.html")
METHOD_CONFIG = REPO_ROOT / "configs" / "learning" / "method.yaml"
_VAL_CACHE: dict[tuple[Path, float], dict] = {}   # a val report's totals, by file and mtime


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
        key = (path, path.stat().st_mtime)
        if key not in _VAL_CACHE:
            try:
                _VAL_CACHE[key] = json.loads(path.read_text(encoding="utf-8"))["overall"]
            except ValueError:  # being written right now; it is complete on the next poll
                continue
        val.append({"step": int(path.stem.rsplit("_", 1)[1]), **_VAL_CACHE[key]})
    selection = run / "selection.json"
    selected = (json.loads(selection.read_text(encoding="utf-8"))["selected"]["step"]
                if selection.is_file() else None)
    method, _, seed = run.name.rpartition("_seed")
    return {"name": run.name, "method": method, "seed": seed, "total_steps": total_steps,
            "loss": rows, "steps_per_s": rate, "val": val, "selected": selected,
            "checkpoints": sorted(int(p.stem.rsplit("_", 1)[1]) for p in run.glob("step_*.pt")),
            "updated": (run / "train_log.csv").stat().st_mtime}


def recent_logs(logs: Path, lines: int = 14, within_s: float = 300.0, most: int = 4) -> list[dict]:
    """The tails of the log files written in the last ``within_s`` seconds, the newest first:
    the pipeline that is running now. Older logs (finished or failed earlier) are left out."""
    now = time.time()
    files = sorted((p for p in logs.glob("*.log") if p.is_file()),
                   key=lambda p: p.stat().st_mtime, reverse=True) if logs.is_dir() else []
    tails = []
    for path in [p for p in files if now - p.stat().st_mtime < within_s][:most]:
        with path.open("rb") as fh:
            fh.seek(max(0, path.stat().st_size - 20000))
            text = fh.read().decode("utf-8", errors="replace").replace("\r", "\n")
        tails.append({"name": path.name, "written": path.stat().st_mtime,
                      "lines": [x for x in text.splitlines() if x.strip()][-lines:]})
    return tails


def current_runs(runs_root: Path) -> dict[str, tuple[str, list[Path]]]:
    """What the page shows: per skill, the task variant trained most recently (pick ->
    pick_vision_varied, not the earlier pick and pick_vision), and per method its newest run
    (the latest seed). {skill: (task, [run directory, ...])}; older runs stay on disk unshown."""
    newest: dict[str, tuple[float, str]] = {}
    logs = sorted(runs_root.glob("*/*/train_log.csv"))
    for log in logs:
        task = log.parent.parent.name
        skills = [k for k in TASKS if task == k or task.startswith(k + "_")]
        if skills:
            skill = max(skills, key=len)
            newest[skill] = max(newest.get(skill, (0.0, "")), (log.stat().st_mtime, task))
    current = {}
    for skill, (_, task) in sorted(newest.items()):
        by_method: dict[str, Path] = {}
        for log in logs:
            if log.parent.parent.name == task:
                method = log.parent.name.rpartition("_seed")[0]
                if method not in by_method or log.stat().st_mtime > \
                        (by_method[method] / "train_log.csv").stat().st_mtime:
                    by_method[method] = log.parent
        current[skill] = (task, [by_method[m] for m in sorted(by_method)])
    return current


def collect(runs_root: Path, videos: EpisodeVideos | None = None) -> dict:
    """Everything the page shows: {"skills": {skill: {"task", "runs": [run, ...]}}, "logs",
    "time", ...}. Each run's "video" is the recorder's status, or None when recording is off."""
    training = yaml.safe_load(METHOD_CONFIG.read_text(encoding="utf-8"))["training"]
    skills = {}
    for skill, (task, run_dirs) in current_runs(runs_root).items():
        runs = [read_run(run, int(training["steps"])) for run in run_dirs]
        for run, directory in zip(runs, run_dirs, strict=True):
            run["video"] = videos.status(directory) if videos else None
        skills[skill] = {"task": task, "runs": runs}
    return {"skills": skills, "logs": recent_logs(runs_root / "logs"), "time": time.time(),
            "videos": videos is not None, "checkpoint_every": int(training["checkpoint_every"])}


class EpisodeVideos:
    """Records one val episode of a run's newest checkpoint, in a background thread.

    Recorded for the shown runs (``current_runs``) that are training now or belong to the
    task the page is showing
    (``want``), one episode at a time. The first val seed every time, so successive
    checkpoints are seen on the same episode. A camera task gets the policy's own camera frames
    with the encoder's keypoints drawn (apps/keypoint_video.py); a state task gets the viewer
    video (apps/evaluate.py --videos).
    """

    def __init__(self, runs_root: Path, poll_s: float = 5.0, live_s: float = 600.0) -> None:
        self.runs_root, self.poll_s, self.live_s = runs_root, poll_s, live_s
        self.out = runs_root / "dashboard"
        self.wanted: dict[str, float] = {}      # tasks open in a browser: when last asked
        self.busy: Path | None = None           # the checkpoint being recorded now
        self.failed: dict[Path, str] = {}

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def want(self, task: str) -> None:
        self.wanted[task] = time.time()

    def _directory(self, checkpoint: Path) -> Path:
        return self.out / checkpoint.parent.parent.name / checkpoint.parent.name / checkpoint.stem

    def status(self, run: Path) -> dict:
        """What the page says about ``run``'s episode: "state" is "none" (no checkpoint yet),
        "ready", "recording", "queued" or "failed"; "shown" is the newest recorded episode
        (maybe of an earlier checkpoint) or None; "newest_step" is the newest checkpoint's."""
        checkpoints = sorted(run.glob("step_*.pt"))
        shown = None
        for checkpoint in reversed(checkpoints):
            files = sorted(self._directory(checkpoint).glob("*.mp4"))
            if files:
                relative = files[0].relative_to(self.out).as_posix()
                shown = {"step": _step(checkpoint), "url": f"/video/{relative}",
                         "outcome": outcome(files[0].stem)}
                break
        if not checkpoints:
            return {"state": "none", "shown": None, "newest_step": None}
        newest = checkpoints[-1]
        state = ("ready" if shown and shown["step"] == _step(newest) else
                 "recording" if self.busy == newest else
                 "failed" if newest in self.failed else "queued")
        return {"state": state, "shown": shown, "newest_step": _step(newest),
                "error": self.failed.get(newest)}

    def _targets(self) -> list[Path]:
        """Newest checkpoints still to record: the shown task's runs first, then live runs."""
        now, first, rest = time.time(), [], []
        shown = [run for _, runs in current_runs(self.runs_root).values() for run in runs]
        for log in (run / "train_log.csv" for run in shown):
            checkpoints = sorted(log.parent.glob("step_*.pt"))
            if not checkpoints or checkpoints[-1] in self.failed \
                    or list(self._directory(checkpoints[-1]).glob("*.mp4")):
                continue
            if now - self.wanted.get(log.parent.parent.name, 0.0) < 60.0:
                first.append(checkpoints[-1])
            elif now - log.stat().st_mtime < self.live_s:
                rest.append(checkpoints[-1])
        return first + rest

    def _loop(self) -> None:
        while True:
            targets = self._targets()
            if targets:
                self.record(targets[0])
            else:
                time.sleep(self.poll_s)

    def record(self, checkpoint: Path) -> None:
        task = checkpoint.parent.parent.name
        spec = load_task_spec(task)
        out = self._directory(checkpoint)
        out.mkdir(parents=True, exist_ok=True)
        app = [sys.executable, "-m"]
        if spec.cameras:
            app += ["dexvision.apps.keypoint_video", "--task", task, "--checkpoint",
                    str(checkpoint), "--seeds", str(spec.splits["val"][0]), "--out-dir", str(out)]
        else:
            app += ["dexvision.apps.evaluate", "--task", task, "--checkpoint", str(checkpoint),
                    "--episodes", "1", "--videos", "1", "--video-dir", str(out),
                    "--out", str(out / "report.json")]
        self.busy = checkpoint
        try:
            done = subprocess.run(app, cwd=REPO_ROOT, capture_output=True, text=True,
                                  env={**os.environ, "OMP_NUM_THREADS": "1"}, check=False)
        finally:
            self.busy = None
        if done.returncode or not list(out.glob("*.mp4")):
            self.failed[checkpoint] = (done.stderr.strip().splitlines() or ["no video written"])[-1]
            print(f"video of {checkpoint} failed:\n{done.stderr[-2000:]}", flush=True)


def _step(checkpoint: Path) -> int:
    return int(checkpoint.stem.rsplit("_", 1)[1])


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
            task = parse_qs(urlsplit(self.path).query).get("task", [""])[0]
            if self.videos and task in {t for t, _ in current_runs(self.runs_root).values()}:
                self.videos.want(task)      # record this task's newest checkpoints on demand
            self._send(json.dumps(collect(self.runs_root, self.videos)).encode(),
                       "application/json")
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
