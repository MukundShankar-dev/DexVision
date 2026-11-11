"""Checkpoint selection on val (AGENTS.md section 2, step 5).

Rule, fixed before any pick training: every checkpoint of a run is evaluated on the same first
n val seeds; the one with the most successes is selected; ties go to the later step.
Choosing the best of several checkpoints on one seed set biases its val rate upward, which is
why the gate is judged on a single test run.

Each episode depends only on its seed and the checkpoint, so when a checkpoint is evaluated
(after training, or while later steps are still training) does not change its result.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from dexvision.eval.evaluator import evaluate
from dexvision.learning.policy import load_learned_policy
from dexvision.policies.base import Policy
from dexvision.sim.workcell import Workcell
from dexvision.tasks import make_task


def select(candidates: Sequence[dict]) -> dict:
    """``candidates``: dicts with "successes" and "step" -> the selected one."""
    if not candidates:
        raise ValueError("No checkpoints to select from")
    return max(candidates, key=lambda c: (c["successes"], c["step"]))


def _policy_in_worker(checkpoint: str, task_name: str, workcell: Workcell) -> Policy:
    return load_learned_policy(Path(checkpoint), make_task(task_name).spec)[0]


def evaluate_checkpoint(checkpoint: Path, task_name: str, n_episodes: int, out_dir: Path,
                        workers: int) -> dict:
    """Evaluate one checkpoint on the first ``n_episodes`` val seeds, ``workers`` episodes at a
    time. The report goes to ``out_dir``; returns its row of the selection table."""
    task = make_task(task_name)
    policy, info = load_learned_policy(checkpoint, task.spec)

    def progress(episodes, total):
        if len(episodes) % 50 == 0:
            print(f"  {checkpoint.parent.name}/{checkpoint.stem}: {len(episodes)}/{total} "
                  f"episodes, {sum(e.success for e in episodes)} successes", flush=True)

    report = evaluate(Workcell(), task, policy, "val", n_episodes, policy_info=info,
                      on_episode=progress, workers=workers,
                      policy_factory=partial(_policy_in_worker, str(checkpoint), task_name))
    out = out_dir / f"val_{checkpoint.stem}.json"
    out.write_text(json.dumps(report.to_dict(), indent=1) + "\n", encoding="utf-8")
    o = report.overall
    return {"checkpoint": info["checkpoint"], "step": info["step"], "successes": o.successes,
            "n": o.n, "ci95": list(o.ci95), "report": out.name}


def evaluate_checkpoints(checkpoints: Sequence[Path], task_name: str, n_episodes: int,
                         out_dir: Path, workers: int) -> list[dict]:
    """The selection table of finished checkpoints, in step order. Reports go to ``out_dir``."""
    rows = [evaluate_checkpoint(Path(c), task_name, n_episodes, out_dir, workers)
            for c in checkpoints]
    return sorted(rows, key=lambda r: r["step"])


def follow(runs: Sequence[Path], last_step: int, evaluate_one: Callable[[Path], dict],
           poll_s: float = 30.0) -> dict[Path, list[dict]]:
    """Evaluate the checkpoints of runs that are still training, as they are saved.

    Checkpoints are taken oldest first, one at a time (``evaluate_one`` uses the cores), until
    every run's ``step_<last_step>.pt`` has been evaluated. Returns each run's selection table in
    step order. A run directory need not exist yet. The trainer must write checkpoints
    atomically (apps/train.py renames a finished file into place)."""
    tables: dict[Path, list[dict]] = {Path(run): [] for run in runs}
    done: set[Path] = set()
    final = {run / f"step_{last_step:06d}.pt" for run in tables}
    while not final <= done:
        waiting = sorted((p for run in tables for p in run.glob("step_*.pt") if p not in done),
                         key=lambda p: (p.stat().st_mtime, p))
        if not waiting:
            time.sleep(poll_s)
            continue
        tables[waiting[0].parent].append(evaluate_one(waiting[0]))
        done.add(waiting[0])
    return {run: sorted(rows, key=lambda r: r["step"]) for run, rows in tables.items()}
