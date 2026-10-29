"""Checkpoint selection on val (AGENTS.md section 2, step 5).

Rule, fixed before any pick training: every checkpoint of a run is evaluated on the same first
n val seeds; the one with the most successes is selected; ties go to the later step.
Choosing the best of several checkpoints on one seed set biases its val rate upward, which is
why the gate is judged on a single test run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def select(candidates: Sequence[dict]) -> dict:
    """``candidates``: dicts with "successes" and "step" -> the selected one."""
    if not candidates:
        raise ValueError("No checkpoints to select from")
    return max(candidates, key=lambda c: (c["successes"], c["step"]))


def _evaluate_one(job: tuple[str, str, str, int, str]) -> dict:
    checkpoint, task_name, split, n_episodes, out = job
    import torch

    from dexvision.eval.evaluator import evaluate
    from dexvision.learning.policy import load_learned_policy
    from dexvision.sim.workcell import Workcell
    from dexvision.tasks import make_task

    torch.set_num_threads(1)  # one process per checkpoint; keep them from oversubscribing
    task = make_task(task_name)
    policy, info = load_learned_policy(Path(checkpoint), task.spec)
    def progress(episodes, total):
        if len(episodes) % 50 == 0:
            print(f"  {Path(checkpoint).stem}: {len(episodes)}/{total} episodes, "
                  f"{sum(e.success for e in episodes)} successes", flush=True)

    report = evaluate(Workcell(), task, policy, split, n_episodes, policy_info=info,
                      on_episode=progress)
    Path(out).write_text(json.dumps(report.to_dict(), indent=1) + "\n", encoding="utf-8")
    o = report.overall
    return {"checkpoint": info["checkpoint"], "step": info["step"], "successes": o.successes,
            "n": o.n, "ci95": list(o.ci95), "report": Path(out).name}


def evaluate_checkpoints(checkpoints: Sequence[Path], task_name: str, n_episodes: int,
                         out_dir: Path, workers: int) -> list[dict]:
    """Evaluate each checkpoint on val in parallel processes. Reports go to ``out_dir``."""
    jobs = [(str(c), task_name, "val", n_episodes, str(out_dir / f"val_{Path(c).stem}.json"))
            for c in checkpoints]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return sorted(pool.map(_evaluate_one, jobs), key=lambda r: r["step"])
