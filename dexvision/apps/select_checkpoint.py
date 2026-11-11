"""Evaluate every checkpoint of a run on val and select one (rule in dexvision/eval/selection.py).

    python -m dexvision.apps.select_checkpoint --task pick --run runs/pick/diffusion_seed0
    python -m dexvision.apps.select_checkpoint --task pick --follow \
        --run runs/pick/diffusion_seed0 runs/pick/mlp_seed0

With ``--follow`` the runs may still be training (or not started): each checkpoint is evaluated
as it is saved, so selection overlaps training instead of following it. Per-checkpoint reports
and selection.json go to the run directory. The selected checkpoint's val report is copied to
results/<task>/<method>_val.json together with the full selection table.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dexvision import REPO_ROOT
from dexvision.eval.selection import evaluate_checkpoint, evaluate_checkpoints, follow, select
from dexvision.learning.train import load_method_config
from dexvision.tasks import TASK_NAMES


def write_selection(run: Path, table: list[dict], task: str, out: Path | None) -> None:
    chosen = select(table)
    print(f"{run}:")
    for row in table:
        mark = "  <- selected" if row is chosen else ""
        lo, hi = row["ci95"]
        print(f"  step {row['step']:6d}: {row['successes']}/{row['n']} "
              f"[{lo:.1%}, {hi:.1%}]{mark}")
    selection = {"rule": "most val successes on the same seeds; ties go to the later step",
                 "candidates": table, "selected": chosen}
    (run / "selection.json").write_text(json.dumps(selection, indent=1) + "\n", "utf-8")

    report = json.loads((run / chosen["report"]).read_text(encoding="utf-8"))
    report["selection"] = selection
    out = out or REPO_ROOT / "results" / task / f"{report['policy']}_val.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"Selected {chosen['checkpoint']}; wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--follow", action="store_true",
                        help="evaluate checkpoints as training saves them, until --last-step")
    parser.add_argument("--last-step", type=int, default=None,
                        help="with --follow: the final checkpoint's step (default: the method's)")
    parser.add_argument("--out", type=Path, default=None,
                        help="one run only; default results/<task>/<method>_val.json")
    args = parser.parse_args()
    if args.out and len(args.run) > 1:
        parser.error("--out names one file; pass a single --run")

    if args.follow:
        last = args.last_step or int(load_method_config()["training"]["steps"])
        print(f"Following {len(args.run)} runs to step {last}: {args.episodes} val episodes per "
              f"checkpoint ({args.workers} workers)", flush=True)

        def one(checkpoint: Path) -> dict:
            return evaluate_checkpoint(checkpoint, args.task, args.episodes, checkpoint.parent,
                                       args.workers)
        tables = follow(args.run, last, one)
    else:
        tables = {}
        for run in args.run:
            checkpoints = sorted(run.glob("step_*.pt"))
            if not checkpoints:
                parser.error(f"No step_*.pt checkpoints in {run}")
            print(f"Evaluating {len(checkpoints)} checkpoints of {run} on {args.episodes} val "
                  f"episodes ({args.workers} workers)", flush=True)
            tables[run] = evaluate_checkpoints(checkpoints, args.task, args.episodes, run,
                                               args.workers)
    for run, table in tables.items():
        write_selection(run, table, args.task, args.out)


if __name__ == "__main__":
    main()
