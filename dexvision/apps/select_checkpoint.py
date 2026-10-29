"""Evaluate every checkpoint of a run on val and select one (rule in dexvision/eval/selection.py).

    python -m dexvision.apps.select_checkpoint --task pick --run runs/pick/diffusion_seed0

Per-checkpoint reports and selection.json go to the run directory. The selected checkpoint's val
report is copied to results/<task>/<method>_val.json together with the full selection table.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dexvision import REPO_ROOT
from dexvision.eval.selection import evaluate_checkpoints, select
from dexvision.tasks import TASK_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", type=Path, default=None,
                        help="default results/<task>/<method>_val.json")
    args = parser.parse_args()

    checkpoints = sorted(args.run.glob("step_*.pt"))
    if not checkpoints:
        parser.error(f"No step_*.pt checkpoints in {args.run}")
    print(f"Evaluating {len(checkpoints)} checkpoints on {args.episodes} val episodes "
          f"({args.workers} workers)", flush=True)
    table = evaluate_checkpoints(checkpoints, args.task, args.episodes, args.run, args.workers)
    chosen = select(table)
    for row in table:
        mark = "  <- selected" if row is chosen else ""
        lo, hi = row["ci95"]
        print(f"  step {row['step']:6d}: {row['successes']}/{row['n']} "
              f"[{lo:.1%}, {hi:.1%}]{mark}")
    selection = {"rule": "most val successes on the same seeds; ties go to the later step",
                 "candidates": table, "selected": chosen}
    (args.run / "selection.json").write_text(json.dumps(selection, indent=1) + "\n", "utf-8")

    report = json.loads((args.run / chosen["report"]).read_text(encoding="utf-8"))
    report["selection"] = selection
    method = report["policy"]
    out = args.out or REPO_ROOT / "results" / args.task / f"{method}_val.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"Selected {chosen['checkpoint']}; wrote {out}")


if __name__ == "__main__":
    main()
