"""Break an evaluation report down to say why episodes failed (docs/PLAN.md, V3 diagnostics).

    python -m dexvision.apps.diagnose results/pick_vision/diffusion_val.json \
        --against results/pick/diffusion_val.json

Prints the phase each episode reached, success by object and by start position, metric medians
for successes and failures, and, with --against, a seed-by-seed comparison with another report
(usually the state policy on the same val seeds). --out also writes it all as JSON.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dexvision.eval.diagnose import metric_summary, paired, phases, position_grid
from dexvision.tasks.spec import load_task_spec


def _rate(k: int, n: int, lo: float, hi: float) -> str:
    return f"{k:3d}/{n:<3d} [{lo:4.0%}, {hi:4.0%}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--against", type=Path, help="a report on the same seeds to compare with")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    episodes = report["episodes"]
    spec = load_task_spec(report.get("task", report["skill"]))
    o = report["overall"]
    print(f"{spec.name} / {report['policy']} / {report['split']}: "
          f"{_rate(o['successes'], o['n'], *o['ci95'])}")
    result: dict = {"report": str(args.report), "phases": phases(episodes)}

    print("\nPhase reached (furthest stage of the success predicate):")
    for name, count in result["phases"].items():
        print(f"  {name:20s} {count:4d}")

    print("\nBy object:")
    for name, rate in report["by_object"].items():
        print(f"  {name:15s} {_rate(rate['successes'], rate['n'], *rate['ci95'])}")

    reset = spec.raw.get("reset", {})
    if "object_center_xy_m" in reset and "start_x_m" in episodes[0]["metrics"]:
        grid = position_grid(episodes, reset["object_center_xy_m"],
                             reset["object_half_extent_xy_m"])
        result["position_grid"] = [[asdict(c) for c in row] for row in grid]
        print("\nBy start position (rows: x low to high, columns: y low to high):")
        for row in grid:
            print("  " + "   ".join(f"{c.successes:3d}/{c.n:<3d}" for c in row))

    names = [k for k in episodes[0]["metrics"] if not k.startswith("start_")]
    result["metric_medians"] = metric_summary(episodes, names)
    print("\nMetric medians (successes | failures; -1 'never' values left out):")
    for name, row in result["metric_medians"].items():
        cells = [f"{row[k]:.4g}" if row[k] is not None else "-" for k in ("success", "failure")]
        print(f"  {name:24s} {cells[0]:>10s} | {cells[1]:>10s}")

    if args.against:
        reference = json.loads(args.against.read_text(encoding="utf-8"))
        p = paired(episodes, reference["episodes"])
        result["paired"] = {"against": str(args.against), **asdict(p)}
        print(f"\nPaired with {reference.get('task', reference['skill'])} / "
              f"{reference['policy']} on {p.both + p.only_this + p.only_reference + p.neither} "
              "shared seeds:")
        print(f"  both {p.both}, only this {p.only_this}, only reference {p.only_reference}, "
              f"neither {p.neither}; exact McNemar p = {p.p_value:.3g}")
        if p.reference_only_seeds:
            print(f"  this policy's phase where only the reference succeeded: "
                  f"{p.reference_only_phases}")
            print(f"  those seeds (first 20): {list(p.reference_only_seeds[:20])}")

    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
