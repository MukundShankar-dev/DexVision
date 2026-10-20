"""Measure the workcell's physical behaviour and write it to a JSON report.

    python -m dexvision.apps.sim_report --out results/sim_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dexvision.sim.diagnostics import full_report
from dexvision.sim.workcell import DEFAULT_CONFIG, Workcell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=Path("results/sim_report.json"))
    args = parser.parse_args()

    print(f"Loading workcell from {args.config}")
    report = full_report(Workcell(args.config))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("tracking", "press", "hand_mass_kg")}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
