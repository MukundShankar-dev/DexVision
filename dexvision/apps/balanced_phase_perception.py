"""Harvest or fit a frozen 5.6 phase-balanced TRAIN repair."""

from __future__ import annotations
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--config", required=True)
    h.add_argument("--stream", choices=("base", "phase"), required=True)
    h.add_argument("--output", required=True)
    a = p.parse_args()
    print("DexVision5.6 phase-balanced repair:", a.command, flush=True)
    try:
        from dexvision.learning.perception_balanced_phase import harvest, train_coverage

        result = (
            harvest(a.config, a.stream, a.output)
            if a.command == "harvest"
            else train_coverage(a.config)
        )
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as e:
        print("Balanced repair failed:", e)
        return 1
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("stream", "TRAIN_rows", "missing", "strict_validation_pass")
                if k in result
            },
            indent=2,
        )
    )
    return 0 if a.command == "harvest" or result["strict_validation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
