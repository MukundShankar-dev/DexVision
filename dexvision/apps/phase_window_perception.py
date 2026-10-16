"""Freeze or render the 5.6 TRAIN/validation stabilization-window amendment."""

from __future__ import annotations
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    for name in ("freeze", "collect"):
        q = s.add_parser(name)
        q.add_argument("--config" if name == "freeze" else "--protocol", required=True)
        q.add_argument("--output-dir", required=True)
    a = p.parse_args()
    print("DexVision5.6 phase-window amendment:", a.command, flush=True)
    try:
        from dexvision.logging.perception_phase_windows import (
            freeze_phase_windows,
            collect_phase_windows,
        )

        result = (
            freeze_phase_windows(a.config, a.output_dir)
            if a.command == "freeze"
            else collect_phase_windows(a.protocol, a.output_dir)
        )
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as e:
        print("Phase amendment failed:", e)
        return 1
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("case_count", "frame_count", "query_count")
                if k in result
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
