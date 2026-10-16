"""Freeze, collect, package and evaluate fresh checkpoint 5.6 confirmation."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output-dir", required=True)
    for name in ("collect", "evaluate", "package"):
        sub = commands.add_parser(name)
        sub.add_argument("--protocol", required=True)
        if name != "collect":
            sub.add_argument("--dataset-dir", required=True)
        sub.add_argument("--release" if name == "package" else "--output-dir", required=True)
    args = parser.parse_args()
    print(f"DexVision checkpoint 5.6 confirmation: {args.command}", flush=True)
    try:
        from dexvision.logging.perception_confirmation import (
            collect_confirmation, freeze_protocol, package_confirmation,
        )
        if args.command == "freeze":
            result = freeze_protocol(args.config, args.output_dir)
            summary = {"cases": result["case_count"], "frozen": True}
        elif args.command == "collect":
            result = collect_confirmation(args.protocol, args.output_dir)
            summary = {k: result[k] for k in ("episode_count", "frame_count", "historical_pixel_collisions")}
        elif args.command == "package":
            result = package_confirmation(args.protocol, args.dataset_dir, args.release)
            summary = {k: result[k] for k in ("archive", "archive_sha256", "archive_bytes")}
        else:
            from dexvision.evaluation.perception_confirmation import evaluate_confirmation
            result = evaluate_confirmation(args.protocol, args.dataset_dir, args.output_dir)
            summary = {k: result[k] for k in ("frame_count", "recall", "position_error_p95_m",
                                               "all_primary_gates_pass", "automated_baseline_pass")}
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, StopIteration) as exc:
        print(f"Confirmation failed: {exc}")
        return 1
    print(json.dumps(summary, indent=2))
    return 2 if args.command == "evaluate" and not result["automated_baseline_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
