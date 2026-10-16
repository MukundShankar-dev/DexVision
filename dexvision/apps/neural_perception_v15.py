"""Checkpoint 5.6 neural RGB-context repair; consumes only separately frozen data."""

from __future__ import annotations
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("train", "validate", "freeze-confirmation"):
        c = sub.add_parser(name)
        c.add_argument("--config", required=True)
        if name != "train":
            c.add_argument("--output-dir", required=True)
    c = sub.add_parser("confirm")
    c.add_argument("--protocol", required=True)
    c.add_argument("--dataset-dir", required=True)
    c.add_argument("--output-dir", required=True)
    args = p.parse_args()
    print("DexVision 5.6 neural RGB-context repair:", args.command, flush=True)
    try:
        from dexvision.learning.perception_neural_v15 import train_context
        from dexvision.evaluation.perception_neural import (
            validate_context,
            confirm_context,
            freeze_context_confirmation,
        )

        if args.command == "train":
            result = train_context(args.config)
        elif args.command == "freeze-confirmation":
            result = freeze_context_confirmation(args.config, args.output_dir)
            print("Frozen context confirmation cases:", result["case_count"])
            return 0
        elif args.command == "validate":
            result = validate_context(args.config, args.output_dir)
        else:
            result = confirm_context(args.protocol, args.dataset_dir, args.output_dir)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print("Context repair failed:", exc)
        return 1
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "recall",
                    "position_error_p95_m",
                    "all_primary_gates_pass",
                    "strict_validation_pass",
                    "automated_baseline_pass",
                )
                if k in result
            },
            indent=2,
        )
    )
    return (
        0
        if result.get(
            "strict_validation_pass", result.get("automated_baseline_pass", False)
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
