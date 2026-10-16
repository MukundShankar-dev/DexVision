"""Separate 5.6 development and fresh confirmation; never advance to 5.7."""
from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-development", "freeze-confirmation", "train", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name != "train":
            p.add_argument("--output-dir", required=True)
    for name in ("collect", "package", "confirm"):
        p = sub.add_parser(name)
        p.add_argument("--protocol", required=True)
        if name != "collect":
            p.add_argument("--dataset-dir", required=True)
        p.add_argument("--release" if name == "package" else "--output-dir", required=True)
    p = sub.add_parser("review")
    p.add_argument("--evaluation-dir", required=True)
    args = parser.parse_args()
    print(f"DexVision 5.6 repair: {args.command}", flush=True)
    try:
        from dexvision.logging.perception_development import collect_dataset, freeze_protocol, package_dataset
        from dexvision.learning.perception_development import load_repair_recipe, train_repair
        from dexvision.evaluation.perception_development import (
            evaluate_confirmation, freeze_confirmation, review_validation, review_artifacts,
        )
        if args.command == "freeze-development":
            result = freeze_protocol(args.config, args.output_dir)
        elif args.command == "freeze-confirmation":
            result = freeze_confirmation(args.config, args.output_dir)
        elif args.command == "collect":
            result = collect_dataset(args.protocol, args.output_dir)
        elif args.command == "package":
            result = package_dataset(args.protocol, args.dataset_dir, args.release)
        elif args.command == "train":
            result = train_repair(args.config, load_repair_recipe(args.config))
        elif args.command == "validate":
            result = review_validation(args.config, args.output_dir)
        elif args.command == "review":
            result = review_artifacts(args.evaluation_dir)
        else:
            result = evaluate_confirmation(args.protocol, args.dataset_dir, args.output_dir)
        keys = ("case_count", "frame_count", "recall", "position_error_p95_m", "primary_validation",
                "all_primary_gates_pass", "automated_baseline_pass", "archive", "archive_sha256",
                "reviewed_sequences", "owner_acceptance")
        print(json.dumps({k: result[k] for k in keys if k in result}, indent=2))
        if args.command in {"confirm", "validate"} and not result["automated_baseline_pass"]:
            return 2
        if args.command == "train" and not result["all_primary_gates_pass"]:
            return 2
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, StopIteration) as exc:
        print(f"5.6 repair failed: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
