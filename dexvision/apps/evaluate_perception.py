"""One-time held-out checkpoint 5.6 evaluation with review artifacts."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    print(f"Perception {args.split} evaluation: {args.config}; output={args.output_dir}")
    try:
        from dexvision.evaluation.evaluate_perception import evaluate_perception
        report = evaluate_perception(args.config, args.output_dir, split=args.split)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f"Perception evaluation failed: {exc}")
        return 1
    print(json.dumps({key: report[key] for key in (
        "frame_count", "recall", "position_error_median_m", "position_error_p95_m",
        "numerical_gates", "baseline_qualified", "checkpoint_complete")}, indent=2))
    return 0 if report["automated_baseline_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
