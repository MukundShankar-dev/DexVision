"""Fit and select the checkpoint 5.6 conventional RGB baseline."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(f"Perception fitting/validation: {args.config}; deterministic CPU baseline")
    try:
        from dexvision.learning.perception_baseline import train_perception
        report = train_perception(args.config)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f"Perception training failed: {exc}")
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
