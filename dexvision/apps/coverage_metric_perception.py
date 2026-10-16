"""Train the frozen 5.6 targeted coverage amendment with exact prior TRAIN cache."""

from __future__ import annotations
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    print("DexVision5.6 targeted coverage and metric-loss repair", flush=True)
    try:
        from dexvision.learning.perception_coverage_metric import train_coverage

        result = train_coverage(args.config)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print("Coverage repair failed:", exc)
        return 1
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("recall", "all_primary_gates_pass", "strict_validation_pass")
            },
            indent=2,
        )
    )
    return 0 if result["strict_validation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
