"""Train the frozen 5.6 role/phase-balanced context repair."""

from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("DexVision5.6 role/phase context refit", flush=True)
    try:
        from dexvision.learning.perception_context_phase import train_phase_context

        report = train_phase_context(args.config)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print("Phase context refit failed:", exc)
        return 1
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("recall", "all_primary_gates_pass", "strict_validation_pass")
            },
            indent=2,
        )
    )
    return 0 if report["strict_validation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
