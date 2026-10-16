"""Run the single fresh confirmation with mandatory release/settle coverage."""

from __future__ import annotations
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "dataset-dir", "output-dir"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    print("DexVision5.6 late-coverage fresh confirmation", flush=True)
    try:
        from dexvision.evaluation.perception_identity_confirmation import (
            confirm_identity_context,
        )

        r = confirm_identity_context(a.protocol, a.dataset_dir, a.output_dir)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as e:
        print("Confirmation failed:", e)
        return 1
    print(
        json.dumps(
            {
                k: r[k]
                for k in (
                    "recall",
                    "position_error_p95_m",
                    "all_primary_gates_pass",
                    "strict_validation_pass",
                    "automated_baseline_pass",
                    "fresh_late_phase_evidence",
                )
            },
            indent=2,
        )
    )
    return 0 if r["automated_baseline_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
