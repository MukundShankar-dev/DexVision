"""Freeze a new 5.6 confirmation only after dense phase coverage passes."""

from __future__ import annotations
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    print("DexVision5.6 phase-qualified fresh confirmation freeze", flush=True)
    try:
        from dexvision.evaluation.perception_phase_confirmation import (
            freeze_phase_confirmation,
        )

        result = freeze_phase_confirmation(a.config, a.output_dir)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as e:
        print("Freeze failed:", e)
        return 1
    print("Frozen cases:", result["case_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
