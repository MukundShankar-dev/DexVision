"""Freeze fresh confirmation only after all late-phase gates pass."""

import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    print("DexVision5.6 late-phase-qualified confirmation freeze", flush=True)
    try:
        from dexvision.evaluation.perception_late_confirmation import (
            freeze_late_confirmation,
        )

        result = freeze_late_confirmation(a.config, a.output_dir)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as e:
        print("Freeze failed:", e)
        return 1
    print("Frozen cases:", result["case_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
