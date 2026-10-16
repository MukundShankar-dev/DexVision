"""Select the frozen TRAIN-hue repair; never consume a confirmation set."""

from __future__ import annotations
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    a = p.parse_args()
    print("DexVision5.6 TRAIN-only hue refinement with frozen pose weights", flush=True)
    try:
        from dexvision.learning.perception_hue_repair import select_hue

        result = select_hue(a.config)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as e:
        print("Hue repair failed:", e)
        return 1
    print("Strict validation:", result["strict_validation_pass"])
    return 0 if result["strict_validation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
