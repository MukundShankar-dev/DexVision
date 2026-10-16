"""Finalize strict selection metadata without changing weights or selection."""

from __future__ import annotations
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("DexVision5.6 selection metadata binding", flush=True)
    try:
        from dexvision.learning.perception_binding import bind_selection

        print(json.dumps(bind_selection(args.config), indent=2))
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print("Binding failed:", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
