"""Build a derived union from two immutable 5.6 development releases."""

from __future__ import annotations
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parent",
        nargs=2,
        action="append",
        required=True,
        metavar=("PROTOCOL", "DATASET"),
    )
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    print("DexVision5.6 frozen development union", flush=True)
    try:
        from dexvision.logging.perception_union import union_development

        result = union_development(args.parent, args.output_dir)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print("Union failed:", exc)
        return 1
    print(
        json.dumps({k: result[k] for k in ("frame_count", "episode_count")}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
