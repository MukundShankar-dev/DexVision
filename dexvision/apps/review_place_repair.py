"""Review saved, qualified Level 5.4A place confirmations in MuJoCo."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/level5/place_policy_safety_v8.yaml'))
    parser.add_argument('--selection-config', type=Path, default=Path('configs/level5/place_safety_selection_v8.yaml'))
    parser.add_argument('--seed', type=int, default=5001)
    parser.add_argument('--playback-speed', type=float, default=1.0)
    args = parser.parse_args(argv)
    print(f'DexVision place repair review: seed {args.seed}, recipe {args.config}', flush=True)
    try:
        from dexvision.evaluation.place_repair_review import review_place

        result = review_place(args.config, args.seed, playback_speed=args.playback_speed,
                              selection_config=args.selection_config)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'ERROR: {exc}. Use the dexvision environment; macOS viewer requires mjpython.',
              file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
