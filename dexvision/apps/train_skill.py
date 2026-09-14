"""Inspect or train a frozen Level 5 skill without camera or visible simulator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, help="Verified extracted release root; defaults to immutable archive")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model", choices=("baseline", "reference"), default="baseline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-steps", type=int, help="Checkpoint and stop after this many steps; resume retains the frozen epoch budget")
    args = parser.parse_args(argv)
    print(f"DexVision Level 5.1 skill training: {args.config}", flush=True)
    try:
        from dexvision.learning.train_skill import run_training_command

        result = run_training_command(args.config, release_root=args.release_root, output_dir=args.output_dir,
            seed=args.seed, device=args.device, model=args.model, dry_run=args.dry_run, resume=args.resume,
            stop_after_steps=args.stop_after_steps)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}. Use the dexvision Conda environment and verified release files.", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
