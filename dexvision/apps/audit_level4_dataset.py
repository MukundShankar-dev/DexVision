"""CLI for the read-only Level 4.8 dataset audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.evaluation.dataset_audit import audit_level4_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/level4_dataset.yaml"))
    parser.add_argument("--splits", type=Path, default=Path("configs/level4_splits.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/demos/level4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/level4/audit"))
    args = parser.parse_args(argv)
    print(f"DexVision Level 4.8 dataset audit: {args.dataset_dir}", flush=True)
    try:
        report = audit_level4_dataset(config_path=args.config, splits_path=args.splits,
                                     dataset_dir=args.dataset_dir, output_dir=args.output_dir,
                                     progress=lambda message: print(message, flush=True))
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Audit {'PASS' if report['passed'] else 'FAIL'}: {args.output_dir / 'report.json'}")
    print(f"Active episodes: {report['active_episode_count']}; issues: {len(report['issues'])}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
