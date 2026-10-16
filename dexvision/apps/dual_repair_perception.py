"""Checkpoint5.6 frozen dual-head localization repair on TRAIN/validation only."""

from __future__ import annotations
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("harvest", "train", "freeze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print("DexVision5.6 dual RGB localization:", args.command, flush=True)
    try:
        if args.command == "freeze":
            if not args.output_dir:
                raise ValueError("--output-dir required to freeze confirmation.")
            from dexvision.evaluation.perception_dual import freeze_context_confirmation

            freeze_context_confirmation(args.config, args.output_dir)
            return 0
        if args.command == "harvest":
            from dexvision.learning.perception_dual_examples import harvest

            harvest(args.config)
            return 0
        from dexvision.learning.perception_dual import train_dual

        report = train_dual(args.config)
        return 0 if report["strict_validation_pass"] else 2
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as error:
        print("Dual localization repair failed:", error, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
