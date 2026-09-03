"""Command-line entry point for corrected Level 3 behavior-cloning training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_CONFIG = Path("configs/level3_reach_bc_v2.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the shared Level 3 goal-conditioned MLP with offline-validation "
            "checkpoint selection."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Versioned behavior-cloning YAML config.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Override the extracted Level 2 dataset root from the config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the checkpoint output directory from the config.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from an epoch-boundary last checkpoint.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        from dexvision.learning.train_bc import load_experiment_config, run_experiment
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "PyTorch is required for training; install the learning dependencies "
                "with `pip install -e '.[learning]'`."
            ) from exc
        raise

    config = load_experiment_config(args.config)
    dataset_root = args.dataset_root or config.dataset_root
    output_dir = args.output_dir or config.output_dir
    print("DexVision Level 3 behavior-cloning training")
    print(f"Skill: {config.skill_name}")
    print(f"Dataset root: {dataset_root}")
    print(f"Evaluation config: {config.evaluation_config}")
    print(f"Dataset eligibility: {config.eligibility}")
    print(f"Goal input mode: {config.goal_input_mode}")
    print(f"Best output: {output_dir / config.best_checkpoint_name}")
    print(f"Last output: {output_dir / config.checkpoint_name}")
    print(
        "Training: "
        f"epochs={config.training.epochs}, batch_size={config.training.batch_size}, "
        f"seed={config.training.seed}, device={config.training.device}"
    )
    if args.resume is not None:
        print(f"Resume checkpoint: {args.resume}")
    result = run_experiment(
        config,
        dataset_root=dataset_root,
        output_dir=output_dir,
        resume_from=args.resume,
    )
    final = result.loss_history[-1]
    print(
        f"Completed epoch {result.completed_epochs}: "
        f"train_loss={final['train_loss']:.8f}, "
        f"validation_loss={final['validation_loss']:.8f}"
    )
    print(
        f"Selected epoch {result.selected_epoch}: "
        f"validation_loss={result.selected_validation_loss:.8f}"
    )
    print(f"Best checkpoint: {result.best_checkpoint_path}")
    print(f"Best SHA-256: {result.best_checkpoint_digest}")
    print(f"Best digest file: {result.best_digest_path}")
    print(f"Last checkpoint: {result.last_checkpoint_path}")
    print(f"Last SHA-256: {result.last_checkpoint_digest}")
    print(f"Last digest file: {result.last_digest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
