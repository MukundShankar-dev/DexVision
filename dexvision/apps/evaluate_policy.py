"""CLI for the frozen Level 3.4 reach-policy rollout matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_CHECKPOINT = Path("outputs/level3/reach_bc_v1/policy.pt")
DEFAULT_PROTOCOL = Path("configs/level3_evaluation.yaml")
DEFAULT_TRAINING_CONFIG = Path("configs/level3_bc.yaml")
DEFAULT_MODEL = Path("assets/mujoco/task_board_scene.xml")
DEFAULT_OUTPUT = Path("outputs/level3/reach_rollout_v1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Level 3.3 policy on the frozen Level 3.4 reach matrix."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="Training config used to reconstruct and verify the source dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Override the extracted Level 2 dataset root in the training config.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sim-steps-per-action",
        type=int,
        default=17,
        help="MuJoCo integration steps per policy control action (default: 17).",
    )
    parser.add_argument(
        "--ablation-name",
        help="Required descriptive name when evaluating an action-subset checkpoint.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        from dexvision.evaluation.evaluate_policy import (
            evaluate_policy,
            load_reach_evaluation_protocol,
        )
        from dexvision.learning.datasets import load_frozen_reach_datasets
        from dexvision.learning.policies import load_checkpoint_policy
        from dexvision.learning.train_bc import load_experiment_config
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "PyTorch is required for policy evaluation; activate the dexvision "
                "Conda environment or install the learning dependencies."
            ) from exc
        raise

    training = load_experiment_config(args.training_config)
    dataset_root = args.dataset_root or training.dataset_root
    bundle = load_frozen_reach_datasets(
        dataset_root,
        evaluation_config_path=args.protocol,
        observation_fields=training.observation_fields,
        include_previous_action=training.include_previous_action,
        normalize=True,
    )
    policy = load_checkpoint_policy(
        args.checkpoint,
        expected_dataset_digest=bundle.manifest.dataset_digest,
    )
    protocol = load_reach_evaluation_protocol(args.protocol)

    print("DexVision Level 3.4 frozen reach closed-loop rollout")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint SHA-256: {policy.checkpoint_digest}")
    print(f"Dataset digest: {policy.dataset_digest}")
    print(f"Protocol: {protocol.version} ({protocol.source_digest})")
    print(f"Scenarios: {len(protocol.scenarios)}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    report = evaluate_policy(
        policy,
        protocol,
        output_dir=args.output_dir,
        model_path=args.model,
        sim_steps_per_action=args.sim_steps_per_action,
        ablation_name=args.ablation_name,
    )
    metrics = report.metrics
    print(
        "Results: "
        f"training_success={metrics['training_target_success_rate']:.3f}, "
        f"held_out_success={metrics['held_out_target_success_rate']:.3f}, "
        f"mean_jerk={metrics['mean_normalized_action_jerk']:.6f}"
    )
    print(
        "Safety: "
        f"invalid={metrics['invalid_action_count']}, "
        f"workspace={metrics['workspace_violation_count']}, "
        f"joint_limits={metrics['joint_limit_violation_count']}"
    )
    print(f"Terminal reasons: {metrics['terminal_reason_distribution']}")
    print(f"Frozen gates: {'PASS' if report.passed else 'FAIL'}")
    print(f"Report: {args.output_dir / 'report.json'}")
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
