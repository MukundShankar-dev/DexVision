"""CLI for frozen Level 3 reach, button, and push rollout matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_TRAINING_CONFIG = Path("configs/level3_reach_bc_v2.yaml")
DEFAULT_MODEL = Path("assets/mujoco/task_board_scene.xml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a selected Level 3 policy on its frozen rollout matrix."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to evaluate (default: config's best-validation checkpoint).",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Frozen protocol override (default: training config value).",
    )
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--sim-steps-per-action",
        type=int,
        help="Override MuJoCo integration steps per policy control action.",
    )
    parser.add_argument(
        "--ablation-name",
        help="Required descriptive name when evaluating an action-subset checkpoint.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        from dexvision.evaluation.evaluate_policy import (
            evaluate_manipulation_policy,
            evaluate_policy,
            load_manipulation_evaluation_protocol,
            load_reach_evaluation_protocol,
            save_reach_v1_v2_comparison,
        )
        from dexvision.learning.datasets import load_frozen_skill_datasets
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
    protocol_path = args.protocol or training.evaluation_config
    protocol_versions = {
        "reach_touch_target": "level3/reach-evaluation-v1",
        "button_press": "level3/button-evaluation-v1",
        "push_cube_to_target": "level3/push-evaluation-v1",
    }
    bundle = load_frozen_skill_datasets(
        dataset_root,
        evaluation_config_path=protocol_path,
        expected_version=protocol_versions[training.skill_name],
        expected_skill_name=training.skill_name,
        observation_fields=training.observation_fields,
        include_previous_action=training.include_previous_action,
        normalize=True,
        eligibility=training.eligibility,
        goal_input_mode=training.goal_input_mode,
    )
    checkpoint = args.checkpoint or training.best_checkpoint_path
    policy = load_checkpoint_policy(
        checkpoint,
        expected_dataset_digest=bundle.manifest.dataset_digest,
    )
    default_outputs = {
        "reach_touch_target": Path("outputs/level3/reach_rollout_v2"),
        "button_press": Path("outputs/level3/button_rollout_v1"),
        "push_cube_to_target": Path("outputs/level3/push_rollout_v1"),
    }
    output_dir = args.output_dir or default_outputs[training.skill_name]

    print("DexVision Level 3 frozen closed-loop rollout")
    print(f"Task: {training.skill_name}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Checkpoint SHA-256: {policy.checkpoint_digest}")
    print(f"Dataset digest: {policy.dataset_digest}")
    print(f"Dataset eligibility: {training.eligibility}")
    print(f"Goal input mode: {policy.goal_input_mode}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    if training.skill_name == "reach_touch_target":
        protocol = load_reach_evaluation_protocol(protocol_path)
        print(f"Protocol: {protocol.version} ({protocol.source_digest})")
        print(f"Scenarios: {len(protocol.scenarios)}")
        report = evaluate_policy(
            policy,
            protocol,
            output_dir=output_dir,
            model_path=args.model,
            sim_steps_per_action=args.sim_steps_per_action or 17,
            ablation_name=args.ablation_name,
        )
        v1_report = Path("outputs/level3/reach_rollout_v1/report.json")
        v1_checkpoint = Path("outputs/level3/reach_bc_v1/policy.pt")
        if v1_report.is_file() and v1_checkpoint.is_file():
            comparison_path = output_dir / "comparison_with_v1.json"
            save_reach_v1_v2_comparison(
                v1_report_path=v1_report,
                v1_checkpoint_path=v1_checkpoint,
                v2_report_path=output_dir / "report.json",
                output_path=comparison_path,
            )
            print(f"V1 comparison: {comparison_path}")
    else:
        protocol = load_manipulation_evaluation_protocol(protocol_path)
        print(f"Protocol: {protocol.version} ({protocol.source_digest})")
        print(f"Scenarios: {len(protocol.scenarios)}")
        report = evaluate_manipulation_policy(
            policy,
            protocol,
            output_dir=output_dir,
            model_path=args.model,
            sim_steps_per_action=args.sim_steps_per_action,
            ablation_name=args.ablation_name,
        )
    metrics = report.metrics
    print(
        "Results: "
        f"training_success={metrics.get('training_target_success_rate', metrics.get('training_goal_success_rate')):.3f}, "
        f"held_out_success={metrics.get('held_out_target_success_rate', metrics.get('held_out_goal_success_rate')):.3f}, "
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
    print(f"Report: {output_dir / 'report.json'}")
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
