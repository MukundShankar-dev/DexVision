"""CLI for the Level 2.10 saved-demo retargeting benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.evaluation.benchmark_retargeters import (
    RETARGETER_NAMES,
    RetargetingBenchmarkError,
    discover_task_episodes,
    replay_push_cube_success,
    run_benchmark,
    save_benchmark_plot,
    save_benchmark_report,
)
from dexvision.logging.replay_demo import DemoReplayError
from dexvision.sim.mujoco_env import MujocoError


DEFAULT_DATASET_ROOT = Path("data/demos/raw")
DEFAULT_CONFIG = Path("configs/level1_teleop.yaml")
DEFAULT_OUTPUT_DIR = Path("data/demos/reports/retargeting")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare curl, fingertip, and optimization retargeters on identical "
            "saved task episodes."
        )
    )
    parser.add_argument("--task", required=True, help="Saved task_id to benchmark.")
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of saved episodes to use."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing one directory per task.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Shared robot target mapping and limits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for benchmark JSON, CSV, and SVG outputs.",
    )
    parser.add_argument(
        "--retargeters",
        nargs="+",
        choices=RETARGETER_NAMES,
        default=list(RETARGETER_NAMES),
        help="Retargeters to compare; curl and fingertip are required.",
    )
    parser.add_argument(
        "--recorded-success",
        action="store_true",
        help=(
            "Use saved operator success labels instead of counterfactual MuJoCo "
            "replay. Required for tasks other than push_cube_to_target."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.episodes <= 0:
        raise RetargetingBenchmarkError("--episodes must be positive.")
    if args.task != "push_cube_to_target" and not args.recorded_success:
        raise RetargetingBenchmarkError(
            "counterfactual success replay currently supports push_cube_to_target; "
            "pass --recorded-success for other tasks."
        )
    episode_dirs = discover_task_episodes(
        args.dataset_root, task_id=args.task, episodes=args.episodes
    )
    stem = f"{args.task}_retargeting_benchmark"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    plot_path = args.output_dir / f"{stem}.svg"

    print("DexVision Level 2.10 retargeting benchmark")
    print(f"Task: {args.task}")
    print(f"Episodes: {len(episode_dirs)}")
    print(f"Retargeters: {', '.join(args.retargeters)}")
    print(f"Config: {args.config}")
    print(
        "Task success: "
        + ("recorded labels" if args.recorded_success else "headless MuJoCo replay")
    )
    report = run_benchmark(
        episode_dirs,
        task_id=args.task,
        config_path=args.config,
        retargeter_names=args.retargeters,
        success_evaluator=None if args.recorded_success else replay_push_cube_success,
    )
    save_benchmark_report(report, json_path=json_path, csv_path=csv_path)
    save_benchmark_plot(report, plot_path)
    for metric in report.metrics:
        fingertip = (
            "n/a"
            if metric.mean_fingertip_error is None
            else f"{metric.mean_fingertip_error:.6f}"
        )
        print(
            f"{metric.retargeter}: latency={metric.mean_latency_ms:.4f} ms, "
            f"jerk={metric.mean_action_jerk:.6f}, "
            f"limit_rate={metric.joint_limit_violation_rate:.6f}, "
            f"fingertip_error={fingertip}, success={metric.task_success_rate:.3f}"
        )
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (DemoReplayError, MujocoError, RetargetingBenchmarkError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
