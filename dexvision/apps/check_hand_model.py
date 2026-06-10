"""Inspect and step the DexVision MuJoCo robot hand model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.sim.hand_model import (
    check_rest_stability,
    format_hand_model_report,
    inspect_hand_model,
    require_controllable_hand,
)
from dexvision.sim.mujoco_env import MujocoError


DEFAULT_MODEL = Path("assets/mujoco/hand_scene.xml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load and inspect a DexVision MuJoCo hand model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"MuJoCo hand XML model path. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=240,
        help="Number of headless rest-stability simulation steps to run.",
    )
    return parser


def run_hand_model_check(model_path: Path, *, steps: int) -> int:
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")

    print("DexVision hand model check")
    print("No camera or MediaPipe is used.")

    info = inspect_hand_model(model_path)
    require_controllable_hand(info)
    stability = check_rest_stability(model_path, steps=steps)
    print(format_hand_model_report(info, stability))

    if not stability.stable:
        raise MujocoError("Hand model rest simulation became unstable.")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_hand_model_check(model_path=args.model, steps=args.steps)
    except (MujocoError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Hand model check closed cleanly.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
