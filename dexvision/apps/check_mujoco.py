"""Load and step a MuJoCo scene without camera or hand tracking."""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

from dexvision.sim.mujoco_env import MujocoEnv, MujocoError, MujocoState


DEFAULT_MODEL = Path("assets/mujoco/simple_scene.xml")
DEFAULT_VIEWER_SLEEP = 1.0 / 60.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load and step a DexVision MuJoCo scene.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"MuJoCo XML model path. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=120,
        help="Number of simulation steps to run.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the MuJoCo passive viewer while stepping.",
    )
    parser.add_argument(
        "--viewer-sleep",
        type=float,
        default=DEFAULT_VIEWER_SLEEP,
        help="Seconds to sleep between viewer syncs when --viewer is set.",
    )
    return parser


def run_mujoco(model_path: Path, steps: int, *, viewer: bool, viewer_sleep: float) -> int:
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")
    if viewer_sleep < 0.0:
        raise ValueError("viewer_sleep must be non-negative.")

    print("DexVision MuJoCo smoke test")
    print(f"Model: {model_path}")
    print(f"Steps: {steps}")
    print(f"Viewer: {'on' if viewer else 'off'}")
    print("No camera or MediaPipe is used.")

    with MujocoEnv(model_path) as env:
        initial = env.reset()
        print(
            "Loaded model "
            f"nq={env.model.nq}, nv={env.model.nv}, nu={env.model.nu}, "
            f"initial_time={initial.time:.3f}s"
        )

        if viewer:
            _ensure_viewer_can_launch(model_path, steps, viewer_sleep)
            final = _run_with_viewer(env, steps, viewer_sleep)
        else:
            final = env.step(n_steps=steps)

    print(f"Simulation stepped to t={final.time:.3f}s.")
    return 0


def _ensure_viewer_can_launch(model_path: Path, steps: int, viewer_sleep: float) -> None:
    if sys.platform != "darwin":
        return

    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - MuJoCo import is tested elsewhere.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    mjpython_base = getattr(viewer, "_MjPythonBase", None)
    mjpython_dispatcher = getattr(viewer, "_MJPYTHON", None)
    if mjpython_base is not None and isinstance(mjpython_dispatcher, mjpython_base):
        return

    command = _format_mjpython_command(model_path, steps, viewer_sleep)
    raise MujocoError(
        "MuJoCo viewer on macOS requires the mjpython launcher.\n"
        "Run this from a regular macOS Terminal or iTerm session:\n"
        f"  {command}\n"
        "Headless stepping still works with python when --viewer is omitted."
    )


def _format_mjpython_command(model_path: Path, steps: int, viewer_sleep: float) -> str:
    command = [
        "mjpython",
        "-m",
        "dexvision.apps.check_mujoco",
        "--model",
        str(model_path),
        "--viewer",
        "--steps",
        str(steps),
    ]
    if viewer_sleep != DEFAULT_VIEWER_SLEEP:
        command.extend(["--viewer-sleep", str(viewer_sleep)])
    return " ".join(shlex.quote(part) for part in command)


def _run_with_viewer(env: MujocoEnv, steps: int, viewer_sleep: float) -> MujocoState:
    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - depends on optional GUI support.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    try:
        with viewer.launch_passive(env.model, env.data) as viewer_handle:
            for _ in range(steps):
                state = env.step()
                viewer_handle.sync()
                if viewer_sleep > 0.0:
                    time.sleep(viewer_sleep)
    except Exception as exc:  # pragma: no cover - requires desktop GUI to exercise.
        raise MujocoError(f"MuJoCo viewer failed to open or run: {exc}") from exc

    return state


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_mujoco(
            model_path=args.model,
            steps=args.steps,
            viewer=args.viewer,
            viewer_sleep=args.viewer_sleep,
        )
    except (MujocoError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. MuJoCo simulation closed cleanly.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
