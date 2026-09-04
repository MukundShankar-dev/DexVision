"""Replay a saved Level 2 demonstration episode in MuJoCo."""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from dexvision.logging.replay_demo import (
    DemoReplayError,
    LoadedReplayDemo,
    ReplayStep,
    load_replay_demo,
    replay_loaded_demo,
)
from dexvision.sim.mujoco_env import MujocoEnv, MujocoError


DEFAULT_SPEED = 1.0
DEFAULT_LEGACY_SIM_STEPS_PER_ACTION = 1
DEFAULT_PRINT_INTERVAL = 30
DEFAULT_VIEWER_SLEEP = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a saved DexVision Level 2 demonstration in MuJoCo."
    )
    parser.add_argument(
        "--demo",
        type=Path,
        required=True,
        help="Saved demo directory containing metadata.json and .npy arrays.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override the MuJoCo model path saved in demo metadata.",
    )
    parser.add_argument(
        "--mocap-body",
        default=None,
        help="Override the hand-base mocap body saved in teleop config metadata.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Replay without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=(
            "Replay speed multiplier. 1.0 follows recorded timestamps; "
            "0.5 is half speed; 2.0 is double speed."
        ),
    )
    parser.add_argument(
        "--sim-steps-per-action",
        type=int,
        default=None,
        help=(
            "MuJoCo integration steps to run for each recorded action row. "
            "Defaults to the saved recording cadence, or 1 for legacy demos."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Replay at most this many recorded action rows.",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=DEFAULT_PRINT_INTERVAL,
        help="Print replay status every N action rows.",
    )
    parser.add_argument(
        "--viewer-sleep",
        type=float,
        default=DEFAULT_VIEWER_SLEEP,
        help="Extra seconds to sleep after each MuJoCo viewer sync.",
    )
    return parser


def run_replay_demo(args: argparse.Namespace) -> int:
    """Load a demo and replay its recorded actions."""

    _validate_args(args)
    loaded = load_replay_demo(
        args.demo,
        model_override=args.model,
        mocap_body_override=args.mocap_body,
    )
    sim_steps_per_action = _resolve_sim_steps_per_action(
        loaded,
        args.sim_steps_per_action,
    )

    print("DexVision Level 2 demo replay")
    print(f"Demo: {loaded.demo_dir}")
    print(f"Task: {loaded.episode.metadata.get('task_id')}")
    print(f"Skill: {loaded.episode.metadata.get('skill_name')}")
    print(f"Episode: {loaded.episode.metadata.get('episode_id')}")
    print(f"MuJoCo model: {loaded.model_path}")
    print(f"Mocap body: {loaded.mocap_body_name}")
    print(f"Action schema: {loaded.action_schema.version}, dim={loaded.action_schema.action_dim}")
    print(f"Finger targets: {', '.join(loaded.finger_target_names)}")
    print(f"Mode: {'headless' if args.headless else 'viewer'}")
    print(f"Speed: {args.speed:g}x")
    print(f"Simulation steps/action: {sim_steps_per_action}")

    with MujocoEnv(loaded.model_path) as env:
        if args.headless:
            result = _run_replay(
                loaded,
                env,
                speed=args.speed,
                sim_steps_per_action=sim_steps_per_action,
                max_steps=args.max_steps,
                print_interval=args.print_interval,
            )
        else:
            result = _run_replay_with_viewer(
                loaded,
                env,
                args=args,
                sim_steps_per_action=sim_steps_per_action,
            )

    status = "stopped early" if result.stopped_early else "complete"
    print(
        "Replay "
        f"{status}: steps={result.steps_replayed}, "
        f"timestamps={_format_timestamp(result.first_timestamp)}"
        f"->{_format_timestamp(result.last_timestamp)}, "
        f"sim_t={_format_timestamp(result.final_sim_time)}s"
    )
    return 0


def _run_replay(
    loaded: LoadedReplayDemo,
    env: MujocoEnv,
    *,
    speed: float,
    sim_steps_per_action: int,
    max_steps: int | None,
    print_interval: int,
) -> object:
    return replay_loaded_demo(
        loaded,
        env,
        speed=speed,
        sim_steps_per_action=sim_steps_per_action,
        max_steps=max_steps,
        progress_callback=_progress_printer(print_interval),
    )


def _run_replay_with_viewer(
    loaded: LoadedReplayDemo,
    env: MujocoEnv,
    *,
    args: argparse.Namespace,
    sim_steps_per_action: int,
) -> object:
    _ensure_viewer_can_launch(args)
    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - MuJoCo import tested elsewhere.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    try:
        with viewer.launch_passive(env.model, env.data) as viewer_handle:
            _configure_saved_workcell_viewer(viewer_handle, loaded)

            def sync_viewer() -> None:
                viewer_handle.sync()
                if args.viewer_sleep > 0.0:
                    time.sleep(args.viewer_sleep)

            return replay_loaded_demo(
                loaded,
                env,
                speed=args.speed,
                sim_steps_per_action=sim_steps_per_action,
                max_steps=args.max_steps,
                viewer_sync=sync_viewer,
                should_stop=lambda: _viewer_was_closed(viewer_handle),
                progress_callback=_progress_printer(args.print_interval),
            )
    except Exception as exc:  # pragma: no cover - requires desktop GUI to exercise.
        if isinstance(exc, (DemoReplayError, MujocoError)):
            raise
        raise MujocoError(f"MuJoCo viewer failed to open or run: {exc}") from exc


def _configure_saved_workcell_viewer(
    viewer_handle: object, loaded: LoadedReplayDemo
) -> None:
    """Restore the operator-facing camera saved with a Level 4 workcell demo."""

    if loaded.episode.metadata.get("task_id") != "level4_workcell":
        return
    task_config = loaded.episode.metadata.get("task_config")
    viewer_config = (
        task_config.get("viewer_config") if isinstance(task_config, dict) else None
    )
    camera = getattr(viewer_handle, "cam", None)
    if not isinstance(viewer_config, dict) or camera is None:
        return
    lookat = np.asarray(viewer_config.get("lookat_m"), dtype=np.float64)
    if lookat.shape == (3,) and np.all(np.isfinite(lookat)):
        camera.lookat[:] = lookat
    camera.distance = float(viewer_config.get("distance_m", camera.distance))
    camera.azimuth = float(viewer_config.get("azimuth_deg", camera.azimuth))
    camera.elevation = float(viewer_config.get("elevation_deg", camera.elevation))


def _progress_printer(print_interval: int):
    def print_progress(step: ReplayStep, state: object) -> None:
        if step.index == 0 or (step.index + 1) % print_interval == 0:
            print(
                f"replayed={step.index + 1:05d} "
                f"demo_t={step.timestamp:.3f}s "
                f"sim_t={_format_timestamp(getattr(state, 'time', None))}s"
            )

    return print_progress


def _validate_args(args: argparse.Namespace) -> None:
    if args.speed <= 0.0:
        raise ValueError("speed must be positive.")
    if args.sim_steps_per_action is not None and args.sim_steps_per_action <= 0:
        raise ValueError("sim_steps_per_action must be positive.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")
    if args.print_interval <= 0:
        raise ValueError("print_interval must be positive.")
    if args.viewer_sleep < 0.0:
        raise ValueError("viewer_sleep must be non-negative.")


def _ensure_viewer_can_launch(args: argparse.Namespace) -> None:
    if sys.platform != "darwin":
        return

    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - MuJoCo import tested elsewhere.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    mjpython_base = getattr(viewer, "_MjPythonBase", None)
    mjpython_dispatcher = getattr(viewer, "_MJPYTHON", None)
    if mjpython_base is not None and isinstance(mjpython_dispatcher, mjpython_base):
        return

    raise MujocoError(
        "MuJoCo viewer on macOS requires the mjpython launcher.\n"
        "Run this from a regular macOS Terminal or iTerm session:\n"
        f"  {_format_mjpython_command(args)}"
    )


def _format_mjpython_command(args: argparse.Namespace) -> str:
    command = [
        "mjpython",
        "-m",
        "dexvision.apps.replay_demo",
        "--demo",
        str(args.demo),
        "--speed",
        str(args.speed),
    ]
    if args.sim_steps_per_action is not None:
        command.extend(
            ["--sim-steps-per-action", str(args.sim_steps_per_action)]
        )
    if args.model is not None:
        command.extend(["--model", str(args.model)])
    if args.mocap_body is not None:
        command.extend(["--mocap-body", str(args.mocap_body)])
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.viewer_sleep != DEFAULT_VIEWER_SLEEP:
        command.extend(["--viewer-sleep", str(args.viewer_sleep)])
    return " ".join(shlex.quote(part) for part in command)


def _resolve_sim_steps_per_action(
    loaded: LoadedReplayDemo,
    override: int | None,
) -> int:
    """Use an explicit replay cadence or the cadence saved by the recorder."""

    if override is not None:
        return override
    recording = loaded.episode.metadata.get("recording")
    if not isinstance(recording, dict) or "sim_steps_per_frame" not in recording:
        return DEFAULT_LEGACY_SIM_STEPS_PER_ACTION
    saved_steps = recording["sim_steps_per_frame"]
    if isinstance(saved_steps, bool) or not isinstance(saved_steps, int) or saved_steps <= 0:
        raise DemoReplayError(
            "metadata recording.sim_steps_per_frame must be a positive integer."
        )
    return saved_steps


def _viewer_was_closed(viewer_handle: object) -> bool:
    is_running = getattr(viewer_handle, "is_running", None)
    if not callable(is_running):
        return False
    return not bool(is_running())


def _format_timestamp(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_replay_demo(args)
    except (DemoReplayError, MujocoError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
