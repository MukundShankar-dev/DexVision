"""Inspect the complete Level 4.1 MuJoCo workcell."""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from dexvision.sim.mujoco_env import MujocoError
from dexvision.sim.workcell import (
    DEFAULT_WORKCELL_CONFIG,
    Workcell,
    WorkcellError,
    create_pick_task,
    create_place_task,
    create_press_task,
    create_push_task,
    create_reach_task,
)
from dexvision.sim.world_state import WorldStateError


DEFAULT_STEPS = 0
DEFAULT_VIEWER_SLEEP = 1.0 / 60.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reset and inspect the complete DexVision Level 4.1 workcell."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_WORKCELL_CONFIG,
        help=f"Workcell YAML. Defaults to {DEFAULT_WORKCELL_CONFIG}.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic reset seed.")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=(
            "Simulation steps to inspect before exiting. Zero keeps the viewer "
            "open until you close it (the default)."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run reset, schema, metric, and stability checks without a viewer.",
    )
    parser.add_argument(
        "--viewer-sleep",
        type=float,
        default=DEFAULT_VIEWER_SLEEP,
        help="Seconds to sleep after each viewer sync.",
    )
    return parser


def run_inspection(
    *, config_path: Path, seed: int, steps: int, headless: bool, viewer_sleep: float
) -> int:
    if steps < 0 or (headless and steps == 0):
        raise ValueError(
            "steps must be positive in headless mode and non-negative in viewer mode."
        )
    if viewer_sleep < 0.0:
        raise ValueError("viewer_sleep must be non-negative.")

    print("DexVision Level 4.1 workcell inspection")
    print(f"Config: {config_path}")
    print(f"Seed: {seed}")
    print(f"Viewer: {'off' if headless else 'on'}")
    print("No camera, recording, or learning code is used.")

    with Workcell(config_path) as workcell:
        first = workcell.reset(seed=seed)
        first_poses = _object_poses(workcell, first)
        second = workcell.reset(seed=seed)
        second_poses = _object_poses(workcell, second)
        if first_poses != second_poses:
            raise WorkcellError("Identical seeds produced different object layouts.")
        print("Same-seed reset reproducibility: PASS")
        print(
            f"World state: schema={second.schema_version}, frame={second.frame}, "
            f"entities={len(second.entities)}"
        )
        print("Objects:")
        for object_id in workcell.config.object_ids:
            observation = second.require_entity(object_id)
            relation = second.relation_for(object_id)
            print(
                f"  {object_id}: class={observation.class_id}, "
                f"position={list(observation.position)}, supported_by={relation.supported_by}"
            )
        print("Targets:")
        for target_id in workcell.config.target_ids:
            print(f"  {target_id}: position={list(second.require_entity(target_id).position)}")
        button = second.require_fixture("start_button")
        print(
            f"Fixture: start_button depth={button.press_depth_m:.6f}m, "
            f"pressed={button.pressed}"
        )
        _print_task_contracts(workcell, second)

        if headless:
            final = workcell.step(n_steps=steps)
        else:
            _ensure_viewer_can_launch(config_path, seed, steps, viewer_sleep)
            final = _run_with_viewer(workcell, seed, steps, viewer_sleep)

        if not (
            np.all(np.isfinite(workcell.env.data.qpos))
            and np.all(np.isfinite(workcell.env.data.qvel))
        ):
            raise WorkcellError("Workcell became non-finite during inspection.")
        print(f"Simulation stepped to t={final.timestamp:.3f}s: PASS")
        print("Press l in the viewer to toggle entity labels.")
    return 0


def _object_poses(workcell: Workcell, state) -> tuple:
    return tuple(
        (
            object_id,
            state.require_entity(object_id).position,
            state.require_entity(object_id).orientation_wxyz,
        )
        for object_id in workcell.config.object_ids
    )


def _print_task_contracts(workcell: Workcell, state) -> None:
    robot_pose = (*state.robot.base_position, *state.robot.base_orientation_wxyz)
    tasks = (
        create_reach_task(workcell, entity_id="block_small", approach_pose=robot_pose),
        create_pick_task(workcell, object_id="block_small"),
        create_place_task(
            workcell, object_id="block_small", target_id="inspection_pad"
        ),
        create_push_task(
            workcell, object_id="puck_light", target_zone="inspection_pad"
        ),
        create_press_task(workcell),
    )
    print("Task metric contracts:")
    for task in tasks:
        result = task.evaluate(state)
        metric_names = ", ".join(result.values)
        print(
            f"  {task.skill_name}: metrics=[{metric_names}], "
            f"dwell={result.required_dwell_steps}"
        )


def _ensure_viewer_can_launch(
    config_path: Path, seed: int, steps: int, viewer_sleep: float
) -> None:
    if sys.platform != "darwin":
        return
    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - dependency error path.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc
    mjpython_base = getattr(viewer, "_MjPythonBase", None)
    mjpython_dispatcher = getattr(viewer, "_MJPYTHON", None)
    if mjpython_base is not None and isinstance(mjpython_dispatcher, mjpython_base):
        return
    command = _format_mjpython_command(config_path, seed, steps, viewer_sleep)
    raise MujocoError(
        "MuJoCo viewer on macOS requires the mjpython launcher.\n"
        "Run this from a regular macOS Terminal or iTerm session:\n"
        f"  {command}"
    )


def _format_mjpython_command(
    config_path: Path, seed: int, steps: int, viewer_sleep: float
) -> str:
    command = [
        "mjpython",
        "-m",
        "dexvision.apps.inspect_workcell",
        "--config",
        str(config_path),
        "--seed",
        str(seed),
    ]
    if steps:
        command.extend(["--steps", str(steps)])
    if viewer_sleep != DEFAULT_VIEWER_SLEEP:
        command.extend(["--viewer-sleep", str(viewer_sleep)])
    return " ".join(shlex.quote(part) for part in command)


def _run_with_viewer(
    workcell: Workcell, seed: int, steps: int, viewer_sleep: float
):
    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - dependency error path.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    reset_requested = False
    next_seed_requested = False
    pause_toggle_requested = False
    label_toggle_requested = False

    def key_callback(keycode: int) -> None:
        nonlocal reset_requested
        nonlocal next_seed_requested
        nonlocal pause_toggle_requested
        nonlocal label_toggle_requested
        if keycode in (ord("R"), ord("r")):
            reset_requested = True
        elif keycode in (ord("N"), ord("n")):
            next_seed_requested = True
        elif keycode in (ord("L"), ord("l")):
            label_toggle_requested = True
        elif keycode == ord(" "):
            pause_toggle_requested = True

    try:
        with viewer.launch_passive(
            workcell.env.model, workcell.env.data, key_callback=key_callback
        ) as viewer_handle:
            _configure_free_camera(viewer_handle, workcell)
            print(
                "Viewer controls: left-drag orbit, right-drag pan, scroll zoom; "
                "r same-seed reset, n next seed, space pause/resume, l labels."
            )
            print(
                "The side-panel Run/Pause buttons are disabled because this is a "
                "passive viewer; use the keyboard controls above."
            )
            state = workcell.get_world_state()
            current_seed = seed
            paused = False
            labels_visible = bool(
                workcell.config.scene["viewer"]["entity_labels_initially_visible"]
            )
            iterations = 0
            while viewer_handle.is_running() and (steps == 0 or iterations < steps):
                if reset_requested:
                    state = workcell.reset(seed=current_seed)
                    reset_requested = False
                    print(f"Repeated deterministic reset for seed {current_seed}.")
                if next_seed_requested:
                    current_seed += 1
                    state = workcell.reset(seed=current_seed)
                    next_seed_requested = False
                    print(f"Loaded deterministic layout for seed {current_seed}.")
                if pause_toggle_requested:
                    paused = not paused
                    pause_toggle_requested = False
                    print(f"Simulation {'paused' if paused else 'resumed'}.")
                if label_toggle_requested:
                    labels_visible = not labels_visible
                    viewer_handle.opt.label = (
                        workcell.env._mujoco.mjtLabel.mjLABEL_SITE
                        if labels_visible
                        else workcell.env._mujoco.mjtLabel.mjLABEL_NONE
                    )
                    label_toggle_requested = False
                    print(f"Entity labels {'shown' if labels_visible else 'hidden'}.")
                if not paused:
                    state = workcell.step()
                viewer_handle.sync()
                if viewer_sleep > 0.0:
                    time.sleep(viewer_sleep)
                iterations += 1
    except Exception as exc:  # pragma: no cover - requires a desktop GUI.
        raise MujocoError(f"MuJoCo viewer failed to open or run: {exc}") from exc
    return state


def _configure_free_camera(viewer_handle, workcell: Workcell) -> None:
    """Start with a full-workcell view while preserving mouse camera control."""

    viewer_config = workcell.config.scene["viewer"]
    viewer_handle.cam.type = workcell.env._mujoco.mjtCamera.mjCAMERA_FREE
    viewer_handle.cam.lookat[:] = np.asarray(viewer_config["lookat_m"], dtype=np.float64)
    viewer_handle.cam.distance = float(viewer_config["distance_m"])
    viewer_handle.cam.azimuth = float(viewer_config["azimuth_deg"])
    viewer_handle.cam.elevation = float(viewer_config["elevation_deg"])
    viewer_handle.opt.label = (
        workcell.env._mujoco.mjtLabel.mjLABEL_SITE
        if viewer_config["entity_labels_initially_visible"]
        else workcell.env._mujoco.mjtLabel.mjLABEL_NONE
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_inspection(
            config_path=args.config,
            seed=args.seed,
            steps=args.steps,
            headless=args.headless,
            viewer_sleep=args.viewer_sleep,
        )
    except (MujocoError, WorkcellError, WorldStateError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Workcell inspection closed cleanly.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
