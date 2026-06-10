"""Drive exactly one MuJoCo hand joint with a periodic command."""

from __future__ import annotations

import argparse
import math
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from dexvision.sim.mujoco_env import MujocoEnv, MujocoError, MujocoState


DEFAULT_MODEL = Path("assets/mujoco/debug_hand_scene.xml")
DEFAULT_JOINT = "index_mcp"
DEFAULT_STEPS = 600
DEFAULT_FREQUENCY_HZ = 1.0
DEFAULT_PRINT_INTERVAL = 30
DEFAULT_VIEWER_SLEEP = 1.0 / 60.0
MAX_ABS_QPOS = 3.0
MAX_ABS_QVEL = 40.0


class ViewerHandle(Protocol):
    """Small protocol for MuJoCo passive viewer handles used by this app."""

    def sync(self) -> None:
        """Synchronize the viewer with the current MuJoCo state."""


@dataclass(frozen=True)
class OneJointBinding:
    """Resolved metadata for driving one scalar hand joint through one actuator."""

    joint_name: str
    actuator_name: str
    joint_id: int
    actuator_id: int
    qpos_index: int
    joint_minimum: float
    joint_maximum: float
    control_minimum: float
    control_maximum: float

    @property
    def target_minimum(self) -> float:
        """Lowest target that satisfies both joint and actuator limits."""

        return max(self.joint_minimum, self.control_minimum)

    @property
    def target_maximum(self) -> float:
        """Highest target that satisfies both joint and actuator limits."""

        return min(self.joint_maximum, self.control_maximum)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drive one DexVision MuJoCo hand joint with a sine-wave target."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"MuJoCo hand XML model path. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--joint",
        default=DEFAULT_JOINT,
        help=(
            "Joint name to drive, or the name of a position actuator attached "
            f"to one joint. Defaults to {DEFAULT_JOINT}."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of simulation steps to run.",
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=DEFAULT_FREQUENCY_HZ,
        help="Sine-wave target frequency in simulation-time Hz.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the MuJoCo viewer. Useful for automated checks.",
    )
    parser.add_argument(
        "--viewer-sleep",
        type=float,
        default=DEFAULT_VIEWER_SLEEP,
        help="Seconds to sleep between viewer syncs when the viewer is open.",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=DEFAULT_PRINT_INTERVAL,
        help="Print joint target/current values every N simulation steps.",
    )
    return parser


def run_one_joint(
    *,
    model_path: Path,
    selected_joint: str,
    steps: int,
    frequency_hz: float,
    viewer: bool,
    viewer_sleep: float,
    print_interval: int,
) -> int:
    """Load the hand model and drive one selected joint/actuator."""

    _validate_run_parameters(
        steps=steps,
        frequency_hz=frequency_hz,
        viewer_sleep=viewer_sleep,
        print_interval=print_interval,
    )

    print("DexVision one-joint MuJoCo check")
    print(f"Model: {model_path}")
    print(f"Selected joint/actuator: {selected_joint}")
    print(f"Steps: {steps}")
    print(f"Viewer: {'on' if viewer else 'off'}")
    print("No camera, MediaPipe, or hand tracking is used.")

    with MujocoEnv(model_path) as env:
        env.reset()
        binding = resolve_one_joint_binding(env, selected_joint)
        _require_valid_target_span(binding)
        print(_format_binding_report(binding))

        if viewer:
            _ensure_viewer_can_launch(
                model_path=model_path,
                selected_joint=selected_joint,
                steps=steps,
                frequency_hz=frequency_hz,
                viewer_sleep=viewer_sleep,
                print_interval=print_interval,
            )
            final_state = _drive_with_viewer(
                env=env,
                binding=binding,
                steps=steps,
                frequency_hz=frequency_hz,
                viewer_sleep=viewer_sleep,
                print_interval=print_interval,
            )
        else:
            final_state = _drive_steps(
                env=env,
                binding=binding,
                steps=steps,
                frequency_hz=frequency_hz,
                print_interval=print_interval,
            )

    print(f"Simulation stepped to t={final_state.time:.3f}s.")
    return 0


def resolve_one_joint_binding(env: MujocoEnv, selected_name: str) -> OneJointBinding:
    """Resolve a user-selected joint or actuator name to one scalar control binding."""

    if not selected_name:
        raise MujocoError("Joint name cannot be empty.")

    mujoco_module = env._mujoco
    model = env.model
    joint_id = mujoco_module.mj_name2id(
        model,
        mujoco_module.mjtObj.mjOBJ_JOINT,
        selected_name,
    )

    if joint_id >= 0:
        actuator_id = _find_single_actuator_for_joint(model, int(joint_id), selected_name)
    else:
        actuator_id = mujoco_module.mj_name2id(
            model,
            mujoco_module.mjtObj.mjOBJ_ACTUATOR,
            selected_name,
        )
        if actuator_id < 0:
            raise MujocoError(
                f"Unknown hand joint or actuator '{selected_name}'. "
                f"Available joints: {', '.join(_available_joint_names(env))}"
            )
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            raise MujocoError(f"Actuator '{selected_name}' is not attached to a joint.")

    joint_name = _name_for_id(
        mujoco_module,
        model,
        mujoco_module.mjtObj.mjOBJ_JOINT,
        int(joint_id),
    )
    actuator_name = _name_for_id(
        mujoco_module,
        model,
        mujoco_module.mjtObj.mjOBJ_ACTUATOR,
        int(actuator_id),
    )

    if not bool(model.jnt_limited[joint_id]):
        raise MujocoError(f"Joint '{joint_name}' is missing required limits.")
    if not bool(model.actuator_ctrllimited[actuator_id]):
        raise MujocoError(f"Actuator '{actuator_name}' is missing required control limits.")

    return OneJointBinding(
        joint_name=joint_name,
        actuator_name=actuator_name,
        joint_id=int(joint_id),
        actuator_id=int(actuator_id),
        qpos_index=int(model.jnt_qposadr[joint_id]),
        joint_minimum=float(model.jnt_range[joint_id, 0]),
        joint_maximum=float(model.jnt_range[joint_id, 1]),
        control_minimum=float(model.actuator_ctrlrange[actuator_id, 0]),
        control_maximum=float(model.actuator_ctrlrange[actuator_id, 1]),
    )


def compute_periodic_target(binding: OneJointBinding, time_seconds: float, frequency_hz: float) -> float:
    """Return a sine-wave target clipped to the binding's valid target range."""

    _require_valid_target_span(binding)
    center = 0.5 * (binding.target_minimum + binding.target_maximum)
    amplitude = 0.45 * (binding.target_maximum - binding.target_minimum)
    target = center + amplitude * math.sin(2.0 * math.pi * frequency_hz * time_seconds)
    return float(np.clip(target, binding.target_minimum, binding.target_maximum))


def _drive_with_viewer(
    *,
    env: MujocoEnv,
    binding: OneJointBinding,
    steps: int,
    frequency_hz: float,
    viewer_sleep: float,
    print_interval: int,
) -> MujocoState:
    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - depends on optional GUI support.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    try:
        with viewer.launch_passive(env.model, env.data) as viewer_handle:
            return _drive_steps(
                env=env,
                binding=binding,
                steps=steps,
                frequency_hz=frequency_hz,
                print_interval=print_interval,
                viewer_handle=viewer_handle,
                viewer_sleep=viewer_sleep,
            )
    except Exception as exc:  # pragma: no cover - requires desktop GUI to exercise.
        raise MujocoError(f"MuJoCo viewer failed to open or run: {exc}") from exc


def _drive_steps(
    *,
    env: MujocoEnv,
    binding: OneJointBinding,
    steps: int,
    frequency_hz: float,
    print_interval: int,
    viewer_handle: ViewerHandle | None = None,
    viewer_sleep: float = 0.0,
) -> MujocoState:
    state = env.get_state()
    for step_index in range(steps):
        target = compute_periodic_target(binding, env.data.time, frequency_hz)
        env.set_joint_targets({binding.actuator_name: target})
        state = env.step()
        current = float(env.data.qpos[binding.qpos_index])
        _raise_if_unstable(state)

        if step_index == 0 or (step_index + 1) % print_interval == 0 or step_index + 1 == steps:
            print(
                f"step={step_index + 1:04d} "
                f"joint={binding.joint_name} "
                f"target={target:.4f} "
                f"current={current:.4f}"
            )

        if viewer_handle is not None:
            viewer_handle.sync()
            if viewer_sleep > 0.0:
                time.sleep(viewer_sleep)

    return state


def _ensure_viewer_can_launch(
    *,
    model_path: Path,
    selected_joint: str,
    steps: int,
    frequency_hz: float,
    viewer_sleep: float,
    print_interval: int,
) -> None:
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

    command = _format_mjpython_command(
        model_path=model_path,
        selected_joint=selected_joint,
        steps=steps,
        frequency_hz=frequency_hz,
        viewer_sleep=viewer_sleep,
        print_interval=print_interval,
    )
    raise MujocoError(
        "MuJoCo viewer on macOS requires the mjpython launcher.\n"
        "Run this from a regular macOS Terminal or iTerm session:\n"
        f"  {command}\n"
        "For automated headless checks, add --headless."
    )


def _format_mjpython_command(
    *,
    model_path: Path,
    selected_joint: str,
    steps: int,
    frequency_hz: float,
    viewer_sleep: float,
    print_interval: int,
) -> str:
    command = [
        "mjpython",
        "-m",
        "dexvision.apps.check_one_joint",
        "--model",
        str(model_path),
        "--joint",
        selected_joint,
        "--steps",
        str(steps),
    ]
    if frequency_hz != DEFAULT_FREQUENCY_HZ:
        command.extend(["--frequency-hz", str(frequency_hz)])
    if viewer_sleep != DEFAULT_VIEWER_SLEEP:
        command.extend(["--viewer-sleep", str(viewer_sleep)])
    if print_interval != DEFAULT_PRINT_INTERVAL:
        command.extend(["--print-interval", str(print_interval)])
    return " ".join(shlex.quote(part) for part in command)


def _format_binding_report(binding: OneJointBinding) -> str:
    return (
        f"Joint: {binding.joint_name} "
        f"range=[{binding.joint_minimum:.3f}, {binding.joint_maximum:.3f}]\n"
        f"Actuator: {binding.actuator_name} "
        f"ctrlrange=[{binding.control_minimum:.3f}, {binding.control_maximum:.3f}]\n"
        f"Command span: [{binding.target_minimum:.3f}, {binding.target_maximum:.3f}]"
    )


def _find_single_actuator_for_joint(model: object, joint_id: int, joint_name: str) -> int:
    actuator_ids = [
        actuator_id
        for actuator_id in range(model.nu)
        if int(model.actuator_trnid[actuator_id, 0]) == joint_id
    ]
    if not actuator_ids:
        raise MujocoError(f"Joint '{joint_name}' has no attached actuator.")
    if len(actuator_ids) > 1:
        raise MujocoError(
            f"Joint '{joint_name}' has multiple attached actuators; select one actuator by name."
        )
    return int(actuator_ids[0])


def _available_joint_names(env: MujocoEnv) -> tuple[str, ...]:
    return tuple(
        _name_for_id(env._mujoco, env.model, env._mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(env.model.njnt)
    )


def _name_for_id(mujoco_module: object, model: object, object_type: object, object_id: int) -> str:
    name = mujoco_module.mj_id2name(model, object_type, object_id)
    if name is None:
        raise MujocoError(f"MuJoCo object id {object_id} is unnamed.")
    return str(name)


def _require_valid_target_span(binding: OneJointBinding) -> None:
    if binding.target_minimum >= binding.target_maximum:
        raise MujocoError(
            f"Joint '{binding.joint_name}' and actuator '{binding.actuator_name}' "
            "do not have overlapping limits."
        )


def _raise_if_unstable(state: MujocoState) -> None:
    if not np.all(np.isfinite(state.qpos)) or not np.all(np.isfinite(state.qvel)):
        raise MujocoError("Simulation became unstable: non-finite qpos or qvel.")
    max_abs_qpos = float(np.max(np.abs(state.qpos))) if state.qpos.size else 0.0
    max_abs_qvel = float(np.max(np.abs(state.qvel))) if state.qvel.size else 0.0
    if max_abs_qpos > MAX_ABS_QPOS or max_abs_qvel > MAX_ABS_QVEL:
        raise MujocoError(
            "Simulation became unstable: "
            f"max_abs_qpos={max_abs_qpos:.6f}, max_abs_qvel={max_abs_qvel:.6f}."
        )


def _validate_run_parameters(
    *,
    steps: int,
    frequency_hz: float,
    viewer_sleep: float,
    print_interval: int,
) -> None:
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    if viewer_sleep < 0.0:
        raise ValueError("viewer_sleep must be non-negative.")
    if print_interval <= 0:
        raise ValueError("print_interval must be a positive integer.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_one_joint(
            model_path=args.model,
            selected_joint=args.joint,
            steps=args.steps,
            frequency_hz=args.frequency_hz,
            viewer=not args.headless,
            viewer_sleep=args.viewer_sleep,
            print_interval=args.print_interval,
        )
    except (MujocoError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. One-joint check closed cleanly.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
