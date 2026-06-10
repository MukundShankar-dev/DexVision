"""Play scripted MuJoCo robot hand gestures without camera input."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dexvision.sim.mujoco_env import MujocoEnv, MujocoError, MujocoState


DEFAULT_MODEL = Path("assets/mujoco/debug_hand_scene.xml")
DEFAULT_GESTURES = Path("configs/hand_gestures.yaml")
DEFAULT_PRINT_INTERVAL = 30
DEFAULT_VIEWER_SLEEP = 1.0 / 60.0
MAX_ABS_QPOS = 3.0
MAX_ABS_QVEL = 40.0
REQUIRED_GESTURES = frozenset(
    {
        "open_hand",
        "fist",
        "point",
        "pinch",
        "peace_sign",
        "relax",
    }
)


class ViewerHandle(Protocol):
    """Small protocol for MuJoCo passive viewer handles used by this app."""

    def sync(self) -> None:
        """Synchronize the viewer with the current MuJoCo state."""


@dataclass(frozen=True)
class ActuatorBinding:
    """Resolved metadata for one scalar position actuator and its joint."""

    actuator_name: str
    joint_name: str
    actuator_id: int
    joint_id: int
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


@dataclass(frozen=True)
class GestureLibrary:
    """Scripted robot hand gestures loaded from a config file."""

    path: Path
    gestures: dict[str, dict[str, float]]
    sequence: tuple[str, ...]
    steps_per_gesture: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play scripted DexVision MuJoCo hand gestures without camera input."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"MuJoCo hand XML model path. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--gestures",
        type=Path,
        default=DEFAULT_GESTURES,
        help=f"Gesture YAML config path. Defaults to {DEFAULT_GESTURES}.",
    )
    parser.add_argument(
        "--sequence",
        nargs="+",
        default=None,
        help="Optional gesture names to play in order. Defaults to the config sequence.",
    )
    parser.add_argument(
        "--steps-per-gesture",
        type=int,
        default=None,
        help="Override the config hold duration for each gesture.",
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
        help="Print simulation status every N steps within each gesture.",
    )
    return parser


def load_gesture_config(config_path: str | Path) -> GestureLibrary:
    """Load and validate the gesture config schema.

    The bundled file is JSON-compatible YAML so it remains readable even in a
    minimal environment without PyYAML. If PyYAML is installed, normal YAML is
    accepted as well.
    """

    path = Path(config_path)
    if not path.exists():
        raise MujocoError(f"Gesture config file does not exist: {path}")
    if not path.is_file():
        raise MujocoError(f"Gesture config path is not a file: {path}")

    text = path.read_text(encoding="utf-8")
    raw_config = _parse_config_text(text, path)
    return _coerce_gesture_library(raw_config, path)


def resolve_actuator_bindings(env: MujocoEnv) -> dict[str, ActuatorBinding]:
    """Resolve all MuJoCo actuators to their attached limited scalar joints."""

    bindings: dict[str, ActuatorBinding] = {}
    for actuator_id in range(env.model.nu):
        joint_id = int(env.model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            raise MujocoError(f"Actuator at id {actuator_id} is not attached to a joint.")
        if not bool(env.model.jnt_limited[joint_id]):
            joint_name = _name_for_id(env, env._mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise MujocoError(f"Joint '{joint_name}' is missing required limits.")
        if not bool(env.model.actuator_ctrllimited[actuator_id]):
            actuator_name = _name_for_id(env, env._mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            raise MujocoError(f"Actuator '{actuator_name}' is missing required control limits.")

        actuator_name = _name_for_id(env, env._mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_name = _name_for_id(env, env._mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        binding = ActuatorBinding(
            actuator_name=actuator_name,
            joint_name=joint_name,
            actuator_id=int(actuator_id),
            joint_id=joint_id,
            qpos_index=int(env.model.jnt_qposadr[joint_id]),
            joint_minimum=float(env.model.jnt_range[joint_id, 0]),
            joint_maximum=float(env.model.jnt_range[joint_id, 1]),
            control_minimum=float(env.model.actuator_ctrlrange[actuator_id, 0]),
            control_maximum=float(env.model.actuator_ctrlrange[actuator_id, 1]),
        )
        if binding.target_minimum >= binding.target_maximum:
            raise MujocoError(
                f"Joint '{joint_name}' and actuator '{actuator_name}' do not have overlapping limits."
            )
        bindings[actuator_name] = binding

    if not bindings:
        raise MujocoError("Hand model has no actuators to drive.")
    return bindings


def validate_gesture_library(
    library: GestureLibrary,
    bindings: Mapping[str, ActuatorBinding],
) -> None:
    """Validate gesture targets against available actuators and limits."""

    binding_names = set(bindings)
    for gesture_name, targets in library.gestures.items():
        target_names = set(targets)
        missing = sorted(binding_names - target_names)
        unknown = sorted(target_names - binding_names)
        if missing:
            raise MujocoError(
                f"Gesture '{gesture_name}' is missing actuator targets: {', '.join(missing)}"
            )
        if unknown:
            raise MujocoError(
                f"Gesture '{gesture_name}' references unknown actuators: {', '.join(unknown)}"
            )

        for actuator_name, target in targets.items():
            binding = bindings[actuator_name]
            if not (binding.target_minimum <= target <= binding.target_maximum):
                raise MujocoError(
                    f"Gesture '{gesture_name}' target for '{actuator_name}'={target:.4f} "
                    f"is outside [{binding.target_minimum:.4f}, {binding.target_maximum:.4f}]."
                )


def run_hand_actuation(
    *,
    model_path: Path,
    gesture_config_path: Path,
    sequence_override: Sequence[str] | None,
    steps_per_gesture_override: int | None,
    viewer: bool,
    viewer_sleep: float,
    print_interval: int,
) -> int:
    """Load the hand model and play scripted gesture targets."""

    _validate_run_parameters(
        steps_per_gesture_override=steps_per_gesture_override,
        viewer_sleep=viewer_sleep,
        print_interval=print_interval,
    )

    library = load_gesture_config(gesture_config_path)
    sequence = _resolve_sequence(library, sequence_override)
    steps_per_gesture = (
        library.steps_per_gesture
        if steps_per_gesture_override is None
        else steps_per_gesture_override
    )

    print("DexVision hand actuation check")
    print(f"Model: {model_path}")
    print(f"Gesture config: {library.path}")
    print(f"Sequence: {', '.join(sequence)}")
    print(f"Steps per gesture: {steps_per_gesture}")
    print(f"Viewer: {'on' if viewer else 'off'}")
    print("No camera, MediaPipe, hand tracking, or retargeting is used.")

    with MujocoEnv(model_path) as env:
        env.reset()
        bindings = resolve_actuator_bindings(env)
        validate_gesture_library(library, bindings)
        print(f"Validated {len(library.gestures)} gestures against {len(bindings)} actuator limits.")

        if viewer:
            _ensure_viewer_can_launch(
                model_path=model_path,
                gesture_config_path=gesture_config_path,
                sequence_override=sequence_override,
                steps_per_gesture_override=steps_per_gesture_override,
                viewer_sleep=viewer_sleep,
                print_interval=print_interval,
            )
            final_state = _play_with_viewer(
                env=env,
                library=library,
                sequence=sequence,
                steps_per_gesture=steps_per_gesture,
                viewer_sleep=viewer_sleep,
                print_interval=print_interval,
            )
        else:
            final_state = _play_sequence(
                env=env,
                library=library,
                sequence=sequence,
                steps_per_gesture=steps_per_gesture,
                print_interval=print_interval,
            )

    print(f"Simulation stepped to t={final_state.time:.3f}s.")
    return 0


def _parse_config_text(text: str, path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise MujocoError(
                "PyYAML is not installed, and the gesture config is not JSON-compatible YAML: "
                f"{path}. Install PyYAML or use the bundled config format."
            ) from exc

    try:
        return yaml.safe_load(text)
    except Exception as exc:  # pragma: no cover - exact PyYAML exception varies.
        raise MujocoError(f"Failed to parse gesture config '{path}': {exc}") from exc


def _coerce_gesture_library(raw_config: Any, path: Path) -> GestureLibrary:
    if not isinstance(raw_config, Mapping):
        raise MujocoError(f"Gesture config must contain a mapping at the top level: {path}")

    raw_gestures = raw_config.get("gestures")
    if not isinstance(raw_gestures, Mapping):
        raise MujocoError("Gesture config must contain a 'gestures' mapping.")

    raw_steps = raw_config.get("steps_per_gesture", 120)
    if not isinstance(raw_steps, int) or raw_steps <= 0:
        raise MujocoError("'steps_per_gesture' must be a positive integer.")

    gestures: dict[str, dict[str, float]] = {}
    for raw_name, raw_targets in raw_gestures.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise MujocoError("Gesture names must be non-empty strings.")
        if not isinstance(raw_targets, Mapping):
            raise MujocoError(f"Gesture '{raw_name}' must contain an actuator target mapping.")

        targets: dict[str, float] = {}
        for raw_actuator_name, raw_target in raw_targets.items():
            if not isinstance(raw_actuator_name, str) or not raw_actuator_name:
                raise MujocoError(f"Gesture '{raw_name}' has a non-string actuator name.")
            if not isinstance(raw_target, int | float):
                raise MujocoError(
                    f"Gesture '{raw_name}' target for '{raw_actuator_name}' must be numeric."
                )
            target = float(raw_target)
            if not math.isfinite(target):
                raise MujocoError(
                    f"Gesture '{raw_name}' target for '{raw_actuator_name}' must be finite."
                )
            targets[raw_actuator_name] = target
        gestures[raw_name] = targets

    missing_required = sorted(REQUIRED_GESTURES - set(gestures))
    if missing_required:
        raise MujocoError(f"Gesture config is missing required gestures: {', '.join(missing_required)}")

    raw_sequence = raw_config.get("sequence", tuple(gestures))
    sequence = _coerce_sequence(raw_sequence, gestures)

    return GestureLibrary(
        path=path,
        gestures=gestures,
        sequence=sequence,
        steps_per_gesture=raw_steps,
    )


def _coerce_sequence(raw_sequence: object, gestures: Mapping[str, Mapping[str, float]]) -> tuple[str, ...]:
    if not isinstance(raw_sequence, Sequence) or isinstance(raw_sequence, str):
        raise MujocoError("'sequence' must be a list of gesture names.")

    sequence: list[str] = []
    for raw_name in raw_sequence:
        if not isinstance(raw_name, str) or not raw_name:
            raise MujocoError("'sequence' entries must be non-empty gesture names.")
        if raw_name not in gestures:
            raise MujocoError(f"Sequence references unknown gesture '{raw_name}'.")
        sequence.append(raw_name)

    if not sequence:
        raise MujocoError("'sequence' must contain at least one gesture.")
    return tuple(sequence)


def _resolve_sequence(
    library: GestureLibrary,
    sequence_override: Sequence[str] | None,
) -> tuple[str, ...]:
    if sequence_override is None:
        return library.sequence
    return _coerce_sequence(tuple(sequence_override), library.gestures)


def _play_with_viewer(
    *,
    env: MujocoEnv,
    library: GestureLibrary,
    sequence: Sequence[str],
    steps_per_gesture: int,
    viewer_sleep: float,
    print_interval: int,
) -> MujocoState:
    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - depends on optional GUI support.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    try:
        with viewer.launch_passive(env.model, env.data) as viewer_handle:
            return _play_sequence(
                env=env,
                library=library,
                sequence=sequence,
                steps_per_gesture=steps_per_gesture,
                print_interval=print_interval,
                viewer_handle=viewer_handle,
                viewer_sleep=viewer_sleep,
            )
    except Exception as exc:  # pragma: no cover - requires desktop GUI to exercise.
        raise MujocoError(f"MuJoCo viewer failed to open or run: {exc}") from exc


def _play_sequence(
    *,
    env: MujocoEnv,
    library: GestureLibrary,
    sequence: Sequence[str],
    steps_per_gesture: int,
    print_interval: int,
    viewer_handle: ViewerHandle | None = None,
    viewer_sleep: float = 0.0,
) -> MujocoState:
    state = env.get_state()
    for gesture_index, gesture_name in enumerate(sequence, start=1):
        targets = library.gestures[gesture_name]
        print(
            f"gesture={gesture_name} "
            f"({gesture_index}/{len(sequence)}) "
            f"targets={_format_targets_summary(targets)}"
        )

        for step_index in range(steps_per_gesture):
            env.set_joint_targets(targets)
            state = env.step()
            _raise_if_unstable(state)

            if (
                step_index == 0
                or (step_index + 1) % print_interval == 0
                or step_index + 1 == steps_per_gesture
            ):
                print(
                    f"  step={step_index + 1:04d}/{steps_per_gesture:04d} "
                    f"t={state.time:.3f}s "
                    f"max_abs_qpos={_max_abs(state.qpos):.4f} "
                    f"max_abs_qvel={_max_abs(state.qvel):.4f}"
                )

            if viewer_handle is not None:
                viewer_handle.sync()
                if viewer_sleep > 0.0:
                    time.sleep(viewer_sleep)

    return state


def _ensure_viewer_can_launch(
    *,
    model_path: Path,
    gesture_config_path: Path,
    sequence_override: Sequence[str] | None,
    steps_per_gesture_override: int | None,
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
        gesture_config_path=gesture_config_path,
        sequence_override=sequence_override,
        steps_per_gesture_override=steps_per_gesture_override,
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
    gesture_config_path: Path,
    sequence_override: Sequence[str] | None,
    steps_per_gesture_override: int | None,
    viewer_sleep: float,
    print_interval: int,
) -> str:
    command = [
        "mjpython",
        "-m",
        "dexvision.apps.check_hand_actuation",
        "--model",
        str(model_path),
        "--gestures",
        str(gesture_config_path),
    ]
    if sequence_override is not None:
        command.append("--sequence")
        command.extend(sequence_override)
    if steps_per_gesture_override is not None:
        command.extend(["--steps-per-gesture", str(steps_per_gesture_override)])
    if viewer_sleep != DEFAULT_VIEWER_SLEEP:
        command.extend(["--viewer-sleep", str(viewer_sleep)])
    if print_interval != DEFAULT_PRINT_INTERVAL:
        command.extend(["--print-interval", str(print_interval)])
    return " ".join(shlex.quote(part) for part in command)


def _format_targets_summary(targets: Mapping[str, float]) -> str:
    return ", ".join(f"{name}={target:.2f}" for name, target in sorted(targets.items()))


def _name_for_id(env: MujocoEnv, object_type: object, object_id: int) -> str:
    name = env._mujoco.mj_id2name(env.model, object_type, object_id)
    if name is None:
        raise MujocoError(f"MuJoCo object id {object_id} is unnamed.")
    return str(name)


def _raise_if_unstable(state: MujocoState) -> None:
    if not np.all(np.isfinite(state.qpos)) or not np.all(np.isfinite(state.qvel)):
        raise MujocoError("Simulation became unstable: non-finite qpos or qvel.")
    max_abs_qpos = _max_abs(state.qpos)
    max_abs_qvel = _max_abs(state.qvel)
    if max_abs_qpos > MAX_ABS_QPOS or max_abs_qvel > MAX_ABS_QVEL:
        raise MujocoError(
            "Simulation became unstable: "
            f"max_abs_qpos={max_abs_qpos:.6f}, max_abs_qvel={max_abs_qvel:.6f}."
        )


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _validate_run_parameters(
    *,
    steps_per_gesture_override: int | None,
    viewer_sleep: float,
    print_interval: int,
) -> None:
    if steps_per_gesture_override is not None and steps_per_gesture_override <= 0:
        raise ValueError("steps_per_gesture must be positive.")
    if viewer_sleep < 0.0:
        raise ValueError("viewer_sleep must be non-negative.")
    if print_interval <= 0:
        raise ValueError("print_interval must be a positive integer.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_hand_actuation(
            model_path=args.model,
            gesture_config_path=args.gestures,
            sequence_override=args.sequence,
            steps_per_gesture_override=args.steps_per_gesture,
            viewer=not args.headless,
            viewer_sleep=args.viewer_sleep,
            print_interval=args.print_interval,
        )
    except (MujocoError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Hand actuation check closed cleanly.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
