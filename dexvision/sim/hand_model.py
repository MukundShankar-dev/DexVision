"""MuJoCo robot hand model inspection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexvision.sim.mujoco_env import MujocoEnv, MujocoError


@dataclass(frozen=True)
class JointLimit:
    """Limited scalar joint range in MuJoCo coordinate units."""

    minimum: float
    maximum: float


@dataclass(frozen=True)
class HandJointInfo:
    """Named robot hand joint and its limit metadata."""

    name: str
    limit: JointLimit


@dataclass(frozen=True)
class HandActuatorInfo:
    """Named actuator and the joint/control range it drives."""

    name: str
    joint_name: str
    control_range: JointLimit


@dataclass(frozen=True)
class HandModelInfo:
    """Static metadata for a MuJoCo robot hand model."""

    model_path: Path
    joint_count: int
    actuator_count: int
    joints: tuple[HandJointInfo, ...]
    actuators: tuple[HandActuatorInfo, ...]

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Joint names in MuJoCo model order."""

        return tuple(joint.name for joint in self.joints)

    @property
    def actuator_names(self) -> tuple[str, ...]:
        """Actuator names in MuJoCo model order."""

        return tuple(actuator.name for actuator in self.actuators)


@dataclass(frozen=True)
class RestStabilityResult:
    """Headless simulation result for the hand held at default controls."""

    initial_time: float
    final_time: float
    max_abs_qpos: float
    max_abs_qvel: float
    stable: bool


def inspect_hand_model(model_path: str | Path) -> HandModelInfo:
    """Load a MuJoCo hand model and return discoverable joint/actuator metadata."""

    with MujocoEnv(model_path) as env:
        mujoco_module = env._mujoco
        model = env.model
        joints = tuple(
            HandJointInfo(
                name=_name_for_id(mujoco_module, model, mujoco_module.mjtObj.mjOBJ_JOINT, joint_id),
                limit=_joint_limit(model, joint_id),
            )
            for joint_id in range(model.njnt)
        )
        actuators = tuple(
            HandActuatorInfo(
                name=_name_for_id(
                    mujoco_module,
                    model,
                    mujoco_module.mjtObj.mjOBJ_ACTUATOR,
                    actuator_id,
                ),
                joint_name=_actuator_joint_name(mujoco_module, model, actuator_id),
                control_range=_control_range(model, actuator_id),
            )
            for actuator_id in range(model.nu)
        )

        return HandModelInfo(
            model_path=env.model_path,
            joint_count=int(model.njnt),
            actuator_count=int(model.nu),
            joints=joints,
            actuators=actuators,
        )


def check_rest_stability(
    model_path: str | Path,
    *,
    steps: int = 240,
    max_abs_qpos: float = 2.5,
    max_abs_qvel: float = 25.0,
) -> RestStabilityResult:
    """Step the hand at rest and report whether state values remain bounded.

    This is intentionally headless and does not open a MuJoCo viewer.
    """

    if steps <= 0:
        raise ValueError("steps must be a positive integer.")
    if max_abs_qpos <= 0.0:
        raise ValueError("max_abs_qpos must be positive.")
    if max_abs_qvel <= 0.0:
        raise ValueError("max_abs_qvel must be positive.")

    with MujocoEnv(model_path) as env:
        initial = env.reset()
        final = env.step(n_steps=steps)

    qpos_abs = float(np.max(np.abs(final.qpos))) if final.qpos.size else 0.0
    qvel_abs = float(np.max(np.abs(final.qvel))) if final.qvel.size else 0.0
    stable = (
        final.time > initial.time
        and np.all(np.isfinite(final.qpos))
        and np.all(np.isfinite(final.qvel))
        and qpos_abs <= max_abs_qpos
        and qvel_abs <= max_abs_qvel
    )

    return RestStabilityResult(
        initial_time=initial.time,
        final_time=final.time,
        max_abs_qpos=qpos_abs,
        max_abs_qvel=qvel_abs,
        stable=stable,
    )


def format_hand_model_report(info: HandModelInfo, stability: RestStabilityResult | None = None) -> str:
    """Format hand model metadata for CLI output."""

    lines = [
        f"Model: {info.model_path}",
        f"Joints ({info.joint_count}):",
    ]
    lines.extend(
        f"  - {joint.name}: range=[{joint.limit.minimum:.3f}, {joint.limit.maximum:.3f}]"
        for joint in info.joints
    )
    lines.append(f"Actuators ({info.actuator_count}):")
    lines.extend(
        "  - "
        f"{actuator.name}: joint={actuator.joint_name}, "
        f"ctrlrange=[{actuator.control_range.minimum:.3f}, {actuator.control_range.maximum:.3f}]"
        for actuator in info.actuators
    )
    if stability is not None:
        status = "PASS" if stability.stable else "FAIL"
        lines.extend(
            [
                "Rest stability:",
                f"  - status={status}",
                f"  - time={stability.initial_time:.3f}s -> {stability.final_time:.3f}s",
                f"  - max_abs_qpos={stability.max_abs_qpos:.6f}",
                f"  - max_abs_qvel={stability.max_abs_qvel:.6f}",
            ]
        )
    return "\n".join(lines)


def require_controllable_hand(info: HandModelInfo, *, min_joints: int = 10) -> None:
    """Validate that hand metadata contains limited joints and named actuators."""

    if info.joint_count < min_joints:
        raise MujocoError(f"Expected at least {min_joints} hand joints, found {info.joint_count}.")
    if info.actuator_count < min_joints:
        raise MujocoError(
            f"Expected at least {min_joints} hand actuators, found {info.actuator_count}."
        )

    joint_names = set(info.joint_names)
    for joint in info.joints:
        if not joint.name:
            raise MujocoError("Hand model contains an unnamed joint.")
        if joint.limit.minimum >= joint.limit.maximum:
            raise MujocoError(f"Joint '{joint.name}' has invalid limits.")

    for actuator in info.actuators:
        if not actuator.name:
            raise MujocoError("Hand model contains an unnamed actuator.")
        if actuator.joint_name not in joint_names:
            raise MujocoError(
                f"Actuator '{actuator.name}' references unknown joint '{actuator.joint_name}'."
            )
        if actuator.control_range.minimum >= actuator.control_range.maximum:
            raise MujocoError(f"Actuator '{actuator.name}' has invalid control range.")


def _joint_limit(model: object, joint_id: int) -> JointLimit:
    if not bool(model.jnt_limited[joint_id]):
        raise MujocoError(f"Joint at id {joint_id} is missing required limits.")
    return JointLimit(
        minimum=float(model.jnt_range[joint_id, 0]),
        maximum=float(model.jnt_range[joint_id, 1]),
    )


def _control_range(model: object, actuator_id: int) -> JointLimit:
    if not bool(model.actuator_ctrllimited[actuator_id]):
        raise MujocoError(f"Actuator at id {actuator_id} is missing required control limits.")
    return JointLimit(
        minimum=float(model.actuator_ctrlrange[actuator_id, 0]),
        maximum=float(model.actuator_ctrlrange[actuator_id, 1]),
    )


def _actuator_joint_name(mujoco_module: object, model: object, actuator_id: int) -> str:
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    if joint_id < 0:
        raise MujocoError(f"Actuator at id {actuator_id} is not attached to a joint.")
    return _name_for_id(mujoco_module, model, mujoco_module.mjtObj.mjOBJ_JOINT, joint_id)


def _name_for_id(mujoco_module: object, model: object, object_type: object, object_id: int) -> str:
    name = mujoco_module.mj_id2name(model, object_type, object_id)
    if name is None:
        raise MujocoError(f"MuJoCo object id {object_id} is unnamed.")
    return str(name)
