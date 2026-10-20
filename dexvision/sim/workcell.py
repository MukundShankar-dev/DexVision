"""MuJoCo workcell: a floating Shadow Hand, six rigid objects, two trays and a wall button.

Honesty guarantees (AGENTS.md section 1):

- After ``reset``, the simulation changes only through actuator ``ctrl`` and the hand mocap
  target. Both are written in exactly one place, ``_apply_inputs``.
- Observations read physical bodies, joints and geoms. The mocap target is reported as
  ``command_*``. It is the controller's own state and never stands in for a measurement.
- The MuJoCo model is never modified after construction.

Action layout, shape [26]:
    [0:3]   hand translation this control step, world frame, metres
    [3:6]   hand rotation this control step, world-frame rotation vector, radians
    [6:26]  absolute targets for the 20 position actuators, in actuator ctrl units
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from dexvision import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "workcell.yaml"
OBJECT_IDS = (
    "block_small",
    "block_large",
    "cylinder_short",
    "cylinder_tall",
    "puck_light",
    "puck_heavy",
)
FINGERS = ("ff", "mf", "rf", "lf", "th")
ACTION_DIM = 26
TRANSLATION = slice(0, 3)
ROTATION = slice(3, 6)
ACTUATOR_TARGETS = slice(6, 26)

TABLE_HALF_EXTENT_XY = (0.19, 0.17)
FLOOR_Z = -0.10
TRAY_CENTERS_XY = ((0.10, 0.11), (0.10, -0.11))
TRAY_FOOTPRINT_RADIUS = 0.054  # circumscribes the 7.6 cm square tray


class WorkcellError(RuntimeError):
    """Invalid configuration, layout or action."""


@dataclass(frozen=True)
class Placement:
    """Reset region for one object: uniform in a box around ``center_xy`` and a yaw range."""

    center_xy: tuple[float, float]
    half_extent_xy: tuple[float, float]
    yaw_range: tuple[float, float] = (-math.pi, math.pi)


@dataclass(frozen=True)
class ObjectState:
    """Measured state of one free object. Vectors are in the world frame."""

    position: np.ndarray  # [3] m
    orientation: np.ndarray  # [4] wxyz
    linear_velocity: np.ndarray  # [3] m/s
    angular_velocity: np.ndarray  # [3] rad/s


@dataclass(frozen=True)
class Observation:
    """One snapshot. Everything except ``command_*`` is measured from the physical model."""

    time: float
    palm_position: np.ndarray  # [3] rh_palm body origin
    palm_orientation: np.ndarray  # [4] wxyz
    grasp_position: np.ndarray  # [3] grasp_site, about 3.5 cm below the palm surface
    hand_qpos: np.ndarray  # [24] wrist and finger joint angles, rad
    hand_qvel: np.ndarray  # [24] rad/s
    fingertip_positions: np.ndarray  # [5, 3] distal pad centres, order FINGERS
    command_position: np.ndarray  # [3] mocap target (controller state)
    command_orientation: np.ndarray  # [4] wxyz mocap target (controller state)
    actuator_targets: np.ndarray  # [20] last commanded actuator targets
    objects: Mapping[str, ObjectState]
    button_depth: float  # m, 0 = released


@dataclass(frozen=True)
class Contact:
    """One active contact between two named bodies, with its normal force in newtons."""

    body1: str
    body2: str
    normal_force: float


class Workcell:
    """Resettable workcell with a 20 Hz action interface. See the module docstring."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        config_path = Path(config_path)
        if not config_path.is_file():
            raise WorkcellError(f"Workcell config not found: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_path = (config_path.parent / config["model"]).resolve()
        if not model_path.is_file():
            raise WorkcellError(f"MuJoCo model not found: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._control = config["control"]
        self._hand = config["hand"]
        self._reset = config["reset"]

        substeps = self._control["control_dt_s"] / self.model.opt.timestep
        if abs(substeps - round(substeps)) > 1e-9:
            raise WorkcellError("control_dt_s must be a multiple of the MuJoCo timestep.")
        self.substeps = int(round(substeps))
        self.control_dt = float(self._control["control_dt_s"])

        m = self.model
        self._target_mocap = int(m.body_mocapid[self._id("body", "hand_target")])
        self._goal_mocap = int(m.body_mocapid[self._id("body", "goal_marker")])
        self._forearm = self._id("body", "rh_forearm")
        self._palm = self._id("body", "rh_palm")
        self._grasp_site = self._id("site", "grasp_site")
        weld = self._id("equality", "hand_weld")
        self._weld_offset = m.eq_data[weld, 3:6].copy()
        self._weld_rotation = m.eq_data[weld, 6:10].copy()
        base_joint = self._id("joint", "rh_base_freejoint")
        self._base_qpos = int(m.jnt_qposadr[base_joint])
        hinge = mujoco.mjtJoint.mjJNT_HINGE
        hand_joints = [
            j
            for j in range(m.njnt)
            if self._name("joint", j).startswith("rh_") and m.jnt_type[j] == hinge
        ]
        self._hand_qpos = np.array([m.jnt_qposadr[j] for j in hand_joints])
        self._hand_dof = np.array([m.jnt_dofadr[j] for j in hand_joints])
        self._tip_geoms = np.array([self._collision_geom(f"rh_{f}distal") for f in FINGERS])
        self._button_qpos = int(m.jnt_qposadr[self._id("joint", "button_joint")])
        self._objects = {name: _ObjectInfo.from_model(self, name) for name in OBJECT_IDS}
        self.ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = m.actuator_ctrlrange[:, 1].copy()
        self.neutral_actuator_targets = np.clip(0.0, self.ctrl_low, self.ctrl_high)
        self._command_position = np.zeros(3)
        self._command_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self._ready = False

    # ------------------------------------------------------------------ public API

    def reset(self, seed: int, layout: Mapping[str, Placement]) -> Observation:
        """Place the hand at neutral, sample object poses for ``seed``, settle physically.

        Objects missing from ``layout`` rest on the floor out of reach.
        """
        unknown = set(layout) - set(OBJECT_IDS)
        if unknown:
            raise WorkcellError(f"Unknown objects in layout: {sorted(unknown)}")
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        position = np.asarray(self._hand["neutral_position_m"], dtype=float)
        orientation = _normalized(self._hand["neutral_orientation_wxyz"])
        self._place_forearm_on_weld(position, orientation)
        self.data.mocap_pos[self._target_mocap] = position
        self.data.mocap_quat[self._target_mocap] = orientation
        self._command_position, self._command_orientation = position, orientation
        self.data.ctrl[:] = self.neutral_actuator_targets

        self._place_objects(rng, layout)
        mujoco.mj_forward(self.model, self.data)
        self._ready = True

        hold = np.concatenate([np.zeros(6), self.neutral_actuator_targets])
        for _ in range(int(round(self._reset["settle_s"] / self.control_dt))):
            self.step(hold)
        return self.observe()

    def step(self, action: np.ndarray) -> Observation:
        """Apply one 26-D action (see module docstring) for one control period."""
        self._require_ready()
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise WorkcellError(f"Action must be finite with shape ({ACTION_DIM},).")
        translation = _clip_norm(action[TRANSLATION], self._control["max_translation_m"])
        rotation = _clip_norm(action[ROTATION], self._control["max_rotation_rad"])
        targets = np.clip(action[ACTUATOR_TARGETS], self.ctrl_low, self.ctrl_high)

        lo = np.asarray(self._control["workspace_min_m"])
        hi = np.asarray(self._control["workspace_max_m"])
        new_position = np.clip(self._command_position + translation, lo, hi)
        new_orientation = _quat_mul(_quat_from_rotvec(rotation), self._command_orientation)
        new_position, new_orientation = self._limit_command_lead(new_position, new_orientation)

        # Interpolate the target across physics substeps so the weld is never jerked.
        start_position, start_orientation = self._command_position, self._command_orientation
        fractions = np.arange(1, self.substeps + 1) / self.substeps
        positions = start_position + fractions[:, None] * (new_position - start_position)
        delta = _rotvec_from_quat(_quat_mul(new_orientation, _quat_conj(start_orientation)))
        orientations = np.array(
            [_quat_mul(_quat_from_rotvec(f * delta), start_orientation) for f in fractions]
        )
        self._apply_inputs(targets, positions, orientations)
        self._command_position, self._command_orientation = new_position, new_orientation
        return self.observe()

    def observe(self) -> Observation:
        """Measure the current state. Calls ``mj_forward``, which never changes the state."""
        self._require_ready()
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        objects = {}
        for name, info in self._objects.items():
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, info.body, velocity, 0)
            objects[name] = ObjectState(
                position=d.xpos[info.body].copy(),
                orientation=d.xquat[info.body].copy(),
                linear_velocity=velocity[3:].copy(),
                angular_velocity=velocity[:3].copy(),
            )
        return Observation(
            time=float(d.time),
            palm_position=d.xpos[self._palm].copy(),
            palm_orientation=d.xquat[self._palm].copy(),
            grasp_position=d.site_xpos[self._grasp_site].copy(),
            hand_qpos=d.qpos[self._hand_qpos].copy(),
            hand_qvel=d.qvel[self._hand_dof].copy(),
            fingertip_positions=d.geom_xpos[self._tip_geoms].copy(),
            command_position=self._command_position.copy(),
            command_orientation=self._command_orientation.copy(),
            actuator_targets=d.ctrl.copy(),
            objects=objects,
            button_depth=float(d.qpos[self._button_qpos]),
        )

    def contacts(self) -> tuple[Contact, ...]:
        """All active contacts between distinct bodies, from the last ``observe``."""
        self._require_ready()
        force = np.zeros(6)
        result = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self._name("body", self.model.geom_bodyid[contact.geom1])
            body2 = self._name("body", self.model.geom_bodyid[contact.geom2])
            if body1 == body2:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, force)
            result.append(Contact(body1, body2, float(force[0])))
        return tuple(result)

    def measured_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the hand target *is*: the measured forearm pose mapped back through the weld.

        Compare with ``command_*`` to see how far the physical hand lags the command.
        """
        forearm_position = self.data.xpos[self._forearm]
        orientation = _quat_mul(self.data.xquat[self._forearm], _quat_conj(self._weld_rotation))
        return forearm_position - _rotate(orientation, self._weld_offset), orientation

    def set_goal_marker(self, position: np.ndarray) -> None:
        """Move the visual-only goal sphere. It has no collision and no mass."""
        self.data.mocap_pos[self._goal_mocap] = np.asarray(position, dtype=float)

    def object_half_height(self, name: str) -> float:
        return self._objects[name].half_height

    # ------------------------------------------------------------------ internals

    def _apply_inputs(
        self,
        actuator_targets: np.ndarray,
        mocap_positions: np.ndarray,
        mocap_orientations: np.ndarray,
    ) -> None:
        """The only place an episode writes to MjData: ctrl, mocap target, mj_step."""
        self.data.ctrl[:] = actuator_targets
        for position, orientation in zip(mocap_positions, mocap_orientations, strict=True):
            self.data.mocap_pos[self._target_mocap] = position
            self.data.mocap_quat[self._target_mocap] = orientation
            mujoco.mj_step(self.model, self.data)

    def _limit_command_lead(
        self, position: np.ndarray, orientation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        measured_position, measured_orientation = self.measured_target_pose()
        position = measured_position + _clip_norm(
            position - measured_position, self._control["max_command_lead_m"]
        )
        lead = _rotvec_from_quat(_quat_mul(orientation, _quat_conj(measured_orientation)))
        lead = _clip_norm(lead, self._control["max_command_lead_rad"])
        return position, _quat_mul(_quat_from_rotvec(lead), measured_orientation)

    def _place_forearm_on_weld(self, position: np.ndarray, orientation: np.ndarray) -> None:
        q = self._base_qpos
        self.data.qpos[q : q + 3] = position + _rotate(orientation, self._weld_offset)
        self.data.qpos[q + 3 : q + 7] = _quat_mul(orientation, self._weld_rotation)

    def _place_objects(self, rng: np.random.Generator, layout: Mapping[str, Placement]) -> None:
        placed: list[tuple[np.ndarray, float]] = [
            (np.asarray(center), TRAY_FOOTPRINT_RADIUS) for center in TRAY_CENTERS_XY
        ]
        clearance = float(self._reset["object_clearance_m"])
        parked = 0
        for name in OBJECT_IDS:
            info = self._objects[name]
            if name not in layout:
                y = self._reset["parking_y_start_m"] + parked * self._reset["parking_spacing_m"]
                xyz = np.array([self._reset["parking_x_m"], y, FLOOR_Z + info.half_height])
                info.set_pose(self.data, xyz, 0.0)
                parked += 1
                continue
            xy, yaw = self._sample_placement(rng, layout[name], info, placed, clearance, name)
            placed.append((xy, info.footprint_radius))
            info.set_pose(self.data, np.array([xy[0], xy[1], info.half_height]), yaw)

    def _sample_placement(self, rng, placement, info, placed, clearance, name):
        center = np.asarray(placement.center_xy, dtype=float)
        half = np.asarray(placement.half_extent_xy, dtype=float)
        limit = np.asarray(TABLE_HALF_EXTENT_XY) - info.footprint_radius
        for _ in range(int(self._reset["max_placement_tries"])):
            xy = center + rng.uniform(-half, half)
            yaw = float(rng.uniform(*placement.yaw_range))
            if np.any(np.abs(xy) > limit):
                continue
            gap = info.footprint_radius + clearance
            if all(np.linalg.norm(xy - c) >= r + gap for c, r in placed):
                return xy, yaw
        raise WorkcellError(f"Could not place '{name}' without overlap; widen its region.")

    def _collision_geom(self, body_name: str) -> int:
        body = self._id("body", body_name)
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == body and self.model.geom_contype[g]:
                return g
        raise WorkcellError(f"Body '{body_name}' has no collision geom.")

    def _id(self, kind: str, name: str) -> int:
        obj = getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}")
        index = mujoco.mj_name2id(self.model, obj, name)
        if index < 0:
            raise WorkcellError(f"MuJoCo model has no {kind} named '{name}'.")
        return int(index)

    def _name(self, kind: str, index: int) -> str:
        obj = getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}")
        return mujoco.mj_id2name(self.model, obj, int(index)) or f"{kind}_{index}"

    def _require_ready(self) -> None:
        if not self._ready:
            raise WorkcellError("Call reset() first.")


@dataclass(frozen=True)
class _ObjectInfo:
    body: int
    qpos: int
    half_height: float
    footprint_radius: float  # yaw-independent radius of the object's table footprint

    @classmethod
    def from_model(cls, workcell: Workcell, name: str) -> _ObjectInfo:
        m = workcell.model
        geom = workcell._id("geom", f"{name}_geom")
        size = m.geom_size[geom]
        if m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            half_height, radius = size[2], math.hypot(size[0], size[1])
        elif m.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half_height, radius = size[1], size[0]
        else:
            raise WorkcellError(f"Unsupported object geom type for '{name}'.")
        qpos = int(m.jnt_qposadr[workcell._id("joint", f"{name}_joint")])
        return cls(workcell._id("body", name), qpos, float(half_height), float(radius))

    def set_pose(self, data: mujoco.MjData, position: np.ndarray, yaw: float) -> None:
        data.qpos[self.qpos : self.qpos + 3] = position
        data.qpos[self.qpos + 3 : self.qpos + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _normalized(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    return q / np.linalg.norm(q)


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector * (limit / norm) if norm > limit else vector


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_negQuat(out, np.asarray(q, dtype=float))
    return out


def _rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    out = np.empty(3)
    mujoco.mju_rotVecQuat(out, np.asarray(vector, dtype=float), np.asarray(q, dtype=float))
    return out


def _quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    out = np.empty(4)
    mujoco.mju_axisAngle2Quat(out, np.asarray(rotvec, dtype=float) / angle, angle)
    return out


def _rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """Shortest rotation vector for ``q`` (angle in [0, pi])."""
    q = np.asarray(q, dtype=float)
    out = np.empty(3)
    mujoco.mju_quat2Vel(out, -q if q[0] < 0 else q, 1.0)
    return out
