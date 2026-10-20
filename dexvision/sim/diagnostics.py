"""Measurements of the workcell itself: how well the hand follows its command, and what it is
physically able to do. These numbers belong in results so physics assumptions stay visible."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from dexvision.sim.workcell import (
    ACTION_DIM,
    ACTUATOR_TARGETS,
    OBJECT_IDS,
    Workcell,
    _quat_conj,
    _quat_mul,
    _rotvec_from_quat,
)


def _hold(workcell: Workcell, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    action = np.zeros(ACTION_DIM)
    action[:3] = translation
    action[ACTUATOR_TARGETS] = workcell.neutral_actuator_targets
    return action


def _tracking_error(workcell: Workcell) -> tuple[float, float]:
    """(position error mm, orientation error deg) between measured hand target and command."""
    measured_position, measured_orientation = workcell.measured_target_pose()
    observation = workcell.observe()
    position = 1000 * np.linalg.norm(measured_position - observation.command_position)
    rotation = _rotvec_from_quat(
        _quat_mul(observation.command_orientation, _quat_conj(measured_orientation))
    )
    return float(position), float(math.degrees(np.linalg.norm(rotation)))


def tracking_report(workcell: Workcell) -> dict:
    """Tracking error at rest and while moving at the maximum commanded speed."""
    workcell.reset(0, {})
    for _ in range(10):
        workcell.step(_hold(workcell))
    rest = _tracking_error(workcell)
    errors = []
    for _ in range(15):
        workcell.step(_hold(workcell, translation=(0.01, 0.0, 0.0)))
        errors.append(_tracking_error(workcell)[0])
    return {
        "rest_position_error_mm": round(rest[0], 2),
        "rest_orientation_error_deg": round(rest[1], 2),
        "moving_position_error_mm_mean": round(float(np.mean(errors[5:])), 2),
        "moving_position_error_mm_max": round(float(np.max(errors)), 2),
        "moving_speed_mps": round(0.01 / workcell.control_dt, 3),
    }


def press_force_report(workcell: Workcell) -> dict:
    """Total hand-table normal force when the command is driven into the table."""
    workcell.reset(0, {})
    for _ in range(60):
        workcell.step(_hold(workcell, translation=(0.0, 0.0, -0.01)))
    workcell.observe()
    forces = [
        c.normal_force for c in workcell.contacts() if "table" in (c.body1, c.body2)
    ]
    return {"hand_on_table_normal_force_n": round(float(sum(forces)), 1)}


def actuator_report(model: mujoco.MjModel) -> list[dict]:
    rows = []
    for a in range(model.nu):
        rows.append(
            {
                "actuator": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a),
                "kp": float(model.actuator_gainprm[a, 0]),
                "force_range": model.actuator_forcerange[a].tolist(),
                "ctrl_range": model.actuator_ctrlrange[a].round(4).tolist(),
            }
        )
    return rows


def object_report(model: mujoco.MjModel) -> list[dict]:
    rows = []
    for name in OBJECT_IDS:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
        body = model.geom_bodyid[geom]
        rows.append(
            {
                "object": name,
                "mass_kg": float(model.body_mass[body]),
                "size_m": model.geom_size[geom].round(4).tolist(),
                "friction_slide_torsion_roll": model.geom_friction[geom].tolist(),
                "condim": int(model.geom_condim[geom]),
            }
        )
    return rows


def full_report(workcell: Workcell) -> dict:
    return {
        "tracking": tracking_report(workcell),
        "press": press_force_report(workcell),
        "hand_mass_kg": round(
            float(
                sum(
                    workcell.model.body_mass[b]
                    for b in range(workcell.model.nbody)
                    if (mujoco.mj_id2name(workcell.model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                    .startswith("rh_")
                )
            ),
            3,
        ),
        "actuators": actuator_report(workcell.model),
        "objects": object_report(workcell.model),
    }
