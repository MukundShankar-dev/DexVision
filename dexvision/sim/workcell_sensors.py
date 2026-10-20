"""Declared ideal simulated encoder/fixture/contact-switch ports for 5.7.

This is explicitly instrumented simulation, not a hardware sensor claim. No
movable object's position, quaternion, velocity or task outcome is exported.
Disabling a port removes it; consumers must reject, never consult truth instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorkcellSensorPacket:
    timestamp: float
    robot_joints: dict | None
    hand_target: tuple | None
    actuator_commands: tuple | None
    fixture_joints: dict | None
    contact_switches: tuple | None


def read_workcell_sensors(env, config, *, disabled=()) -> WorkcellSensorPacket:
    model, data = env.model, env.data
    robot = {}
    for j in range(model.njnt):
        name = model.joint(j).name
        if name != config.scene["hand_base_free_joint"] and not name.startswith("rh_"):
            continue
        kind = int(model.jnt_type[j])
        nq, nv = (7, 6) if kind == 0 else (4, 3) if kind == 1 else (1, 1)
        q, v = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        robot[name] = (tuple(data.qpos[q : q + nq]), tuple(data.qvel[v : v + nv]))
    fixtures = {}
    for fixture in config.fixtures.values():
        joint = model.joint(fixture["joint"])
        fixtures[fixture["joint"]] = (
            float(data.qpos[int(joint.qposadr[0])]),
            float(data.qvel[int(joint.dofadr[0])]),
        )
    mocap = int(model.body(config.scene["hand_base_target"]).mocapid[0])
    target = (tuple(data.mocap_pos[mocap]), tuple(data.mocap_quat[mocap]))
    pairs = set()
    for contact in data.contact[: data.ncon]:
        bodies = tuple(
            int(model.geom_bodyid[int(g)]) for g in (contact.geom1, contact.geom2)
        )
        names = tuple(model.body(b).name for b in bodies)
        if names[0] != names[1]:
            pairs.add(tuple(sorted(names)))
    packet = WorkcellSensorPacket(
        float(data.time),
        robot,
        target,
        tuple(data.ctrl),
        fixtures,
        tuple(sorted(pairs)),
    )
    values = {
        name: (None if name in disabled else getattr(packet, name))
        for name in (
            "robot_joints",
            "hand_target",
            "actuator_commands",
            "fixture_joints",
            "contact_switches",
        )
    }
    return WorkcellSensorPacket(packet.timestamp, **values)


def populate_encoder_state(model, data, config, packet: WorkcellSensorPacket):
    """Populate only explicitly named robot/fixture measurements on a fresh copy."""
    if any(
        getattr(packet, name) is None
        for name in (
            "robot_joints",
            "hand_target",
            "actuator_commands",
            "fixture_joints",
            "contact_switches",
        )
    ):
        raise ValueError(
            "Required declared sensor port unavailable; truth fallback prohibited"
        )
    required = {
        model.joint(j).name
        for j in range(model.njnt)
        if model.joint(j).name == config.scene["hand_base_free_joint"]
        or model.joint(j).name.startswith("rh_")
    }
    if set(packet.robot_joints) != required:
        raise ValueError("Incomplete/extra robot encoder ports")
    for name, (position, velocity) in packet.robot_joints.items():
        if name != config.scene["hand_base_free_joint"] and not name.startswith("rh_"):
            raise ValueError("Object state cannot enter the robot encoder port")
        joint = model.joint(name)
        kind = int(joint.type[0])
        expected = (7, 6) if kind == 0 else (4, 3) if kind == 1 else (1, 1)
        if (len(position), len(velocity)) != expected:
            raise ValueError("Robot encoder shape mismatch")
        q, v = int(joint.qposadr[0]), int(joint.dofadr[0])
        data.qpos[q : q + len(position)] = position
        data.qvel[v : v + len(velocity)] = velocity
    if set(packet.fixture_joints) != {f["joint"] for f in config.fixtures.values()}:
        raise ValueError("Incomplete fixture encoder ports")
    for name, measurement in packet.fixture_joints.items():
        if len(measurement) != 2:
            raise ValueError("Fixture encoder shape mismatch")
        data.qpos[int(model.joint(name).qposadr[0])] = measurement[0]
        data.qvel[int(model.joint(name).dofadr[0])] = measurement[1]
    mocap = int(model.body(config.scene["hand_base_target"]).mocapid[0])
    if (
        len(packet.hand_target) != 2
        or len(packet.hand_target[0]) != 3
        or len(packet.hand_target[1]) != 4
        or len(packet.actuator_commands) != model.nu
    ):
        raise ValueError("Hand target/actuator telemetry shape mismatch")
    data.mocap_pos[mocap], data.mocap_quat[mocap] = packet.hand_target
    data.ctrl[:] = packet.actuator_commands
    data.time = packet.timestamp
    if not all(
        np.isfinite(value).all()
        for value in (
            data.qpos,
            data.qvel,
            data.mocap_pos,
            data.mocap_quat,
            data.ctrl,
        )
    ):
        raise ValueError("Non-finite encoder packet")
