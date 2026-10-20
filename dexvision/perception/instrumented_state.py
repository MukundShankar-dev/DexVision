"""Fuse optical poses and explicitly declared simulated sensor packets.

The estimator owns a fresh model/data copy populated from measurements. It
never copies live simulation data. Only this estimated copy may enter the
versioned place preview; truth remains outside the control API.
"""

from __future__ import annotations

import copy
import time
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dexvision.perception.object_observations import (
    make_object_observation,
    INFERRED_PERCEPTION,
)
from dexvision.sim.workcell_sensors import populate_encoder_state
from dexvision.sim.world_state import EntityRelation


class InstrumentedStateEstimator:
    def __init__(self, calibrated_workcell, optical, required_ids):
        # Model parameters are a declared calibrated simulator model. No MjData
        # or initial WorldState is carried over from the evaluation simulator.
        self.workcell = copy.copy(calibrated_workcell)
        self.workcell._initial_state = None
        self.workcell.env = copy.copy(calibrated_workcell.env)
        self.workcell.env.model = copy.copy(calibrated_workcell.env.model)
        self.workcell.env.data = mujoco.MjData(self.workcell.env.model)
        self.collision_types = self.workcell.env.model.geom_contype.copy()
        self.collision_affinities = self.workcell.env.model.geom_conaffinity.copy()
        self.optical = optical
        self.required_ids = set(required_ids) - set(calibrated_workcell.config.targets)
        self.latest = {}
        self.last_world = None
        self.last_poses = ()

    def observe(self, rgb, packet, *, now):
        started = time.perf_counter()
        if (
            not np.isfinite(now)
            or now < packet.timestamp
            or now - packet.timestamp > 0.1
        ):
            raise ValueError("Stale/future instrumented sensor packet")
        poses = self.optical.infer(rgb, packet.timestamp) if rgb is not None else ()
        now += time.perf_counter() - started
        if now - packet.timestamp > 0.1:
            raise ValueError("Inferred frame exceeds 0.1-second delivery age")
        self.last_poses = poses
        self.latest.update({p.object_id: p for p in poses})
        current = {
            k: p for k, p in self.latest.items() if 0 <= now - p.timestamp <= 0.1
        }
        if self.required_ids - set(current):
            raise ValueError(
                "Missing fresh two-marker identity/pose/motion: "
                + ",".join(sorted(self.required_ids - set(current)))
            )
        env, config = self.workcell.env, self.workcell.config
        unknown = set(config.object_ids) - set(current)
        if any(
            (a in unknown and (b.startswith("rh_") or b in self.required_ids))
            or (b in unknown and (a.startswith("rh_") or a in self.required_ids))
            for a, b in packet.contact_switches or ()
        ):
            raise ValueError(
                "Unobserved movable object entered the instrumented interaction"
            )
        # The original scripted entry parks non-target bodies outside the
        # interaction workspace. Unknown movable poses are NOT filled from model
        # defaults: omit their collision geometry from this explicitly partial
        # estimated preview, and reject unexpected measured interactions above.
        for item in config.objects:
            geoms = env.model.geom_bodyid == env.model.body(item.body).id
            env.model.geom_contype[geoms] = (
                self.collision_types[geoms] if item.object_id in current else 0
            )
            env.model.geom_conaffinity[geoms] = (
                self.collision_affinities[geoms] if item.object_id in current else 0
            )
        mujoco.mj_resetData(env.model, env.data)
        populate_encoder_state(env.model, env.data, config, packet)
        for item in config.objects:
            p = current.get(item.object_id)
            if p is None:
                continue
            joint = env.model.joint(item.joint)
            q, v = int(joint.qposadr[0]), int(joint.dofadr[0])
            env.data.qpos[q : q + 7] = (*p.position, *p.orientation_wxyz)
            rotation = Rotation.from_quat(np.roll(p.orientation_wxyz, -1)).as_matrix()
            env.data.qvel[v : v + 6] = (
                *p.linear_velocity,
                *(rotation.T @ p.angular_velocity),
            )
        mujoco.mj_forward(env.model, env.data)
        predicted = self.workcell.get_world_state()
        by_id = {}
        for original in predicted.entities:
            name = original.object_id
            if name in config.targets:
                # Fixed request-target geometry is calibrated, not a zero-filled
                # unknown movable pose. Its zero motion is a declared invariant.
                by_id[name] = replace(
                    original, source=INFERRED_PERCEPTION, confidence=0.99
                )
            elif name in current:
                p = current[name]
                by_id[name] = make_object_observation(
                    object_id=p.object_id,
                    class_id=original.class_id,
                    position=p.position,
                    orientation_wxyz=p.orientation_wxyz,
                    linear_velocity=p.linear_velocity,
                    angular_velocity=p.angular_velocity,
                    source=INFERRED_PERCEPTION,
                    confidence=0.99,
                    timestamp=p.timestamp,
                    frame=predicted.frame,
                )
        if self.required_ids - set(by_id):
            raise ValueError("Required inferred entity missing")
        # Press depth is an explicit encoder measurement. The accepted press
        # transform derives depth from +X displacement, so fuse that measured
        # coordinate while preserving camera-derived cross-axis alignment/id.
        for fixture_id, runtime in config.fixtures.items():
            if fixture_id in by_id:
                visual = by_id[fixture_id]
                measured = predicted.require_entity(fixture_id)
                position = (
                    measured.position[0],
                    visual.position[1],
                    visual.position[2],
                )
                by_id[fixture_id] = replace(visual, position=position)
        contacts = packet.contact_switches
        relations = tuple(
            self.workcell._relation_for(item, by_id, contacts)
            for item in config.objects
            if item.object_id in by_id
        )
        # Frozen source features use false manipulation flags for mounted
        # fixtures/static targets. Record that known, inapplicable relation
        # explicitly instead of relying on World's unknown-relation default.
        relation_ids = {r.object_id for r in relations}
        relations += tuple(
            EntityRelation(name, None, None, None)
            for name in by_id
            if name not in relation_ids
        )
        self.last_world = replace(
            predicted,
            entities=tuple(by_id.values()),
            relations=relations,
            contacts=contacts,
        )
        return self.last_world

    def preview_task(self, *, skill_name, goal, initial_world):
        if self.last_world is None:
            raise ValueError("Estimated preview requires a complete inferred snapshot")
        return SimpleNamespace(
            env=self.workcell.env,
            workcell=self.workcell,
            skill_name=skill_name,
            goal=goal,
            initial_world_state=initial_world,
        )
