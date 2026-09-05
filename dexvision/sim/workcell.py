"""Resettable Level 4 workcell and executable task-metric contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dexvision.perception.object_observations import (
    SIMULATOR_GROUND_TRUTH,
    make_object_observation,
)
from dexvision.sim.mujoco_env import MujocoEnv, MujocoError
from dexvision.sim.world_state import (
    EntityRelation,
    FixtureObservation,
    RobotObservation,
    WorldState,
    WorldStateError,
)


REQUIRED_SKILLS = (
    "reach_object",
    "pick_object",
    "place_held_object",
    "push_object_to_target",
    "press_button",
)
OPTIONAL_DIAL_SKILL = "rotate_dial"
DEFAULT_WORKCELL_CONFIG = Path("configs/workcell.yaml")


class WorkcellError(RuntimeError):
    """Raised for invalid configuration, resets, ids, or task requests."""


@dataclass(frozen=True)
class WorkcellEntitySpec:
    """Runtime MuJoCo names and footprint metadata for one rigid object."""

    object_id: str
    class_id: str
    family: str
    body: str
    joint: str
    geom: str
    resting_height_m: float
    footprint_radius_m: float
    spawn_anchor_xy_m: tuple[float, float]
    spawn_jitter_xy_m: tuple[float, float]


@dataclass(frozen=True)
class WorkcellConfig:
    """Validated runtime configuration plus the frozen Level 4 authority."""

    version: str
    config_path: Path
    model_path: Path
    requirements_path: Path
    world_state_version: str
    coordinate_frame: str
    length_units: str
    time_units: str
    scene: Mapping[str, Any]
    objects: tuple[WorkcellEntitySpec, ...]
    fixtures: Mapping[str, Mapping[str, Any]]
    targets: Mapping[str, Mapping[str, Any]]
    requirements: Mapping[str, Any]

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(spec.object_id for spec in self.objects)

    @property
    def fixture_ids(self) -> tuple[str, ...]:
        return tuple(self.fixtures)

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(self.targets)


@dataclass(frozen=True)
class TaskMetricResult:
    """One recomputable skill-metric evaluation."""

    skill_name: str
    values: Mapping[str, object]
    qualifies: bool
    dwell_steps: int
    required_dwell_steps: int
    success: bool


class WorkcellTask:
    """Typed goal plus causal dwell tracking for one Level 4 task contract."""

    def __init__(
        self,
        workcell: "Workcell",
        *,
        skill_name: str,
        goal: Mapping[str, object],
        initial_state: WorldState,
    ) -> None:
        self.workcell = workcell
        self.skill_name = skill_name
        self.goal = dict(goal)
        self.initial_state = initial_state
        self.spec = workcell.config.requirements["skills"][skill_name]
        self._dwell_steps = 0

    def evaluate(self, state: WorldState | None = None) -> TaskMetricResult:
        """Compute current metrics and update consecutive qualifying dwell."""

        current = self.workcell.get_world_state() if state is None else state
        values, qualifies = _compute_task_metrics(
            self.skill_name,
            self.goal,
            current,
            self.initial_state,
            self.workcell.config,
        )
        self._dwell_steps = self._dwell_steps + 1 if qualifies else 0
        required = int(self.spec["success_metric"]["required_consecutive_samples"])
        values = dict(values)
        dwell_field = _dwell_field(self.skill_name)
        values[dwell_field] = self._dwell_steps
        return TaskMetricResult(
            skill_name=self.skill_name,
            values=values,
            qualifies=qualifies,
            dwell_steps=self._dwell_steps,
            required_dwell_steps=required,
            success=qualifies and self._dwell_steps >= required,
        )

    def reset_dwell(self) -> None:
        """Reset task-local consecutive-sample state without resetting the scene."""

        self._dwell_steps = 0


class Workcell:
    """One MuJoCo scene containing every required Level 4.1 entity."""

    def __init__(self, config_path: str | Path = DEFAULT_WORKCELL_CONFIG) -> None:
        self.config = load_workcell_config(config_path)
        self.env = MujocoEnv(self.config.model_path)
        self._seed: int | None = None
        self._initial_state: WorldState | None = None
        self._validate_model_names()
        self._configure_operator_visuals()

    def reset(self, *, seed: int) -> WorldState:
        """Deterministically reset all movable entities for ``seed``."""

        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise WorkcellError("Workcell seed must be an integer.")
        self.env.reset()
        rng = np.random.default_rng(int(seed))

        for spec in self.config.objects:
            family_reset = self.config.requirements["reset_ranges"][spec.family]
            position_range = family_reset["position_range_m"]
            minimum = np.asarray(position_range["min"], dtype=np.float64)
            maximum = np.asarray(position_range["max"], dtype=np.float64)
            anchor = np.asarray(spec.spawn_anchor_xy_m, dtype=np.float64)
            jitter_limit = np.asarray(spec.spawn_jitter_xy_m, dtype=np.float64)
            jitter = rng.uniform(-jitter_limit, jitter_limit)
            position = np.array(
                [anchor[0] + jitter[0], anchor[1] + jitter[1], spec.resting_height_m],
                dtype=np.float64,
            )
            if np.any(position < minimum) or np.any(position > maximum):
                raise WorkcellError(
                    f"Reset position for '{spec.object_id}' escaped frozen "
                    f"{spec.family} bounds: {position.tolist()}."
                )
            yaw_min, yaw_max = (float(value) for value in family_reset["yaw_range_rad"])
            yaw = float(rng.uniform(yaw_min, yaw_max))
            quaternion = np.array(
                [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
                dtype=np.float64,
            )
            self._set_free_joint(spec.joint, position, quaternion)

        self._assert_collision_free_layout()
        self._assert_objects_clear_of_setup_slots()
        self._set_scalar_joint(str(self.config.fixtures["start_button"]["joint"]), 0.0)
        self.env.set_mocap_pose(
            str(self.config.scene["hand_base_target"]),
            position=self.config.scene["hand_neutral_position_m"],
            orientation_quat=self.config.scene["hand_neutral_orientation_wxyz"],
        )
        self._align_hand_free_joint_to_weld()
        self.env._mujoco.mj_forward(self.env.model, self.env.data)
        self._seed = int(seed)
        state = self.get_world_state()
        self._initial_state = state
        return state

    def step(self, *, n_steps: int = 1) -> WorldState:
        """Advance the complete scene and return typed simulator truth."""

        self._require_reset()
        self.env.step(n_steps=n_steps)
        return self.get_world_state()

    def prepare_single_object_trial(
        self,
        *,
        object_id: str,
        parking_x_m: float,
        parking_y_m: Sequence[float],
        parking_surface_z_m: float,
    ) -> WorldState:
        """Move non-target objects to stable floor parking for one-object trials."""

        self._require_reset()
        if object_id not in self.config.object_ids:
            raise WorkcellError(f"Unknown single-object trial target: {object_id!r}.")
        parking_y = tuple(float(value) for value in parking_y_m)
        if len(parking_y) != len(self.config.objects) - 1:
            raise WorkcellError(
                "single-object trial parking_y_m must provide one slot per "
                "non-target object."
            )
        if not all(
            math.isfinite(value)
            for value in (float(parking_x_m), float(parking_surface_z_m), *parking_y)
        ):
            raise WorkcellError("single-object trial parking coordinates must be finite.")
        initial = self.get_world_state()
        slots = iter(parking_y)
        for spec in self.config.objects:
            if spec.object_id == object_id:
                continue
            entity = initial.require_entity(spec.object_id)
            self._set_free_joint(
                spec.joint,
                np.asarray(
                    [
                        float(parking_x_m),
                        next(slots),
                        float(parking_surface_z_m) + spec.resting_height_m,
                    ],
                    dtype=np.float64,
                ),
                np.asarray(entity.orientation_wxyz, dtype=np.float64),
            )
        self.env._mujoco.mj_forward(self.env.model, self.env.data)
        state = self.get_world_state()
        self._initial_state = state
        return state

    def set_pilot_goal_marker(self, position: Sequence[float]) -> None:
        """Place the non-colliding operator cue at one resolved task target."""

        self._require_reset()
        marker_position = np.asarray(position, dtype=np.float64)
        if marker_position.shape != (3,) or not np.all(np.isfinite(marker_position)):
            raise WorkcellError("Pilot goal marker position must be a finite 3-vector.")
        self.env.set_mocap_pose(
            str(self.config.scene["pilot_goal_marker"]),
            position=marker_position,
            orientation_quat=(1.0, 0.0, 0.0, 0.0),
        )
        self.env._mujoco.mj_forward(self.env.model, self.env.data)

    def set_hand_base_reset_pose(
        self,
        position: Sequence[float],
        orientation_wxyz: Sequence[float],
    ) -> WorldState:
        """Directly align the welded hand and mocap target for a reset variant."""

        self._require_reset()
        target_position = np.asarray(position, dtype=np.float64)
        target_orientation = np.asarray(orientation_wxyz, dtype=np.float64)
        if target_position.shape != (3,) or not np.all(np.isfinite(target_position)):
            raise WorkcellError("hand base reset position must be a finite 3-vector.")
        if target_orientation.shape != (4,) or not np.all(
            np.isfinite(target_orientation)
        ):
            raise WorkcellError("hand base reset orientation must be finite wxyz.")
        norm = float(np.linalg.norm(target_orientation))
        if norm <= 0.0:
            raise WorkcellError("hand base reset orientation must be non-zero.")
        target_orientation = target_orientation / norm
        self.env.set_mocap_pose(
            str(self.config.scene["hand_base_target"]),
            position=target_position,
            orientation_quat=target_orientation,
        )
        self._align_hand_free_joint_to_target(target_position, target_orientation)
        self.env._mujoco.mj_forward(self.env.model, self.env.data)
        return self.get_world_state()

    def preserve_object_orientation(
        self, object_id: str, orientation_wxyz: Sequence[float]
    ) -> None:
        """Hold one selected object's orientation without constraining translation."""

        self._require_reset()
        try:
            spec = next(item for item in self.config.objects if item.object_id == object_id)
        except StopIteration as exc:
            raise WorkcellError(f"Unknown object orientation target: {object_id!r}.") from exc
        self.env.preserve_free_joint_orientation(spec.joint, orientation_wxyz)

    def configure_contact_dynamics(
        self,
        *,
        table_condim: int,
        family_friction: Mapping[str, Sequence[float | None]],
    ) -> None:
        """Apply task-local table contact dimensions and family friction."""

        self._require_reset()
        geom_friction: dict[str, Sequence[float | None]] = {}
        for spec in self.config.objects:
            raw = family_friction.get(spec.family)
            if raw is None:
                continue
            geom_friction[spec.geom] = raw
        try:
            self.env.configure_contact_dynamics(
                table_geom_name="workcell_table_geom",
                table_condim=table_condim,
                geom_friction=geom_friction,
            )
        except MujocoError as exc:
            raise WorkcellError(str(exc)) from exc

    def set_pilot_task_cue(
        self, *, entity_id: str, goal_position: Sequence[float]
    ) -> None:
        """Outline the selected entity and place a separate task-goal cross."""

        state = self.get_world_state()
        entity = state.require_entity(entity_id)
        outline_position = np.asarray(entity.position, dtype=np.float64)
        self.env.set_mocap_pose(
            str(self.config.scene["pilot_target_outline"]),
            position=outline_position,
            orientation_quat=(1.0, 0.0, 0.0, 0.0),
        )
        self.set_pilot_goal_marker(goal_position)

    def get_world_state(self) -> WorldState:
        """Extract all named entities through the shared observation schema."""

        self._require_reset()
        timestamp = float(self.env.data.time)
        frame = self.config.coordinate_frame
        observations = []
        for spec in self.config.objects:
            observations.append(
                self._body_observation(
                    object_id=spec.object_id,
                    class_id=spec.class_id,
                    body_name=spec.body,
                    timestamp=timestamp,
                    frame=frame,
                    dynamic=True,
                )
            )
        frozen_workcell = self.config.requirements["workcell"]
        for fixture_id, runtime in self.config.fixtures.items():
            class_id = str(frozen_workcell["fixtures"][fixture_id]["fixture_type"])
            observations.append(
                self._body_observation(
                    object_id=fixture_id,
                    class_id=class_id,
                    body_name=str(runtime["body"]),
                    timestamp=timestamp,
                    frame=frame,
                    dynamic=True,
                )
            )
        for target_id, runtime in self.config.targets.items():
            class_id = str(frozen_workcell["targets"][target_id]["target_type"])
            observations.append(
                self._body_observation(
                    object_id=target_id,
                    class_id=class_id,
                    body_name=str(runtime["body"]),
                    timestamp=timestamp,
                    frame=frame,
                    dynamic=False,
                )
            )

        contacts = self._contact_pairs()
        observations_by_id = {item.object_id: item for item in observations}
        relations = tuple(
            self._relation_for(spec, observations_by_id, contacts)
            for spec in self.config.objects
        )
        button_runtime = self.config.fixtures["start_button"]
        button_depth = self._scalar_joint_position(str(button_runtime["joint"]))
        fixtures = (
            FixtureObservation(
                fixture_id="start_button",
                press_depth_m=button_depth,
                pressed=button_depth >= float(button_runtime["pressed_threshold_m"]),
            ),
        )
        robot = self._robot_observation()
        return WorldState(
            schema_version=self.config.world_state_version,
            timestamp=timestamp,
            frame=frame,
            entities=tuple(observations),
            relations=relations,
            fixtures=fixtures,
            robot=robot,
            contacts=contacts,
        )

    def create_task(self, skill_name: str, **goal: object) -> WorkcellTask:
        """Create one of the five frozen tasks after validating its typed goal."""

        self._require_reset()
        if skill_name == OPTIONAL_DIAL_SKILL:
            raise WorkcellError(
                "Optional task 'rotate_dial' is disabled by the frozen Level 4.0 plan."
            )
        if skill_name not in REQUIRED_SKILLS:
            raise WorkcellError(
                f"Unsupported Level 4 workcell skill '{skill_name}'. "
                f"Available skills: {', '.join(REQUIRED_SKILLS)}."
            )
        spec = self.config.requirements["skills"][skill_name]
        _validate_goal(skill_name, goal, spec)
        return WorkcellTask(
            self,
            skill_name=skill_name,
            goal=goal,
            initial_state=self._initial_state_or_raise(),
        )

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "Workcell":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _validate_model_names(self) -> None:
        for spec in self.config.objects:
            self._require_mujoco_name("body", spec.body)
            self._require_mujoco_name("joint", spec.joint)
            self._require_mujoco_name("geom", spec.geom)
        for runtime in self.config.fixtures.values():
            self._require_mujoco_name("body", str(runtime["body"]))
            self._require_mujoco_name("joint", str(runtime["joint"]))
            self._require_mujoco_name("geom", str(runtime["geom"]))
        for runtime in self.config.targets.values():
            self._require_mujoco_name("body", str(runtime["body"]))
            self._require_mujoco_name("geom", str(runtime["geom"]))
        self._require_mujoco_name("body", str(self.config.scene["table_body"]))
        self._require_mujoco_name("body", str(self.config.scene["clearing_region"]))
        self._require_mujoco_name("body", str(self.config.scene["hand_body"]))
        self._require_mujoco_name("body", str(self.config.scene["hand_base_target"]))
        self._require_mujoco_name("body", str(self.config.scene["pilot_goal_marker"]))
        self._require_mujoco_name(
            "body", str(self.config.scene["pilot_target_outline"])
        )
        self._require_mujoco_name("equality", str(self.config.scene["hand_base_weld"]))
        self._require_mujoco_name(
            "joint", str(self.config.scene["hand_base_free_joint"])
        )
        self._require_mujoco_name("site", str(self.config.scene["hand_site"]))
        self._require_mujoco_name("camera", str(self.config.scene["fixed_camera"]))

    def _align_hand_free_joint_to_weld(self) -> None:
        """Start the dynamic hand at the mocap weld pose without a violent transient."""

        target_position = np.asarray(
            self.config.scene["hand_neutral_position_m"], dtype=np.float64
        )
        target_orientation = np.asarray(
            self.config.scene["hand_neutral_orientation_wxyz"], dtype=np.float64
        )
        self._align_hand_free_joint_to_target(target_position, target_orientation)

    def _align_hand_free_joint_to_target(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
    ) -> None:
        """Resolve the weld-relative free-joint pose for one mocap target."""

        equality_id = self._require_mujoco_name(
            "equality", str(self.config.scene["hand_base_weld"])
        )
        relative_pose = np.asarray(
            self.env.model.eq_data[equality_id, 3:10], dtype=np.float64
        )
        rotated_offset = np.empty(3, dtype=np.float64)
        self.env._mujoco.mju_rotVecQuat(
            rotated_offset, relative_pose[:3], target_orientation
        )
        free_joint_orientation = np.empty(4, dtype=np.float64)
        self.env._mujoco.mju_mulQuat(
            free_joint_orientation, target_orientation, relative_pose[3:]
        )
        self._set_free_joint(
            str(self.config.scene["hand_base_free_joint"]),
            target_position + rotated_offset,
            free_joint_orientation,
        )

    def _configure_operator_visuals(self) -> None:
        """Hide the attached control site so only the task target is emphasized."""

        hand_site_id = self._require_mujoco_name(
            "site", str(self.config.scene["hand_site"])
        )
        self.env.model.site_rgba[hand_site_id, 3] = 0.0

    def _require_mujoco_name(self, kind: str, name: str) -> int:
        enum = {
            "body": self.env._mujoco.mjtObj.mjOBJ_BODY,
            "joint": self.env._mujoco.mjtObj.mjOBJ_JOINT,
            "geom": self.env._mujoco.mjtObj.mjOBJ_GEOM,
            "site": self.env._mujoco.mjtObj.mjOBJ_SITE,
            "camera": self.env._mujoco.mjtObj.mjOBJ_CAMERA,
            "equality": self.env._mujoco.mjtObj.mjOBJ_EQUALITY,
        }[kind]
        object_id = int(self.env._mujoco.mj_name2id(self.env.model, enum, name))
        if object_id < 0:
            raise WorkcellError(
                f"Workcell config references unknown MuJoCo {kind} '{name}'."
            )
        return object_id

    def _set_free_joint(
        self,
        joint_name: str,
        position: np.ndarray,
        orientation_wxyz: np.ndarray,
    ) -> None:
        joint_id = self._require_mujoco_name("joint", joint_name)
        qpos_address = int(self.env.model.jnt_qposadr[joint_id])
        dof_address = int(self.env.model.jnt_dofadr[joint_id])
        self.env.data.qpos[qpos_address : qpos_address + 3] = position
        self.env.data.qpos[qpos_address + 3 : qpos_address + 7] = orientation_wxyz
        self.env.data.qvel[dof_address : dof_address + 6] = 0.0

    def _set_scalar_joint(self, joint_name: str, value: float) -> None:
        joint_id = self._require_mujoco_name("joint", joint_name)
        qpos_address = int(self.env.model.jnt_qposadr[joint_id])
        dof_address = int(self.env.model.jnt_dofadr[joint_id])
        self.env.data.qpos[qpos_address] = value
        self.env.data.qvel[dof_address] = 0.0

    def _scalar_joint_position(self, joint_name: str) -> float:
        joint_id = self._require_mujoco_name("joint", joint_name)
        value = float(self.env.data.qpos[int(self.env.model.jnt_qposadr[joint_id])])
        if bool(self.env.model.jnt_limited[joint_id]):
            lower, upper = self.env.model.jnt_range[joint_id]
            value = float(np.clip(value, lower, upper))
        return value

    def _body_observation(
        self,
        *,
        object_id: str,
        class_id: str,
        body_name: str,
        timestamp: float,
        frame: str,
        dynamic: bool,
    ):
        body_id = self._require_mujoco_name("body", body_name)
        velocity = np.zeros(6, dtype=np.float64)
        if dynamic:
            self.env._mujoco.mj_objectVelocity(
                self.env.model,
                self.env.data,
                self.env._mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                velocity,
                0,
            )
        return make_object_observation(
            object_id=object_id,
            class_id=class_id,
            position=self.env.data.xpos[body_id],
            orientation_wxyz=self.env.data.xquat[body_id],
            linear_velocity=velocity[3:] if dynamic else (0.0, 0.0, 0.0),
            angular_velocity=velocity[:3] if dynamic else (0.0, 0.0, 0.0),
            source=SIMULATOR_GROUND_TRUTH,
            confidence=1.0,
            timestamp=timestamp,
            frame=frame,
        )

    def _robot_observation(self) -> RobotObservation:
        base_name = str(self.config.scene["hand_base_target"])
        base_id = self._require_mujoco_name("body", base_name)
        hand_id = self._require_mujoco_name("body", str(self.config.scene["hand_body"]))
        site_id = self._require_mujoco_name("site", str(self.config.scene["hand_site"]))
        neutral = np.asarray(
            self.config.scene["hand_neutral_position_m"], dtype=np.float64
        )
        base_position = np.asarray(self.env.data.xpos[base_id], dtype=np.float64)
        return RobotObservation(
            base_position=tuple(float(value) for value in base_position),
            base_orientation_wxyz=tuple(
                float(value) for value in self.env.data.xquat[base_id]
            ),
            end_effector_position=tuple(
                float(value) for value in self.env.data.site_xpos[site_id]
            ),
            end_effector_orientation_wxyz=tuple(
                float(value) for value in self.env.data.xquat[hand_id]
            ),
            safe_neutral=bool(np.linalg.norm(base_position - neutral) <= 1e-6),
        )

    def _relation_for(
        self,
        spec: WorkcellEntitySpec,
        observations: Mapping[str, Any],
        contacts: tuple[tuple[str, str], ...],
    ) -> EntityRelation:
        observation = observations[spec.object_id]
        contact_entities = {
            right if left == spec.object_id else left
            for left, right in contacts
            if spec.object_id in (left, right)
        }
        hand_contacts = {
            entity for entity in contact_entities if entity.startswith("rh_")
        }
        held_by = "rh_palm" if len(hand_contacts) >= 2 else None
        supported_by = (
            str(self.config.scene["table_body"])
            if (
                str(self.config.scene["table_body"]) in contact_entities
                or observation.position[2] <= spec.resting_height_m + 0.004
            )
            and abs(observation.linear_velocity[2]) <= 0.05
            else None
        )
        receptacle_id = None
        frozen_targets = self.config.requirements["workcell"]["targets"]
        for target_id in self.config.target_ids:
            target = observations[target_id]
            tolerance = float(frozen_targets[target_id]["tolerance_radius_m"])
            if math.dist(observation.position[:2], target.position[:2]) <= tolerance:
                receptacle_id = target_id
                break
        return EntityRelation(
            object_id=spec.object_id,
            supported_by=supported_by,
            held_by=held_by,
            receptacle_id=receptacle_id,
        )

    def _contact_pairs(self) -> tuple[tuple[str, str], ...]:
        pairs: set[tuple[str, str]] = set()
        for index in range(self.env.data.ncon):
            contact = self.env.data.contact[index]
            body_ids = (
                int(self.env.model.geom_bodyid[int(contact.geom1)]),
                int(self.env.model.geom_bodyid[int(contact.geom2)]),
            )
            names = tuple(self._body_name(body_id) for body_id in body_ids)
            if names[0] != names[1]:
                pairs.add(tuple(sorted(names)))
        return tuple(sorted(pairs))

    def _body_name(self, body_id: int) -> str:
        name = self.env._mujoco.mj_id2name(
            self.env.model, self.env._mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        return str(name) if name else f"unnamed_body_{body_id}"

    def _assert_collision_free_layout(self) -> None:
        positions = {}
        for spec in self.config.objects:
            joint_id = self._require_mujoco_name("joint", spec.joint)
            address = int(self.env.model.jnt_qposadr[joint_id])
            positions[spec.object_id] = self.env.data.qpos[address : address + 2].copy()
        clearance = float(self.config.scene["object_clearance_m"])
        for index, first in enumerate(self.config.objects):
            for second in self.config.objects[index + 1 :]:
                distance = float(
                    np.linalg.norm(
                        positions[first.object_id] - positions[second.object_id]
                    )
                )
                minimum = (
                    first.footprint_radius_m + second.footprint_radius_m + clearance
                )
                if distance < minimum:
                    raise WorkcellError(
                        f"Deterministic reset overlaps '{first.object_id}' and "
                        f"'{second.object_id}': {distance:.6f}m < {minimum:.6f}m."
                    )

    def _assert_objects_clear_of_setup_slots(self) -> None:
        frozen_targets = self.config.requirements["workcell"]["targets"]
        slot_radius = float(self.config.scene["setup_slot_visual_radius_m"])
        for spec in self.config.objects:
            joint_id = self._require_mujoco_name("joint", spec.joint)
            address = int(self.env.model.jnt_qposadr[joint_id])
            object_xy = np.asarray(self.env.data.qpos[address : address + 2])
            for target_id in ("setup_slot_a", "setup_slot_b"):
                target_xy = np.asarray(
                    frozen_targets[target_id]["center_m"][:2], dtype=np.float64
                )
                distance = float(np.linalg.norm(object_xy - target_xy))
                minimum = spec.footprint_radius_m + slot_radius
                if distance < minimum:
                    raise WorkcellError(
                        f"Reset places '{spec.object_id}' over '{target_id}': "
                        f"{distance:.6f}m < {minimum:.6f}m."
                    )

    def _require_reset(self) -> None:
        if self._seed is None:
            raise WorkcellError(
                "Call Workcell.reset(seed=...) before reading or stepping."
            )

    def _initial_state_or_raise(self) -> WorldState:
        if self._initial_state is None:
            raise WorkcellError("Call Workcell.reset(seed=...) before creating a task.")
        return self._initial_state


def load_workcell_config(config_path: str | Path) -> WorkcellConfig:
    """Load runtime YAML and cross-check it against the frozen Level 4.0 config."""

    path = Path(config_path)
    if not path.is_file():
        raise WorkcellError(f"Workcell config does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WorkcellError(f"Could not read workcell config '{path}': {exc}") from exc
    if not isinstance(raw, Mapping):
        raise WorkcellError("Workcell YAML root must be a mapping.")
    config_dir = path.resolve().parent
    model_path = _resolve_config_path(config_dir, raw.get("model_path"), "model_path")
    requirements_path = _resolve_config_path(
        config_dir, raw.get("requirements_path"), "requirements_path"
    )
    try:
        requirements = yaml.safe_load(requirements_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WorkcellError(
            f"Could not read frozen Level 4 requirements '{requirements_path}': {exc}"
        ) from exc
    if not isinstance(requirements, Mapping):
        raise WorkcellError("Frozen Level 4 requirements root must be a mapping.")

    scene = _mapping(raw, "scene")
    object_runtime = _mapping(raw, "objects")
    fixtures = _mapping(raw, "fixtures")
    targets = _mapping(raw, "targets")
    frozen_workcell = _mapping(requirements, "workcell")
    frozen_objects = _mapping(frozen_workcell, "objects")
    frozen_fixtures = _mapping(frozen_workcell, "fixtures")
    frozen_targets = _mapping(frozen_workcell, "targets")
    _require_exact_ids("objects", object_runtime, frozen_objects)
    _require_exact_ids("fixtures", fixtures, frozen_fixtures)
    _require_exact_ids("targets", targets, frozen_targets)
    if str(raw.get("world_state_version")) != str(
        _mapping(requirements, "schema_versions")["world_state"]
    ):
        raise WorkcellError("Runtime and frozen world-state schema versions disagree.")
    if str(raw.get("coordinate_frame")) != str(frozen_workcell["coordinate_frame"]):
        raise WorkcellError("Runtime and frozen workcell coordinate frames disagree.")
    if str(raw.get("length_units")) != str(frozen_workcell["length_units"]):
        raise WorkcellError("Runtime and frozen workcell length units disagree.")
    skills = _mapping(requirements, "skills")
    if set(skills) != set(REQUIRED_SKILLS):
        raise WorkcellError(
            "Frozen config does not contain exactly the five required skills."
        )
    optional = _mapping(_mapping(requirements, "optional_skills"), "rotate_dial")
    runtime_optional = _mapping(_mapping(raw, "optional_fixtures"), "mode_dial")
    if bool(optional.get("enabled")) or bool(runtime_optional.get("enabled")):
        raise WorkcellError("mode_dial/rotate_dial must remain disabled in Level 4.1.")

    object_specs = []
    for object_id, runtime_value in object_runtime.items():
        runtime = _as_mapping(runtime_value, f"objects.{object_id}")
        frozen = _as_mapping(frozen_objects[object_id], f"workcell.objects.{object_id}")
        family = str(frozen["family"])
        class_id = str(_mapping(frozen_workcell, "object_families")[family]["class_id"])
        object_specs.append(
            WorkcellEntitySpec(
                object_id=str(object_id),
                class_id=class_id,
                family=family,
                body=str(runtime["body"]),
                joint=str(runtime["joint"]),
                geom=str(runtime["geom"]),
                resting_height_m=float(runtime["resting_height_m"]),
                footprint_radius_m=float(runtime["footprint_radius_m"]),
                spawn_anchor_xy_m=_pair(
                    runtime["spawn_anchor_xy_m"],
                    f"objects.{object_id}.spawn_anchor_xy_m",
                ),
                spawn_jitter_xy_m=_pair(
                    runtime["spawn_jitter_xy_m"],
                    f"objects.{object_id}.spawn_jitter_xy_m",
                ),
            )
        )
    if any(
        spec.resting_height_m <= 0.0 or spec.footprint_radius_m <= 0.0
        for spec in object_specs
    ):
        raise WorkcellError(
            "Object resting heights and footprint radii must be positive."
        )

    return WorkcellConfig(
        version=str(raw.get("version")),
        config_path=path.resolve(),
        model_path=model_path,
        requirements_path=requirements_path,
        world_state_version=str(raw["world_state_version"]),
        coordinate_frame=str(raw["coordinate_frame"]),
        length_units=str(raw["length_units"]),
        time_units=str(raw["time_units"]),
        scene=scene,
        objects=tuple(object_specs),
        fixtures=fixtures,
        targets=targets,
        requirements=requirements,
    )


def create_reach_task(
    workcell: Workcell, *, entity_id: str, approach_pose: Sequence[float]
) -> WorkcellTask:
    return workcell.create_task(
        "reach_object", entity_id=entity_id, approach_pose=tuple(approach_pose)
    )


def create_pick_task(
    workcell: Workcell,
    *,
    object_id: str,
    approach_pose: Sequence[float] | None = None,
) -> WorkcellTask:
    goal: dict[str, object] = {"object_id": object_id}
    if approach_pose is not None:
        goal["approach_pose"] = tuple(approach_pose)
    return workcell.create_task("pick_object", **goal)


def create_place_task(
    workcell: Workcell, *, object_id: str, target_id: str
) -> WorkcellTask:
    return workcell.create_task(
        "place_held_object", object_id=object_id, target_id=target_id
    )


def create_push_task(
    workcell: Workcell, *, object_id: str, target_zone: str
) -> WorkcellTask:
    return workcell.create_task(
        "push_object_to_target", object_id=object_id, target_zone=target_zone
    )


def create_press_task(
    workcell: Workcell,
    *,
    button_id: str = "start_button",
    target_press_depth_m: float = 0.008,
    target_pressed_state: bool = True,
) -> WorkcellTask:
    return workcell.create_task(
        "press_button",
        button_id=button_id,
        target_press_depth_m=target_press_depth_m,
        target_pressed_state=target_pressed_state,
    )


def _compute_task_metrics(
    skill_name: str,
    goal: Mapping[str, object],
    state: WorldState,
    initial: WorldState,
    config: WorkcellConfig,
) -> tuple[dict[str, object], bool]:
    state.require_entity(_goal_entity_id(skill_name, goal))
    conditions = {
        item["field"]: item
        for item in config.requirements["skills"][skill_name]["success_metric"][
            "conditions"
        ]
    }
    if skill_name == "reach_object":
        entity_id = str(goal["entity_id"])
        state.require_entity(entity_id)
        pose = np.asarray(goal["approach_pose"], dtype=np.float64)
        distance = float(
            np.linalg.norm(np.asarray(state.robot.base_position) - pose[:3])
        )
        orientation_error = _quaternion_angle(
            state.robot.base_orientation_wxyz, pose[3:]
        )
        disturbance = max(
            math.dist(
                state.require_entity(object_id).position,
                initial.require_entity(object_id).position,
            )
            for object_id in config.object_ids
        )
        values = {
            "approach_distance_m": distance,
            "approach_orientation_error_rad": orientation_error,
            "maximum_scene_disturbance_m": disturbance,
            "terminal_reason": None,
        }
        qualifies = (
            distance <= float(conditions["approach_distance_m"]["value"])
            and orientation_error
            <= float(conditions["approach_orientation_error_rad"]["value"])
            and disturbance <= float(conditions["maximum_scene_disturbance_m"]["value"])
        )
        return values, qualifies

    if skill_name == "pick_object":
        object_id = str(goal["object_id"])
        observation = state.require_entity(object_id)
        relation = state.relation_for(object_id)
        initial_z = initial.require_entity(object_id).position[2]
        lift_height = observation.position[2] - initial_z
        held_object_id = object_id if relation.held_by == "rh_palm" else None
        values = {
            "held_object_id": held_object_id,
            "object_height_above_support_m": lift_height,
            "terminal_reason": None,
        }
        qualifies = held_object_id == object_id and lift_height >= float(
            conditions["object_height_above_support_m"]["value"]
        )
        return values, qualifies

    if skill_name == "place_held_object":
        object_id = str(goal["object_id"])
        target_id = str(goal["target_id"])
        observation = state.require_entity(object_id)
        target = state.require_entity(target_id)
        relation = state.relation_for(object_id)
        distance = math.dist(observation.position, target.position)
        speed = _speed(observation.linear_velocity)
        angular_speed = _speed(observation.angular_velocity)
        upright_tilt = _upright_tilt_rad(observation.orientation_wxyz)
        supported = relation.supported_by == str(config.scene["table_body"])
        tolerance = float(
            config.requirements["workcell"]["targets"][target_id]["tolerance_radius_m"]
        )
        held_object_id = object_id if relation.held_by == "rh_palm" else None
        inside = distance <= tolerance
        values = {
            "object_to_target_distance_m": distance,
            "object_linear_speed_mps": speed,
            "object_angular_speed_radps": angular_speed,
            "object_upright_tilt_rad": upright_tilt,
            "object_inside_target": inside,
            "object_supported": supported,
            "held_object_id": held_object_id,
            "terminal_reason": None,
        }
        qualifies = (
            distance <= float(conditions["object_to_target_distance_m"]["value"])
            and inside
            and supported
            and held_object_id is None
            and speed <= float(conditions["object_linear_speed_mps"]["value"])
            and angular_speed
            <= float(conditions["object_angular_speed_radps"]["value"])
        )
        return values, qualifies

    if skill_name == "push_object_to_target":
        object_id = str(goal["object_id"])
        target_id = str(goal["target_zone"])
        observation = state.require_entity(object_id)
        relation = state.relation_for(object_id)
        target = state.require_entity(target_id)
        distance = math.dist(observation.position[:2], target.position[:2])
        speed = _speed(observation.linear_velocity)
        supported = relation.supported_by == str(config.scene["table_body"])
        upright_tilt = _upright_tilt_rad(observation.orientation_wxyz)
        board = config.requirements["workcell"]["board_workspace"]
        margin = float(board["safe_edge_margin_m"])
        minimum = np.asarray(board["min_xy_m"], dtype=np.float64) + margin
        maximum = np.asarray(board["max_xy_m"], dtype=np.float64) - margin
        xy = np.asarray(observation.position[:2], dtype=np.float64)
        on_board = bool(np.all(xy >= minimum) and np.all(xy <= maximum))
        values = {
            "planar_object_to_target_distance_m": distance,
            "object_linear_speed_mps": speed,
            "object_on_board": on_board,
            "object_supported": supported,
            "object_upright_tilt_rad": upright_tilt,
            "terminal_reason": None,
        }
        qualifies = (
            distance <= float(conditions["planar_object_to_target_distance_m"]["value"])
            and on_board
            and supported
            and upright_tilt
            <= float(conditions["object_upright_tilt_rad"]["value"])
            and speed <= float(conditions["object_linear_speed_mps"]["value"])
        )
        return values, qualifies

    if skill_name == "press_button":
        button_id = str(goal["button_id"])
        fixture = state.require_fixture(button_id)
        target_depth = float(goal["target_press_depth_m"])
        target_pressed = bool(goal["target_pressed_state"])
        other_depths = [
            item.press_depth_m
            for item in state.fixtures
            if item.fixture_id != button_id
        ]
        other_max = max(other_depths, default=0.0)
        values = {
            "button_id": button_id,
            "press_depth_m": fixture.press_depth_m,
            "button_pressed": fixture.pressed,
            "target_pressed_state": target_pressed,
            "other_button_max_depth_m": other_max,
            "terminal_reason": None,
        }
        qualifies = (
            fixture.press_depth_m >= target_depth
            and fixture.pressed == target_pressed
            and other_max <= float(conditions["other_button_max_depth_m"]["value"])
        )
        return values, qualifies

    raise WorkcellError(f"No metric implementation exists for '{skill_name}'.")


def _validate_goal(
    skill_name: str, goal: Mapping[str, object], spec: Mapping[str, Any]
) -> None:
    fields = _mapping(spec, "goal_fields")
    unexpected = set(goal) - set(fields)
    if unexpected:
        raise WorkcellError(
            f"Task '{skill_name}' received unsupported goal fields: {sorted(unexpected)}."
        )
    missing = [
        name for name, field in fields.items() if field["required"] and name not in goal
    ]
    if missing:
        raise WorkcellError(
            f"Task '{skill_name}' is missing required goal fields: {missing}."
        )
    for name, value in goal.items():
        field = fields[name]
        if "allowed_ids" in field and value not in field["allowed_ids"]:
            raise WorkcellError(
                f"Task '{skill_name}' goal '{name}' has unsupported value {value!r}; "
                f"allowed values: {field['allowed_ids']}."
            )
        field_type = str(field["type"])
        if field_type == "string" and not isinstance(value, str):
            raise WorkcellError(f"Task '{skill_name}' goal '{name}' must be a string.")
        if field_type == "bool" and not isinstance(value, bool):
            raise WorkcellError(f"Task '{skill_name}' goal '{name}' must be a boolean.")
        if field_type == "float64":
            array = np.asarray(value, dtype=np.float64)
            if not np.all(np.isfinite(array)):
                raise WorkcellError(
                    f"Task '{skill_name}' goal '{name}' must contain finite values."
                )
            expected_shape = field["shape"]
            if expected_shape == "scalar":
                if array.shape != ():
                    raise WorkcellError(
                        f"Task '{skill_name}' goal '{name}' must be scalar."
                    )
                numeric_range = field.get("range")
                if numeric_range and not float(numeric_range[0]) <= float(
                    array
                ) <= float(numeric_range[1]):
                    raise WorkcellError(
                        f"Task '{skill_name}' goal '{name}' is outside frozen range "
                        f"{numeric_range}."
                    )
            elif array.shape != tuple(expected_shape):
                raise WorkcellError(
                    f"Task '{skill_name}' goal '{name}' must have shape "
                    f"{tuple(expected_shape)}, got {array.shape}."
                )
            elif array.shape == (7,):
                position_range = field["range"]["position_m"]
                position = array[:3]
                if np.any(position < np.asarray(position_range["min"])) or np.any(
                    position > np.asarray(position_range["max"])
                ):
                    raise WorkcellError(
                        f"Task '{skill_name}' goal '{name}' position is outside "
                        "its frozen range."
                    )
                quaternion = array[3:]
                component_range = field["range"]["quaternion_component"]
                if np.any(quaternion < float(component_range[0])) or np.any(
                    quaternion > float(component_range[1])
                ):
                    raise WorkcellError(
                        f"Task '{skill_name}' goal '{name}' quaternion components "
                        "are outside the frozen range."
                    )
                if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6):
                    raise WorkcellError(
                        f"Task '{skill_name}' goal '{name}' must contain a unit "
                        "wxyz quaternion."
                    )
    if skill_name == "push_object_to_target":
        object_id = str(goal["object_id"])
        object_spec = spec["goal_fields"]["object_id"]
        if object_id not in object_spec["allowed_ids"]:
            raise WorkcellError(f"Object '{object_id}' is not push-compatible.")


def _goal_entity_id(skill_name: str, goal: Mapping[str, object]) -> str:
    if skill_name == "reach_object":
        return str(goal["entity_id"])
    if skill_name == "press_button":
        return str(goal["button_id"])
    return str(goal["object_id"])


def _dwell_field(skill_name: str) -> str:
    if skill_name == "pick_object":
        return "held_dwell_steps"
    if skill_name == "place_held_object":
        return "settle_steps"
    return "dwell_steps"


def _quaternion_angle(first: Sequence[float], second: Sequence[float]) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    second_norm = float(np.linalg.norm(second_array))
    if second_array.shape != (4,) or second_norm <= 0.0:
        raise WorkcellError("Approach orientation must be a non-zero wxyz quaternion.")
    dot = float(np.dot(first_array, second_array / second_norm))
    return 2.0 * math.acos(float(np.clip(abs(dot), 0.0, 1.0)))


def _upright_tilt_rad(quaternion_wxyz: Sequence[float]) -> float:
    """Return the angle between an object's local up axis and world up."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not math.isfinite(norm) or norm <= 0.0:
        raise WorldStateError("Object orientation must be a finite quaternion.")
    _, x, y, _ = quaternion / norm
    world_up_dot = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(world_up_dot, -1.0, 1.0)))


def _speed(velocity: Sequence[float] | None) -> float:
    if velocity is None:
        raise WorldStateError("Task metric requires object linear velocity.")
    return float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)))


def _resolve_config_path(config_dir: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise WorkcellError(f"Workcell config field '{field}' must be a path string.")
    path = Path(value)
    resolved = path if path.is_absolute() else config_dir / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise WorkcellError(f"Workcell {field} does not exist: {resolved}")
    return resolved


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in parent:
        raise WorkcellError(f"Missing required configuration mapping '{key}'.")
    return _as_mapping(parent[key], key)


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkcellError(f"Configuration field '{name}' must be a mapping.")
    return value


def _require_exact_ids(
    category: str, runtime: Mapping[str, Any], frozen: Mapping[str, Any]
) -> None:
    if set(runtime) != set(frozen):
        raise WorkcellError(
            f"Runtime and frozen {category} ids disagree: "
            f"runtime={sorted(runtime)}, frozen={sorted(frozen)}."
        )


def _pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise WorkcellError(f"Configuration field '{name}' must contain two values.")
    pair = tuple(float(item) for item in value)
    if not all(np.isfinite(item) for item in pair):
        raise WorkcellError(f"Configuration field '{name}' must be finite.")
    return pair
