"""Append-only review records and pilot discovery for Level 4 collection."""

from __future__ import annotations

import json
import os
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dexvision.sim.workcell import TaskMetricResult, Workcell, WorkcellTask
from dexvision.sim.world_state import WorldState


PILOT_REVIEW_VERSION = "level4/pilot-review-v1"
PILOT_REVIEW_FILENAME = "pilot_review.json"
MANUAL_REPLAY_MANIFEST_VERSION = "level4/manual-replay-manifest-v1"
MANUAL_REPLAY_MANIFEST_FILENAME = "manual_replay_manifest.json"
DEFAULT_LEVEL4_CONFIG = Path("configs/level4_dataset.yaml")
DEFAULT_PILOT_DATASET_DIR = Path("data/demos/level4_pilot")
WORKCELL_PILOT_TASK_ID = "level4_workcell"
WORKCELL_PILOT_SKILLS = (
    "reach_object",
    "pick_object",
    "pick_place_sequence",
    "push_object_to_target",
    "press_button",
)
WORKCELL_PHASES = (
    "approach",
    "acquire",
    "lift",
    "stabilize",
    "transport",
    "place",
    "release",
    "settle",
    "push_contact",
    "fixture_contact",
    "retract",
)
GROUP_BY_PILOT_SKILL = {
    "reach_object": "reach",
    "pick_object": "pick_place",
    "pick_place_sequence": "pick_place",
    "push_object_to_target": "push",
    "press_button": "button",
}
LEVEL4_EPISODE_SOURCES = (
    "scripted",
    "teleoperation",
    "policy_rollout",
    "corrective_intervention",
)
LEVEL4_CORE_GROUPS = ("reach", "push", "button")


class Level4CollectionError(ValueError):
    """Raised when pilot configuration, metadata, or review evidence is invalid."""


@dataclass(frozen=True)
class PilotProtocol:
    """Frozen minimum accepted counts for the Level 4.3 pilot."""

    minimum_genuine_sessions: int
    accepted_by_group: Mapping[str, int]
    optional_dial_decision: str


@dataclass(frozen=True)
class CoreCollectionAssignment:
    """One deterministic Level 4.4 minimum-coverage recording assignment."""

    sequence: int
    coverage_cell_id: str
    data_group: str
    skill_name: str
    source: str
    split: str
    session_slot: str
    repetition: int
    seed: int


@dataclass(frozen=True)
class PilotReview:
    """Immutable post-recording evidence for one Level 4 pilot attempt."""

    episode_id: str
    schema_validation: bool
    timestamp_alignment: bool
    headless_replay: bool
    terminal_metric_recomputation: bool
    recomputed_success: bool
    operator_label_agreement: bool
    quality_thresholds: bool
    coverage_assignment: bool
    split_session_leakage_audit: bool
    expert_accepted: bool
    rejection_reasons: tuple[str, ...] = ()
    version: str = PILOT_REVIEW_VERSION

    @property
    def gates(self) -> Mapping[str, bool]:
        return {
            "schema_validation": self.schema_validation,
            "timestamp_alignment": self.timestamp_alignment,
            "headless_replay": self.headless_replay,
            "terminal_metric_recomputation": self.terminal_metric_recomputation,
            "operator_label_agreement": self.operator_label_agreement,
            "quality_thresholds": self.quality_thresholds,
            "coverage_assignment": self.coverage_assignment,
            "split_session_leakage_audit": self.split_session_leakage_audit,
        }

    def validate(self) -> None:
        if self.version != PILOT_REVIEW_VERSION:
            raise Level4CollectionError(
                f"pilot review version must be {PILOT_REVIEW_VERSION!r}."
            )
        if not self.episode_id.strip():
            raise Level4CollectionError("pilot review episode_id must be non-empty.")
        if any(not reason.strip() for reason in self.rejection_reasons):
            raise Level4CollectionError(
                "pilot rejection reasons must be non-empty strings."
            )
        if self.expert_accepted:
            failed = [name for name, passed in self.gates.items() if not passed]
            if failed:
                raise Level4CollectionError(
                    "expert_accepted review has failed gates: " + ", ".join(failed)
                )
            if not self.recomputed_success:
                raise Level4CollectionError(
                    "expert_accepted review requires recomputed_success=true."
                )
            if self.rejection_reasons:
                raise Level4CollectionError(
                    "expert_accepted review cannot contain rejection reasons."
                )
        elif not self.rejection_reasons:
            raise Level4CollectionError(
                "a non-accepted pilot review must preserve at least one rejection reason."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PilotReview":
        required_booleans = (
            "schema_validation",
            "timestamp_alignment",
            "headless_replay",
            "terminal_metric_recomputation",
            "recomputed_success",
            "operator_label_agreement",
            "quality_thresholds",
            "coverage_assignment",
            "split_session_leakage_audit",
            "expert_accepted",
        )
        for field_name in required_booleans:
            if not isinstance(payload.get(field_name), bool):
                raise Level4CollectionError(
                    f"pilot review {field_name} must be a boolean."
                )
        reasons = payload.get("rejection_reasons", [])
        if isinstance(reasons, str) or not isinstance(reasons, Sequence):
            raise Level4CollectionError(
                "pilot review rejection_reasons must be a sequence."
            )
        if any(not isinstance(reason, str) for reason in reasons):
            raise Level4CollectionError(
                "pilot review rejection_reasons must contain only strings."
            )
        review = cls(
            version=_required_string(payload, "version"),
            episode_id=_required_string(payload, "episode_id"),
            schema_validation=payload["schema_validation"],
            timestamp_alignment=payload["timestamp_alignment"],
            headless_replay=payload["headless_replay"],
            terminal_metric_recomputation=payload["terminal_metric_recomputation"],
            recomputed_success=payload["recomputed_success"],
            operator_label_agreement=payload["operator_label_agreement"],
            quality_thresholds=payload["quality_thresholds"],
            coverage_assignment=payload["coverage_assignment"],
            split_session_leakage_audit=payload["split_session_leakage_audit"],
            expert_accepted=payload["expert_accepted"],
            rejection_reasons=tuple(str(reason) for reason in reasons),
        )
        review.validate()
        return review


@dataclass(frozen=True)
class ManualReplayReview:
    """User-confirmed visible replay evidence, stored outside episode directories."""

    episode_id: str
    verified_skills: tuple[str, ...]
    passed: bool
    notes: str

    def validate(self) -> None:
        if not self.episode_id.strip():
            raise Level4CollectionError("manual replay episode_id must be non-empty.")
        allowed = {
            "reach_object",
            "pick_object",
            "place_held_object",
            "push_object_to_target",
            "press_button",
        }
        if not self.verified_skills or not set(self.verified_skills) <= allowed:
            raise Level4CollectionError(
                "manual replay verified_skills must contain only required Level 4 skills."
            )
        if len(set(self.verified_skills)) != len(self.verified_skills):
            raise Level4CollectionError("manual replay verified_skills must be unique.")
        if not isinstance(self.passed, bool):
            raise Level4CollectionError("manual replay passed must be a boolean.")
        if not self.notes.strip():
            raise Level4CollectionError("manual replay notes must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "episode_id": self.episode_id,
            "verified_skills": list(self.verified_skills),
            "passed": self.passed,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class PilotEpisode:
    """One discovered Level 4 attempt plus its separate review evidence."""

    path: Path
    metadata: Mapping[str, Any]
    review: PilotReview | None
    size_bytes: int
    duration_seconds: float

    @property
    def episode_id(self) -> str:
        return str(self.metadata["episode_id"])

    @property
    def session_id(self) -> str:
        return str(self.metadata["recording_session_id"])

    @property
    def skill_name(self) -> str:
        return str(self.metadata["skill_name"])

    @property
    def goal_condition_id(self) -> str:
        return str(self.metadata["goal_condition_id"])

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    @property
    def expert_accepted(self) -> bool:
        return bool(self.review is not None and self.review.expert_accepted)


@dataclass(frozen=True)
class WorkcellPilotState:
    """Dense, logger-compatible state for one Level 4 workcell sample."""

    world_state: WorldState
    online_phase: str
    task_values: Mapping[str, object]
    qualifies: bool
    dwell_steps: int
    required_dwell_steps: int
    success: bool
    failure_reason: str | None
    object_state: np.ndarray

    def as_task_state(self) -> np.ndarray:
        phase_index = WORKCELL_PHASES.index(self.online_phase)
        numeric = [
            _numeric_metric(value)
            for key, value in sorted(self.task_values.items())
            if key != "terminal_reason" and not key.endswith("dwell_steps")
        ][:3]
        numeric.extend([0.0] * (3 - len(numeric)))
        return np.asarray(
            [
                phase_index,
                int(self.qualifies),
                self.dwell_steps,
                self.required_dwell_steps,
                int(self.success),
                *numeric,
            ],
            dtype=float,
        )

    def as_object_state(self) -> np.ndarray:
        return self.object_state

    @property
    def status_text(self) -> str:
        status = (
            f"phase={self.online_phase} dwell={self.dwell_steps}/"
            f"{self.required_dwell_steps} success={self.success}"
        )
        if "approach_distance_m" in self.task_values:
            status += f" distance={float(self.task_values['approach_distance_m']):.3f}m"
        if "maximum_scene_disturbance_m" in self.task_values:
            status += (
                " disturbance="
                f"{float(self.task_values['maximum_scene_disturbance_m']):.3f}m"
            )
        return status


class WorkcellPilotTask:
    """Adapter that makes frozen workcell tasks recordable by the live app."""

    def __init__(
        self,
        *,
        workcell_config: str | Path,
        dataset_config: str | Path,
        skill_name: str,
        goal_condition_id: str,
        seed: int,
    ) -> None:
        if skill_name not in WORKCELL_PILOT_SKILLS:
            raise Level4CollectionError(
                f"Level 4 pilot skill must be one of {WORKCELL_PILOT_SKILLS}."
            )
        config, _ = load_level4_collection_config(dataset_config)
        self.collection_config = config
        cells = {
            str(cell["id"]): cell
            for cell in config["coverage_cells"]
            if isinstance(cell, Mapping)
        }
        if goal_condition_id not in cells:
            raise Level4CollectionError(
                f"unknown Level 4 goal condition id: {goal_condition_id}"
            )
        self.coverage_cell = dict(cells[goal_condition_id])
        expected_group = GROUP_BY_PILOT_SKILL[skill_name]
        if self.coverage_cell.get("data_group") != expected_group:
            raise Level4CollectionError(
                f"coverage cell {goal_condition_id!r} belongs to "
                f"{self.coverage_cell.get('data_group')!r}, not {expected_group!r}."
            )
        self.skill_name = skill_name
        self.goal_condition_id = goal_condition_id
        self.workcell = Workcell(workcell_config)
        self.env = self.workcell.env
        self.initial_world_state = self.workcell.reset(seed=seed)
        if self.skill_name in {
            "pick_object",
            "pick_place_sequence",
            "push_object_to_target",
        }:
            settings_key = {
                "pick_object": "scripted_grasp",
                "pick_place_sequence": "scripted_place",
                "push_object_to_target": "scripted_push",
            }[self.skill_name]
            setup = config["pilot"][settings_key]["trial_setup"]
            self.initial_world_state = self.workcell.prepare_single_object_trial(
                object_id=str(self.coverage_cell["object_id"]),
                parking_x_m=float(setup["parking_x_m"]),
                parking_y_m=setup["parking_y_m"],
                parking_surface_z_m=float(setup["parking_surface_z_m"]),
            )
            if self.skill_name in {"pick_object", "pick_place_sequence"}:
                contact = config["pilot"]["scripted_grasp"]["contact_dynamics"]
                self.workcell.configure_contact_dynamics(
                    table_condim=int(contact["table_condim"]),
                    family_friction=contact["family_friction"],
                )
        self.goal = self._resolve_goal()
        self._configure_pilot_cue()
        self._phase = "approach"
        self._task: WorkcellTask | None = None
        self._pick_task: WorkcellTask | None = None
        self._place_task: WorkcellTask | None = None
        self._had_object_contact = False
        self._had_button_contact = False
        self._configure_tasks()
        self.current_state = self._snapshot_initial()

    def __enter__(self) -> "WorkcellPilotTask":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.workcell.close()

    def step(self, *, n_steps: int = 1) -> WorkcellPilotState:
        world = self.workcell.step(n_steps=n_steps)
        if self.skill_name == "pick_place_sequence":
            state = self._evaluate_pick_place(world)
        else:
            assert self._task is not None
            result = self._task.evaluate(world)
            state = self._evaluate_single(world, result)
        self.current_state = state
        return state

    def metadata_task_config(self) -> dict[str, Any]:
        object_ids = [self.goal["object_id"]] if "object_id" in self.goal else []
        task_config = {
            "required_objects": object_ids,
            "requires_task_state": True,
            "requires_success_metric_inputs": True,
            "required_observation_fields": ("object_state", "task_state"),
            "parameters": dict(self.goal),
            "viewer_config": dict(self.workcell.config.scene["viewer"]),
            "initial_state": {
                "goal_condition_id": self.goal_condition_id,
                "entity_positions_m": {
                    entity.object_id: list(entity.position)
                    for entity in self.initial_world_state.entities
                },
                "objects": {
                    spec.object_id: {
                        "joint_name": spec.joint,
                        "position_m": list(
                            self.initial_world_state.require_entity(
                                spec.object_id
                            ).position
                        ),
                        "orientation_wxyz": list(
                            self.initial_world_state.require_entity(
                                spec.object_id
                            ).orientation_wxyz
                        ),
                        "linear_velocity_mps": list(
                            self.initial_world_state.require_entity(
                                spec.object_id
                            ).linear_velocity
                            or (0.0, 0.0, 0.0)
                        ),
                        "angular_velocity_radps": list(
                            self.initial_world_state.require_entity(
                                spec.object_id
                            ).angular_velocity
                            or (0.0, 0.0, 0.0)
                        ),
                    }
                    for spec in self.workcell.config.objects
                },
            },
            "task_state_fields": (
                "phase_index",
                "qualifies",
                "dwell_steps",
                "required_dwell_steps",
                "success",
                "metric_0",
                "metric_1",
                "metric_2",
            ),
            "object_state_fields": "six objects x position/quaternion/linear/angular velocity",
        }
        if self.skill_name in {"pick_object", "pick_place_sequence"}:
            contact = self.collection_config["pilot"]["scripted_grasp"][
                "contact_dynamics"
            ]
            family_friction = contact["family_friction"]
            task_config["contact_dynamics"] = {
                "table_geom_name": "workcell_table_geom",
                "table_condim": int(contact["table_condim"]),
                "geom_friction": {
                    spec.geom: list(family_friction[spec.family])
                    for spec in self.workcell.config.objects
                    if spec.family in family_friction
                },
            }
        return task_config

    def _resolve_goal(self) -> dict[str, object]:
        cell = self.coverage_cell
        if self.skill_name == "reach_object":
            entity_id = str(cell["entity_id"])
            entity = self.initial_world_state.require_entity(entity_id)
            position = np.asarray(entity.position, dtype=float)
            if entity_id == "start_button":
                position += np.asarray(
                    self.collection_config["pilot"]["scripted_button"][
                        "precontact_offset_m"
                    ],
                    dtype=float,
                )
            else:
                position[0] = max(-0.16, min(0.20, position[0] - 0.035))
            # This is a collision-free pre-grasp reach, not a contact task. The
            # Shadow Hand hangs well below its logical palm control point, so a
            # low marker among the staged objects causes unavoidable collateral
            # contact before the palm can qualify.
            if entity_id != "start_button":
                position[2] = max(0.14, min(0.23, position[2] + 0.13))
            return {
                "entity_id": entity_id,
                "approach_pose": (
                    *position.tolist(),
                    *self.initial_world_state.robot.base_orientation_wxyz,
                ),
            }
        if self.skill_name == "pick_object":
            return {"object_id": str(cell["object_id"])}
        if self.skill_name == "pick_place_sequence":
            return {
                "object_id": str(cell["object_id"]),
                "target_id": str(cell["target_id"]),
            }
        if self.skill_name == "push_object_to_target":
            return {
                "object_id": str(cell["object_id"]),
                "target_zone": str(cell["target_id"]),
            }
        return {
            "button_id": str(cell["button_id"]),
            "target_press_depth_m": float(cell["target_depth_m"]),
            "target_pressed_state": True,
        }

    def _configure_tasks(self) -> None:
        if self.skill_name == "pick_place_sequence":
            object_id = str(self.goal["object_id"])
            self._pick_task = self.workcell.create_task(
                "pick_object", object_id=object_id
            )
            self._place_task = self.workcell.create_task(
                "place_held_object",
                object_id=object_id,
                target_id=str(self.goal["target_id"]),
            )
            return
        self._task = self.workcell.create_task(self.skill_name, **self.goal)

    def _configure_pilot_cue(self) -> None:
        """Use one consistent source outline and destination cross vocabulary."""

        if self.skill_name == "reach_object":
            entity_id = str(self.goal["entity_id"])
            goal_position = self.goal["approach_pose"][:3]
        elif self.skill_name in {"pick_object", "pick_place_sequence"}:
            entity_id = str(self.goal["object_id"])
            if self.skill_name == "pick_object":
                source = self.initial_world_state.require_entity(entity_id)
                goal_position = (*source.position[:2], source.position[2] + 0.08)
            else:
                target = self.initial_world_state.require_entity(
                    str(self.goal["target_id"])
                )
                goal_position = (
                    *target.position[:2],
                    max(0.06, target.position[2] + 0.05),
                )
        elif self.skill_name == "push_object_to_target":
            entity_id = str(self.goal["object_id"])
            target = self.initial_world_state.require_entity(str(self.goal["target_zone"]))
            goal_position = (*target.position[:2], max(0.04, target.position[2] + 0.03))
        else:
            entity_id = str(self.goal["button_id"])
            target = self.initial_world_state.require_entity(entity_id)
            goal_position = target.position
        self.workcell.set_pilot_task_cue(
            entity_id=entity_id,
            goal_position=goal_position,
        )

    def _snapshot_initial(self) -> WorkcellPilotState:
        return WorkcellPilotState(
            world_state=self.initial_world_state,
            online_phase=self._phase,
            task_values={},
            qualifies=False,
            dwell_steps=0,
            required_dwell_steps=1,
            success=False,
            failure_reason=None,
            object_state=_world_object_vector(self.workcell, self.initial_world_state),
        )

    def _evaluate_single(
        self,
        world: WorldState,
        result: TaskMetricResult,
    ) -> WorkcellPilotState:
        phase_for_sample = self._phase
        if self.skill_name == "pick_object":
            object_id = str(self.goal["object_id"])
            contact = any(
                object_id in pair and any(name.startswith("rh_") for name in pair)
                for pair in world.contacts
            )
            self._had_object_contact = self._had_object_contact or contact
            relation = world.relation_for(object_id)
            height = float(result.values["object_height_above_support_m"])
            if result.success or height >= 0.040:
                self._phase = "stabilize"
            elif relation.held_by == "rh_palm":
                self._phase = "lift"
            elif self._had_object_contact:
                self._phase = "acquire"
        elif self.skill_name == "push_object_to_target":
            object_id = str(self.goal["object_id"])
            contact = any(
                object_id in pair and any(name.startswith("rh_") for name in pair)
                for pair in world.contacts
            )
            self._had_object_contact = self._had_object_contact or contact
            if result.success:
                self._phase = "retract"
            elif result.qualifies:
                self._phase = "settle"
            elif self._had_object_contact:
                self._phase = "push_contact"
        elif self.skill_name == "press_button":
            button_id = str(self.goal["button_id"])
            contact = any(button_id in pair for pair in world.contacts)
            self._had_button_contact = self._had_button_contact or contact
            self._phase = (
                "retract"
                if result.success
                else ("fixture_contact" if self._had_button_contact else "approach")
            )
        else:
            self._phase = "retract" if result.success else "approach"
        return self._state_from_result(world, phase_for_sample, result)

    def _evaluate_pick_place(self, world: WorldState) -> WorkcellPilotState:
        assert self._pick_task is not None and self._place_task is not None
        phase_for_sample = self._phase
        pick = self._pick_task.evaluate(world)
        place = self._place_task.evaluate(world)
        object_id = str(self.goal["object_id"])
        relation = world.relation_for(object_id)
        object_position = world.require_entity(object_id).position
        initial_z = self.initial_world_state.require_entity(object_id).position[2]
        target = world.require_entity(str(self.goal["target_id"]))
        target_distance = math.dist(object_position, target.position)

        if self._phase == "approach" and any(
            object_id in pair and any(name.startswith("rh_") for name in pair)
            for pair in world.contacts
        ):
            self._phase = "acquire"
        elif self._phase == "acquire" and relation.held_by == "rh_palm":
            self._phase = "lift"
        elif self._phase == "lift" and object_position[2] - initial_z >= 0.040:
            self._phase = "stabilize"
        elif self._phase == "stabilize" and pick.success:
            self._phase = "transport"
        elif self._phase == "transport" and target_distance <= 0.025:
            self._phase = "place"
        elif self._phase == "place" and relation.held_by != "rh_palm":
            self._phase = "release"
        elif self._phase == "release":
            self._phase = "settle"
        elif self._phase == "settle" and place.success:
            self._phase = "retract"

        active = (
            place if self._phase in {"place", "release", "settle", "retract"} else pick
        )
        success = bool(place.success and self._phase == "retract")
        return WorkcellPilotState(
            world_state=world,
            online_phase=phase_for_sample,
            task_values={**dict(pick.values), **dict(place.values)},
            qualifies=active.qualifies,
            dwell_steps=active.dwell_steps,
            required_dwell_steps=active.required_dwell_steps,
            success=success,
            failure_reason=_workspace_failure(world, self.workcell),
            object_state=_world_object_vector(self.workcell, world),
        )

    def _state_from_result(
        self,
        world: WorldState,
        phase: str,
        result: TaskMetricResult,
    ) -> WorkcellPilotState:
        return WorkcellPilotState(
            world_state=world,
            online_phase=phase,
            task_values=result.values,
            qualifies=result.qualifies,
            dwell_steps=result.dwell_steps,
            required_dwell_steps=result.required_dwell_steps,
            success=result.success,
            failure_reason=_workspace_failure(world, self.workcell),
            object_state=_world_object_vector(self.workcell, world),
        )


def load_level4_collection_config(
    path: str | Path = DEFAULT_LEVEL4_CONFIG,
) -> tuple[Mapping[str, Any], PilotProtocol]:
    """Load the frozen dataset plan and validate the Level 4.3 pilot block."""

    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Level4CollectionError(
            f"could not read Level 4 config {config_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise Level4CollectionError(
            f"could not parse Level 4 config {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise Level4CollectionError("Level 4 config root must be a mapping.")
    pilot = _mapping(payload, "pilot")
    accepted = _mapping(pilot, "minimum_accepted_episodes")
    required_groups = ("reach", "pick_place", "push", "button")
    counts: dict[str, int] = {}
    for group in required_groups:
        value = accepted.get(group)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Level4CollectionError(
                f"pilot minimum_accepted_episodes.{group} must be a positive integer."
            )
        counts[group] = value
    sessions = pilot.get("minimum_genuine_sessions")
    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 2:
        raise Level4CollectionError(
            "pilot minimum_genuine_sessions must be at least 2."
        )
    optional = _mapping(payload, "optional_skills")
    dial = _mapping(optional, "rotate_dial")
    decision = dial.get("level4_3_decision")
    if decision not in {"promoted", "deferred"}:
        raise Level4CollectionError(
            "optional rotate_dial level4_3_decision must be promoted or deferred."
        )
    coverage_cells = payload.get("coverage_cells")
    if not isinstance(coverage_cells, list) or not coverage_cells:
        raise Level4CollectionError("Level 4 coverage_cells must be a non-empty list.")
    ids = [cell.get("id") for cell in coverage_cells if isinstance(cell, Mapping)]
    if (
        len(ids) != len(coverage_cells)
        or any(not isinstance(cell_id, str) or not cell_id.strip() for cell_id in ids)
        or len(set(ids)) != len(ids)
    ):
        raise Level4CollectionError("Level 4 coverage cell ids must be unique strings.")
    _validate_final_coverage_matrix(payload, coverage_cells=coverage_cells)
    _validate_core_collection_config(payload, coverage_cells=coverage_cells)
    review_filename = pilot.get("expert_acceptance_review_filename")
    if review_filename != PILOT_REVIEW_FILENAME:
        raise Level4CollectionError(
            f"pilot expert_acceptance_review_filename must be {PILOT_REVIEW_FILENAME!r}."
        )
    manual_filename = pilot.get("manual_replay_manifest_filename")
    if manual_filename != MANUAL_REPLAY_MANIFEST_FILENAME:
        raise Level4CollectionError(
            "pilot manual_replay_manifest_filename must be "
            f"{MANUAL_REPLAY_MANIFEST_FILENAME!r}."
        )
    expert_audit = _mapping(pilot, "expert_architecture_audit")
    if expert_audit.get("version") != "level4/expert-replay-audit-v1":
        raise Level4CollectionError(
            "pilot expert_architecture_audit version must be "
            "'level4/expert-replay-audit-v1'."
        )
    audit_repeats = expert_audit.get("minimum_repeats_per_source_skill")
    if (
        isinstance(audit_repeats, bool)
        or not isinstance(audit_repeats, int)
        or audit_repeats < 2
    ):
        raise Level4CollectionError(
            "pilot expert architecture audit requires at least two repeats."
        )
    audit_skills = expert_audit.get("required_source_skills")
    if (
        not isinstance(audit_skills, Sequence)
        or isinstance(audit_skills, str)
        or set(audit_skills) != set(WORKCELL_PILOT_SKILLS)
    ):
        raise Level4CollectionError(
            "pilot expert architecture audit must require every recordable skill."
        )
    return payload, PilotProtocol(sessions, counts, str(decision))


def build_level4_core_collection_plan(
    path: str | Path = DEFAULT_LEVEL4_CONFIG,
) -> tuple[CoreCollectionAssignment, ...]:
    """Expand the frozen Level 4.4 core-cell minima into a stable work list."""

    payload, _ = load_level4_collection_config(path)
    core = _mapping(payload, "level4_4_core_collection")
    slots_by_split = _mapping(core, "session_slots_by_split")
    seed_bases = _mapping(core, "seed_base_by_split")
    seed_overrides = _mapping(core, "seed_override_by_cell")
    split_offsets = {"train": 0, "validation": 0, "test": 0}
    assignments: list[CoreCollectionAssignment] = []
    for raw_cell in payload["coverage_cells"]:
        if not isinstance(raw_cell, Mapping):
            continue
        group = str(raw_cell.get("data_group", ""))
        if group not in LEVEL4_CORE_GROUPS:
            continue
        split = str(raw_cell["split_owner"])
        minima = _mapping(raw_cell, "minimum_accepted_by_split")
        count = int(minima[split])
        raw_slots = slots_by_split[split]
        assert isinstance(raw_slots, Sequence) and not isinstance(raw_slots, str)
        skill_name = {
            "reach": "reach_object",
            "push": "push_object_to_target",
            "button": "press_button",
        }[group]
        for repetition in range(1, count + 1):
            offset = split_offsets[split]
            cell_id = str(raw_cell["id"])
            assignments.append(
                CoreCollectionAssignment(
                    sequence=len(assignments) + 1,
                    coverage_cell_id=cell_id,
                    data_group=group,
                    skill_name=skill_name,
                    source=str(raw_cell["required_source"]),
                    split=split,
                    session_slot=str(raw_slots[offset % len(raw_slots)]),
                    repetition=repetition,
                    seed=int(seed_overrides.get(cell_id, int(seed_bases[split]) + offset)),
                )
            )
            split_offsets[split] += 1
    return tuple(assignments)


def _validate_core_collection_config(
    payload: Mapping[str, Any],
    *,
    coverage_cells: Sequence[Any],
) -> None:
    core = _mapping(payload, "level4_4_core_collection")
    if core.get("version") != "level4/core-collection-v2":
        raise Level4CollectionError(
            "level4_4_core_collection.version must be level4/core-collection-v2."
        )
    groups = core.get("data_groups")
    if list(groups or ()) != list(LEVEL4_CORE_GROUPS):
        raise Level4CollectionError(
            "Level 4.4 core data_groups must be reach, push, and button."
        )
    if core.get("required_source_policy") != "scripted_only":
        raise Level4CollectionError(
            "Level 4.4 core collection must not require teleoperation."
        )
    if (
        core.get("nonrequired_source_attempt_policy")
        != "preserve_and_exclude_without_blocking"
    ):
        raise Level4CollectionError(
            "Level 4.4 must preserve and non-blockingly exclude optional-source attempts."
        )
    core_sources = {
        str(cell.get("required_source"))
        for cell in coverage_cells
        if isinstance(cell, Mapping)
        and cell.get("data_group") in LEVEL4_CORE_GROUPS
    }
    if core_sources != {"scripted"}:
        raise Level4CollectionError(
            "Level 4.4 reach, push, and button cells must all require scripted data."
        )
    expected_total = sum(
        sum(int(value) for value in _mapping(cell, "minimum_accepted_by_split").values())
        for cell in coverage_cells
        if isinstance(cell, Mapping) and cell.get("data_group") in LEVEL4_CORE_GROUPS
    )
    if int(core.get("required_accepted_episodes", -1)) != expected_total:
        raise Level4CollectionError(
            "Level 4.4 required accepted total must match the frozen core cells."
        )
    minimum_sessions = _mapping(core, "minimum_sessions_by_split")
    if minimum_sessions != {"train": 2, "validation": 1, "test": 1}:
        raise Level4CollectionError(
            "Level 4.4 requires two train, one validation, and one test session."
        )
    slots = _mapping(core, "session_slots_by_split")
    for split, minimum in minimum_sessions.items():
        raw = slots.get(split)
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, str)
            or len(raw) < int(minimum)
            or len(set(raw)) != len(raw)
        ):
            raise Level4CollectionError(
                f"Level 4.4 session slots for {split!r} do not meet the minimum."
            )
    seed_bases = _mapping(core, "seed_base_by_split")
    if set(seed_bases) != {"train", "validation", "test"} or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in seed_bases.values()
    ):
        raise Level4CollectionError(
            "Level 4.4 seed bases must be integers for every split."
        )
    seed_overrides = _mapping(core, "seed_override_by_cell")
    core_cells = {
        str(cell["id"]): cell
        for cell in coverage_cells
        if isinstance(cell, Mapping)
        and cell.get("data_group") in LEVEL4_CORE_GROUPS
    }
    for cell_id, seed in seed_overrides.items():
        cell = core_cells.get(str(cell_id))
        if cell is None:
            raise Level4CollectionError(
                f"Level 4.4 seed override names unknown core cell {cell_id!r}."
            )
        minima = _mapping(cell, "minimum_accepted_by_split")
        if sum(int(value) for value in minima.values()) != 1:
            raise Level4CollectionError(
                f"Level 4.4 seed override cell {cell_id!r} must require one episode."
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise Level4CollectionError(
                f"Level 4.4 seed override for {cell_id!r} must be an integer."
            )
    session_fraction = core.get("maximum_single_session_fraction")
    target_delta = core.get("maximum_target_share_delta_from_frozen_minimum")
    if not isinstance(session_fraction, (int, float)) or not 0.0 < float(
        session_fraction
    ) <= 1.0:
        raise Level4CollectionError(
            "Level 4.4 maximum_single_session_fraction must be in (0, 1]."
        )
    if not isinstance(target_delta, (int, float)) or not 0.0 <= float(
        target_delta
    ) <= 1.0:
        raise Level4CollectionError(
            "Level 4.4 target-share delta must be in [0, 1]."
        )


def _validate_final_coverage_matrix(
    payload: Mapping[str, Any],
    *,
    coverage_cells: Sequence[Any],
) -> None:
    """Reject drift between the final cell, source, group, and envelope totals."""

    source_mix = _mapping(payload, "source_mix")
    categories = source_mix.get("categories")
    if list(categories or ()) != list(LEVEL4_EPISODE_SOURCES):
        raise Level4CollectionError(
            "source_mix.categories must preserve the four frozen provenance classes."
        )
    if source_mix.get("provenance_must_remain_separate") is not True:
        raise Level4CollectionError("Level 4 source provenance must remain separate.")
    source_minima = _mapping(source_mix, "minimum_accepted_by_source")
    if set(source_minima) != set(LEVEL4_EPISODE_SOURCES):
        raise Level4CollectionError(
            "source_mix.minimum_accepted_by_source must cover every source category."
        )
    budget = _mapping(payload, "episode_budget")
    groups = _mapping(budget, "groups")
    totals_by_group: Counter[str] = Counter()
    totals_by_source: Counter[str] = Counter()
    for raw_cell in coverage_cells:
        if not isinstance(raw_cell, Mapping):
            raise Level4CollectionError("Level 4 coverage cells must be mappings.")
        cell_id = str(raw_cell["id"])
        group = raw_cell.get("data_group")
        source = raw_cell.get("required_source")
        split_owner = raw_cell.get("split_owner")
        if group not in groups:
            raise Level4CollectionError(
                f"coverage cell {cell_id!r} has unknown data_group {group!r}."
            )
        if source not in LEVEL4_EPISODE_SOURCES:
            raise Level4CollectionError(
                f"coverage cell {cell_id!r} must name one frozen required_source."
            )
        if split_owner not in {"train", "validation", "test"}:
            raise Level4CollectionError(
                f"coverage cell {cell_id!r} has invalid split_owner."
            )
        minima = _mapping(raw_cell, "minimum_accepted_by_split")
        if set(minima) != {"train", "validation", "test"}:
            raise Level4CollectionError(
                f"coverage cell {cell_id!r} must state all three split minima."
            )
        positive: set[str] = set()
        minimum = 0
        for split, value in minima.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Level4CollectionError(
                    f"coverage cell {cell_id!r} split minima must be nonnegative integers."
                )
            minimum += value
            if value:
                positive.add(str(split))
        if positive != {split_owner}:
            raise Level4CollectionError(
                f"coverage cell {cell_id!r} minimum must belong only to its split owner."
            )
        totals_by_group[str(group)] += minimum
        totals_by_source[str(source)] += minimum

    # minimum_new_accepted is the authority; keep this separate from the
    # planning range so malformed YAML cannot silently change a minimum.
    expected_groups = {
        str(group): int(values["minimum_new_accepted"])
        for group, values in groups.items()
        if isinstance(values, Mapping)
    }
    if dict(totals_by_group) != expected_groups:
        raise Level4CollectionError(
            "coverage cell totals do not match episode_budget group minima."
        )
    observed_source_totals = {
        source: int(totals_by_source[source]) for source in LEVEL4_EPISODE_SOURCES
    }
    expected_source_totals = {
        source: int(source_minima[source]) for source in LEVEL4_EPISODE_SOURCES
    }
    if observed_source_totals != expected_source_totals:
        raise Level4CollectionError(
            "coverage cell totals do not match source_mix minima."
        )
    required_total = int(budget["required_total_minimum"])
    planning_maximum = int(budget["required_total_planning_maximum"])
    if sum(totals_by_group.values()) != required_total:
        raise Level4CollectionError(
            "coverage cell totals do not match required_total_minimum."
        )
    planning_sum = sum(
        int(values["planning_range"][1])
        for values in groups.values()
        if isinstance(values, Mapping)
    )
    if planning_sum != planning_maximum or planning_maximum < required_total:
        raise Level4CollectionError(
            "episode budget planning maximum must equal the group planning maxima."
        )

    exclusions = _mapping(payload, "coverage_exclusions").get("cells")
    if not isinstance(exclusions, Sequence) or isinstance(exclusions, str):
        raise Level4CollectionError("coverage_exclusions.cells must be a sequence.")
    excluded_ids = [
        item.get("id") for item in exclusions if isinstance(item, Mapping)
    ]
    if (
        len(excluded_ids) != len(exclusions)
        or any(not isinstance(cell_id, str) or not cell_id for cell_id in excluded_ids)
        or len(set(excluded_ids)) != len(excluded_ids)
        or any(
            not isinstance(item.get("reason"), str) or not item["reason"]
            for item in exclusions
            if isinstance(item, Mapping)
        )
    ):
        raise Level4CollectionError("coverage exclusion ids must be unique strings.")
    if set(excluded_ids) & {str(item["id"]) for item in coverage_cells}:
        raise Level4CollectionError("coverage exclusions cannot also be required cells.")

    storage = _mapping(_mapping(payload, "pilot"), "storage_projection")
    if (
        storage.get("frozen_payload_handling") != "git_lfs"
        or storage.get("working_data_git_policy") != "ignored_never_force_added"
        or storage.get("existing_release_overwrite_allowed") is not False
        or list(storage.get("release_artifacts", ()))
        != ["immutable_tar_gz", "sha256", "manifest"]
    ):
        raise Level4CollectionError(
            "Level 4 storage and immutable release rules must remain frozen."
        )


def discover_pilot_episodes(dataset_dir: str | Path) -> tuple[PilotEpisode, ...]:
    """Discover attempts recursively without editing operator-owned data."""

    root = Path(dataset_dir)
    if not root.exists():
        return ()
    episodes = []
    seen_ids: set[str] = set()
    for metadata_path in sorted(root.rglob("metadata.json")):
        episode_dir = metadata_path.parent
        metadata = _load_json_mapping(metadata_path, label="episode metadata")
        _validate_level4_metadata(metadata, metadata_path=metadata_path)
        episode_id = str(metadata["episode_id"])
        if episode_id in seen_ids:
            raise Level4CollectionError(
                f"duplicate Level 4 episode_id discovered: {episode_id}"
            )
        seen_ids.add(episode_id)
        review_path = episode_dir / PILOT_REVIEW_FILENAME
        review = load_pilot_review(review_path) if review_path.exists() else None
        if review is not None and review.episode_id != episode_id:
            raise Level4CollectionError(
                f"{review_path} episode_id does not match {metadata_path}."
            )
        if (
            review is not None
            and review.expert_accepted
            and metadata.get("success") is not True
        ):
            raise Level4CollectionError(
                f"{review_path} accepts an episode whose operator success label is not true."
            )
        episodes.append(
            PilotEpisode(
                path=episode_dir,
                metadata=metadata,
                review=review,
                size_bytes=_directory_size(episode_dir),
                duration_seconds=_episode_duration_seconds(episode_dir, metadata),
            )
        )
    return tuple(episodes)


def save_pilot_review(
    episode_dir: str | Path,
    review: PilotReview,
) -> Path:
    """Atomically write one review once; accepted episode evidence is append-only."""

    review.validate()
    directory = Path(episode_dir)
    metadata_path = directory / "metadata.json"
    metadata = _load_json_mapping(metadata_path, label="episode metadata")
    _validate_level4_metadata(metadata, metadata_path=metadata_path)
    if metadata["episode_id"] != review.episode_id:
        raise Level4CollectionError(
            "pilot review episode_id does not match episode metadata."
        )
    if review.expert_accepted and metadata.get("success") is not True:
        raise Level4CollectionError(
            "expert acceptance requires an operator success label of true."
        )
    output_path = directory / PILOT_REVIEW_FILENAME
    if output_path.exists():
        raise Level4CollectionError(
            f"pilot review already exists and is append-only: {output_path}"
        )
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(review.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    return output_path


def load_pilot_review(path: str | Path) -> PilotReview:
    """Load and validate one immutable pilot review record."""

    payload = _load_json_mapping(Path(path), label="pilot review")
    return PilotReview.from_mapping(payload)


def append_manual_replay_review(
    dataset_dir: str | Path,
    review: ManualReplayReview,
) -> Path:
    """Append one visible replay result without changing episode review evidence."""

    review.validate()
    path = Path(dataset_dir) / MANUAL_REPLAY_MANIFEST_FILENAME
    if path.exists():
        payload = _load_json_mapping(path, label="manual replay manifest")
        if payload.get("version") != MANUAL_REPLAY_MANIFEST_VERSION:
            raise Level4CollectionError("manual replay manifest version mismatch.")
        raw_reviews = payload.get("reviews")
        if not isinstance(raw_reviews, list):
            raise Level4CollectionError(
                "manual replay manifest reviews must be a list."
            )
        reviews = list(raw_reviews)
    else:
        reviews = []
    if any(
        isinstance(item, Mapping) and item.get("episode_id") == review.episode_id
        for item in reviews
    ):
        raise Level4CollectionError(
            f"manual replay review already exists for episode {review.episode_id!r}."
        )
    reviews.append(review.to_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(
            {"version": MANUAL_REPLAY_MANIFEST_VERSION, "reviews": reviews},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)
    return path


def load_manual_replay_reviews(
    dataset_dir: str | Path,
) -> tuple[ManualReplayReview, ...]:
    """Load user-confirmed manual replay records when the manifest exists."""

    path = Path(dataset_dir) / MANUAL_REPLAY_MANIFEST_FILENAME
    if not path.exists():
        return ()
    payload = _load_json_mapping(path, label="manual replay manifest")
    if payload.get("version") != MANUAL_REPLAY_MANIFEST_VERSION:
        raise Level4CollectionError("manual replay manifest version mismatch.")
    raw_reviews = payload.get("reviews")
    if not isinstance(raw_reviews, list):
        raise Level4CollectionError("manual replay manifest reviews must be a list.")
    reviews: list[ManualReplayReview] = []
    seen: set[str] = set()
    for raw in raw_reviews:
        if not isinstance(raw, Mapping):
            raise Level4CollectionError(
                "manual replay review entries must be mappings."
            )
        skills = raw.get("verified_skills")
        if isinstance(skills, str) or not isinstance(skills, Sequence):
            raise Level4CollectionError(
                "manual replay verified_skills must be a sequence."
            )
        if any(not isinstance(skill, str) for skill in skills):
            raise Level4CollectionError(
                "manual replay verified_skills must contain only strings."
            )
        review = ManualReplayReview(
            episode_id=_required_string(raw, "episode_id"),
            verified_skills=tuple(skills),
            passed=raw.get("passed"),
            notes=_required_string(raw, "notes"),
        )
        review.validate()
        if review.episode_id in seen:
            raise Level4CollectionError(
                f"duplicate manual replay episode_id: {review.episode_id}"
            )
        seen.add(review.episode_id)
        reviews.append(review)
    return tuple(reviews)


def rejection_reason_counts(episodes: Sequence[PilotEpisode]) -> Mapping[str, int]:
    """Count explicit rejection reasons without treating failures as successes."""

    counts: Counter[str] = Counter()
    for episode in episodes:
        if episode.review is not None and not episode.review.expert_accepted:
            counts.update(episode.review.rejection_reasons)
        elif episode.review is None:
            counts["review_missing"] += 1
    return dict(sorted(counts.items()))


def _world_object_vector(workcell: Workcell, state: WorldState) -> np.ndarray:
    values: list[float] = []
    for object_id in workcell.config.object_ids:
        entity = state.require_entity(object_id)
        values.extend(entity.position)
        values.extend(entity.orientation_wxyz)
        values.extend(entity.linear_velocity or (0.0, 0.0, 0.0))
        values.extend(entity.angular_velocity or (0.0, 0.0, 0.0))
    return np.asarray(values, dtype=np.float64)


def _numeric_metric(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _workspace_failure(state: WorldState, workcell: Workcell) -> str | None:
    workspace = workcell.config.requirements["workcell"]["safe_workspace"]
    position = np.asarray(state.robot.base_position, dtype=np.float64)
    if np.any(position < np.asarray(workspace["min"], dtype=np.float64)) or np.any(
        position > np.asarray(workspace["max"], dtype=np.float64)
    ):
        return "workspace_violation"
    return None


def _validate_level4_metadata(
    metadata: Mapping[str, Any], *, metadata_path: Path
) -> None:
    required = (
        "episode_id",
        "recording_session_id",
        "skill_name",
        "goal_condition_id",
    )
    for field_name in required:
        value = metadata.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise Level4CollectionError(
                f"{metadata_path} is missing non-empty Level 4 field {field_name!r}."
            )
    if metadata.get("episode_schema_version") != "level4/episode-v1":
        raise Level4CollectionError(
            f"{metadata_path} does not declare level4/episode-v1."
        )


def _episode_duration_seconds(episode_dir: Path, metadata: Mapping[str, Any]) -> float:
    declared = metadata.get("collection_duration_seconds")
    if isinstance(declared, (int, float)) and not isinstance(declared, bool):
        if declared < 0:
            raise Level4CollectionError(
                "collection_duration_seconds cannot be negative."
            )
        return float(declared)
    timestamp_path = episode_dir / "timestamps.npy"
    if not timestamp_path.exists():
        return 0.0
    try:
        import numpy as np

        timestamps = np.load(timestamp_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise Level4CollectionError(f"could not read {timestamp_path}: {exc}") from exc
    if timestamps.ndim != 1 or not timestamps.size:
        return 0.0
    duration = float(timestamps[-1] - timestamps[0])
    if duration < 0:
        raise Level4CollectionError(f"{timestamp_path} is not monotonic.")
    return duration


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Level4CollectionError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise Level4CollectionError(f"{label} root must be a mapping: {path}")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise Level4CollectionError(f"Level 4 config {key} must be a mapping.")
    return value


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise Level4CollectionError(f"pilot review {key} must be a non-empty string.")
    return value
