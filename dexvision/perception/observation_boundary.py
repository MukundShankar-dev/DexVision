"""Checkpoint 5.6 field evidence around the unchanged Level 4 typed schema.

This offline adapter does not assign semantic ids, certify RGB hypotheses, or
connect to policies. Producers must supply evidence for every required field.
Unavailable optional motion stays None, with its evidence retained separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Generic, TypeVar

from dexvision.perception.object_observations import (
    INFERRED_PERCEPTION,
    ObjectObservation,
    ObjectObservationError,
)
from dexvision.sim.world_state import (
    WORLD_STATE_VERSION,
    EntityRelation,
    FixtureObservation,
    RobotObservation,
    WorldState,
)

if TYPE_CHECKING:
    from dexvision.perception.object_pose import PoseEstimate

T = TypeVar("T")
WORLD_FRAME = "mujoco_world"
BODY_ORIGIN = "object_body_center"


@dataclass(frozen=True)
class FieldEvidence(Generic[T]):
    """One producer's field, acquisition time and explicit availability.

    'hypothesis' retains an unqualified estimate for diagnostics only.
    'unavailable' must carry None; an available empty tuple is a measured set.
    Producer names are declarations, not proof of a deployed sensor or accuracy.
    """

    value: T | None
    availability: str
    producer: str
    timestamp: float
    confidence: float
    frame: str
    units: str
    body_origin: str
    valid: bool = True

    def __post_init__(self) -> None:
        if self.availability not in {"available", "hypothesis", "unavailable"}:
            raise ObjectObservationError("Unknown field availability.")
        if (self.value is None) != (self.availability == "unavailable"):
            raise ObjectObservationError("Unavailable fields must carry None; estimates need values.")
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ObjectObservationError("Field acquisition time must be finite and nonnegative.")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ObjectObservationError("Field confidence must be within [0, 1].")
        if not all((self.producer, self.frame, self.units, self.body_origin)):
            raise ObjectObservationError("Field provenance, frame, units and body origin are required.")


@dataclass(frozen=True)
class BoundaryLimits:
    """Inherited 5.0 delivery limits; no validation/test threshold selection."""

    minimum_confidence: float = .8
    maximum_age_s: float = .1

    def __post_init__(self) -> None:
        if (not isfinite(self.minimum_confidence) or not 0 <= self.minimum_confidence <= 1
                or not isfinite(self.maximum_age_s) or self.maximum_age_s < 0):
            raise ObjectObservationError("Invalid boundary confidence/age limits.")

    def require(self, field: FieldEvidence, name: str, *, now: float,
                producers: set[str], frame: str, units: str, body_origin: str) -> None:
        if field.availability != "available" or not field.valid:
            raise ObjectObservationError(f"{name}: unavailable, uncertified or invalid.")
        if field.producer not in producers:
            raise ObjectObservationError(f"{name}: unsupported producer {field.producer!r}.")
        if (field.frame, field.units, field.body_origin) != (frame, units, body_origin):
            raise ObjectObservationError(f"{name}: frame, units or body origin mismatch.")
        if not isfinite(now) or now < field.timestamp or now-field.timestamp > self.maximum_age_s:
            raise ObjectObservationError(f"{name}: stale or future acquisition time.")
        if field.confidence < self.minimum_confidence:
            raise ObjectObservationError(f"{name}: low confidence.")


@dataclass(frozen=True)
class ObjectEvidence:
    """Object fields, separate from truth and episode-local visual track ids."""

    object_id: FieldEvidence[str]
    class_id: FieldEvidence[str]
    position: FieldEvidence[tuple[float, float, float]]
    orientation_wxyz: FieldEvidence[tuple[float, float, float, float]]
    linear_velocity: FieldEvidence[tuple[float, float, float]]
    angular_velocity: FieldEvidence[tuple[float, float, float]]
    diagnostic_reason: str = ""

    def require_observation(self, *, now: float, limits: BoundaryLimits = BoundaryLimits(),
                            required_motion: tuple[str, ...] = ()) -> ObjectObservation:
        """Use the frozen type only after every required field passes delivery.

        Full semantic identity and orientation are required even for position
        consumers. V8 cannot satisfy these requirements. Motion requirements
        are explicit; optional unavailable/hypothesis motion is never zero-filled.
        """
        if set(required_motion) - {"linear_velocity", "angular_velocity"}:
            raise ObjectObservationError("Unknown required motion field.")
        specs = {
            "object_id": ("semantic", "id", "semantic_identity"),
            "class_id": ("semantic", "id", "semantic_identity"),
            "position": (WORLD_FRAME, "m", BODY_ORIGIN),
            "orientation_wxyz": (WORLD_FRAME, "wxyz_unit", BODY_ORIGIN),
            "linear_velocity": (WORLD_FRAME, "m/s", BODY_ORIGIN),
            "angular_velocity": (WORLD_FRAME, "rad/s", BODY_ORIGIN),
        }
        consumed = []
        for name, (frame, units, origin) in specs.items():
            field = getattr(self, name)
            optional = name in {"linear_velocity", "angular_velocity"}
            if optional and name not in required_motion and field.availability != "available":
                continue
            limits.require(field, name, now=now, producers={INFERRED_PERCEPTION},
                           frame=frame, units=units, body_origin=origin)
            consumed.append(field)
        if not isinstance(self.object_id.value, str) or self.object_id.value.startswith("visual/"):
            raise ObjectObservationError("semantic_identity_unavailable: visual id is not a workcell id.")
        return ObjectObservation(
            object_id=self.object_id.value, class_id=self.class_id.value,
            position=self.position.value, orientation_wxyz=self.orientation_wxyz.value,
            linear_velocity=(self.linear_velocity.value
                             if self.linear_velocity.availability == "available" else None),
            angular_velocity=(self.angular_velocity.value
                              if self.angular_velocity.availability == "available" else None),
            source=INFERRED_PERCEPTION, confidence=min(f.confidence for f in consumed),
            timestamp=min(f.timestamp for f in consumed), frame=WORLD_FRAME,
        )


@dataclass(frozen=True)
class WorldEvidence:
    """Timestamped world fields and object evidence, retained as a sidecar.

    No field comes from evaluation truth. Relation records are required for
    every object so World's unknown-relation default cannot hide missing input.
    This is a synthetic/offline boundary, not a sensor integration or runtime.
    """

    entities: tuple[ObjectEvidence, ...]
    robot: FieldEvidence[RobotObservation]
    relations: FieldEvidence[tuple[EntityRelation, ...]]
    fixtures: FieldEvidence[tuple[FixtureObservation, ...]]
    contacts: FieldEvidence[tuple[tuple[str, str], ...]]

    def require_world_state(self, *, now: float, limits: BoundaryLimits = BoundaryLimits(),
                            required_motion: tuple[str, ...] = ()) -> WorldState:
        specs = {
            "robot": ({"declared_robot_sensor"}, "m,wxyz_unit", "named_robot_bodies"),
            "relations": ({INFERRED_PERCEPTION, "declared_robot_sensor"}, "relation", "named_entities"),
            "fixtures": ({INFERRED_PERCEPTION, "declared_fixture_sensor"}, "m,bool", "fixture_rest_axis"),
            "contacts": ({INFERRED_PERCEPTION, "declared_robot_sensor"}, "id_pairs", "named_entities"),
        }
        for name, (producers, units, origin) in specs.items():
            limits.require(getattr(self, name), name, now=now, producers=producers,
                           frame=WORLD_FRAME, units=units, body_origin=origin)
        if not isinstance(self.robot.value, RobotObservation):
            raise ObjectObservationError("robot: expected typed RobotObservation.")
        if (not isinstance(self.relations.value, tuple)
                or any(not isinstance(r, EntityRelation) for r in self.relations.value)):
            raise ObjectObservationError("relations: expected typed relation tuple.")
        if (not isinstance(self.fixtures.value, tuple)
                or any(not isinstance(f, FixtureObservation) for f in self.fixtures.value)):
            raise ObjectObservationError("fixtures: expected typed fixture tuple.")
        if (not isinstance(self.contacts.value, tuple)
                or any(not isinstance(pair, tuple) or len(pair) != 2
                       or any(not isinstance(item, str) or not item for item in pair)
                       for pair in self.contacts.value)):
            raise ObjectObservationError("contacts: expected named contact pairs.")
        entities = tuple(e.require_observation(now=now, limits=limits, required_motion=required_motion)
                         for e in self.entities)
        if {e.object_id for e in entities} != {r.object_id for r in self.relations.value}:
            raise ObjectObservationError("relations: every entity needs an explicit measured relation record.")
        return WorldState(WORLD_STATE_VERSION, now, WORLD_FRAME, entities,
                          self.relations.value, self.fixtures.value, self.robot.value, self.contacts.value)


def evidence_from_pose(estimate: PoseEstimate) -> ObjectEvidence:
    """Retain frozen V8 RGB output as uncertified evidence, never promote it.

    Only the pose result enters this adapter. Track ids and upright yaw are
    deliberately excluded from semantic id and full orientation. Even a
    caller-supplied pose with an orientation is not certified by this function.
    """
    timestamp = estimate.track.timestamp
    confidence = estimate.track.detection.confidence
    valid = not estimate.track.ambiguous

    def field(value, frame, units, origin):
        return FieldEvidence(value, "unavailable" if value is None else "hypothesis",
                             INFERRED_PERCEPTION, timestamp, confidence,
                             frame, units, origin, valid)

    return ObjectEvidence(
        object_id=field(None, "semantic", "id", "semantic_identity"),
        class_id=field(estimate.track.detection.class_id, "semantic", "id", "semantic_identity"),
        position=field(estimate.position, WORLD_FRAME, "m", BODY_ORIGIN),
        orientation_wxyz=field(None, WORLD_FRAME, "wxyz_unit", BODY_ORIGIN),
        linear_velocity=field(estimate.linear_velocity, WORLD_FRAME, "m/s", BODY_ORIGIN),
        angular_velocity=field(None, WORLD_FRAME, "rad/s", BODY_ORIGIN),
        diagnostic_reason=estimate.rejection_reason or "uncertified_RGB_hypothesis",
    )
