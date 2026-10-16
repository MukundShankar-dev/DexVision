"""Availability-aware shared 5.6 estimates, separate from accepted control state.

Both evaluation truth and RGB use this type. Partial estimates are representable
without fabricated semantic ids, quaternions, contacts, or sensor telemetry.
The unchanged Level 4 control schema remains behind its stricter boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from dexvision.perception.object_observations import (
    INFERRED_PERCEPTION, SIMULATOR_GROUND_TRUTH, ObjectObservationError,
)
from dexvision.perception.object_pose import PoseEstimate
from dexvision.perception.observation_boundary import (
    BODY_ORIGIN, WORLD_FRAME, BoundaryLimits, FieldEvidence, ObjectEvidence,
    WorldEvidence, evidence_from_pose,
)
from dexvision.sim.world_state import EntityRelation, FixtureObservation, RobotObservation, WorldState

ESTIMATE_VERSION = "level5/world-estimate-v1"


@dataclass(frozen=True)
class EntityEstimate:
    """Stable episode-local observation id, distinct from a semantic workcell id."""

    observation_id: str
    evidence: ObjectEvidence

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ObjectObservationError("Estimate needs a nonempty observation id.")
        specs = {"object_id": ("semantic", "id", "semantic_identity"),
                 "class_id": ("semantic", "id", "semantic_identity"),
                 "position": (WORLD_FRAME, "m", BODY_ORIGIN),
                 "orientation_wxyz": (WORLD_FRAME, "wxyz_unit", BODY_ORIGIN),
                 "linear_velocity": (WORLD_FRAME, "m/s", BODY_ORIGIN),
                 "angular_velocity": (WORLD_FRAME, "rad/s", BODY_ORIGIN)}
        for name, expected in specs.items():
            field = getattr(self.evidence, name)
            if (field.frame, field.units, field.body_origin) != expected:
                raise ObjectObservationError("Estimate frame/units/body origin mismatch.")
        for field, length in ((self.evidence.position, 3), (self.evidence.orientation_wxyz, 4),
                              (self.evidence.linear_velocity, 3), (self.evidence.angular_velocity, 3)):
            if field.value is not None:
                if len(field.value) != length or not all(isfinite(v) for v in field.value):
                    raise ObjectObservationError("Invalid estimate vector shape/value.")
        quaternion = self.evidence.orientation_wxyz.value
        if quaternion is not None and abs(sum(v*v for v in quaternion)-1) > 2e-6:
            raise ObjectObservationError("Estimated orientation must be unit wxyz.")

    def require_field(self, name: str, *, now: float, allow_hypothesis: bool = False,
                      limits: BoundaryLimits = BoundaryLimits()):
        """Read an explicitly supported field; hypotheses require opt-in diagnostics.

        This never constructs policy features or a control WorldState. Full
        control consumers must use WorldEstimate.require_control_state instead.
        """
        if name not in {"object_id", "class_id", "position", "orientation_wxyz",
                        "linear_velocity", "angular_velocity"}:
            raise ObjectObservationError(f"Unknown estimated field: {name}")
        field = getattr(self.evidence, name)
        permitted = {"available", "hypothesis"} if allow_hypothesis else {"available"}
        if field.availability not in permitted or not field.valid:
            raise ObjectObservationError(f"{name}: unavailable or uncertified.")
        if field.producer != INFERRED_PERCEPTION:
            raise ObjectObservationError("Evaluation truth is not inferred input.")
        if not isfinite(now) or now < field.timestamp or now-field.timestamp > limits.maximum_age_s:
            raise ObjectObservationError(f"{name}: stale or future.")
        if field.confidence < limits.minimum_confidence:
            raise ObjectObservationError(f"{name}: low confidence.")
        return field.value


@dataclass(frozen=True)
class WorldEstimate:
    """Shared typed estimate with an explicit unavailable world-field sidecar.

    Observation ids have only episode-local visual continuity. A world envelope
    is not an assertion that a full control snapshot or semantic map exists.
    """

    schema_version: str
    timestamp: float
    frame: str
    source: str
    entities: tuple[EntityEstimate, ...]
    robot: FieldEvidence[RobotObservation]
    relations: FieldEvidence[tuple[EntityRelation, ...]]
    fixtures: FieldEvidence[tuple[FixtureObservation, ...]]
    contacts: FieldEvidence[tuple[tuple[str, str], ...]]

    def __post_init__(self) -> None:
        if self.schema_version != ESTIMATE_VERSION or self.frame != WORLD_FRAME:
            raise ObjectObservationError("Unsupported estimate schema/frame.")
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ObjectObservationError("Invalid estimate timestamp.")
        if self.source not in {INFERRED_PERCEPTION, SIMULATOR_GROUND_TRUTH}:
            raise ObjectObservationError("Unsupported estimate producer.")
        if len({e.observation_id for e in self.entities}) != len(self.entities):
            raise ObjectObservationError("Ambiguous duplicate observation ids.")
        fields = [self.robot, self.relations, self.fixtures, self.contacts]
        for field, units, origin in ((self.robot, "m,wxyz_unit", "named_robot_bodies"),
                                      (self.relations, "relation", "named_entities"),
                                      (self.fixtures, "m,bool", "fixture_rest_axis"),
                                      (self.contacts, "id_pairs", "named_entities")):
            if (field.frame, field.units, field.body_origin) != (WORLD_FRAME, units, origin):
                raise ObjectObservationError("World estimate frame/units/body origin mismatch.")
        for entity in self.entities:
            fields.extend(getattr(entity.evidence, name) for name in (
                "object_id", "class_id", "position", "orientation_wxyz",
                "linear_velocity", "angular_velocity"))
        if any(f.producer != self.source or f.timestamp > self.timestamp for f in fields):
            raise ObjectObservationError("Mixed truth/inferred producer or future field.")

    def require_control_state(self, *, now: float, required_motion: tuple[str, ...] = ()) -> WorldState:
        """Keep accepted v1 consumers blocked on unsupported fields or truth."""
        if self.source != INFERRED_PERCEPTION:
            raise ObjectObservationError("Evaluation truth cannot cross inferred control boundary.")
        return WorldEvidence(tuple(e.evidence for e in self.entities), self.robot,
                             self.relations, self.fixtures, self.contacts).require_world_state(
                                 now=now, required_motion=required_motion)


def _world_field(value, source: str, timestamp: float, units: str, origin: str):
    return FieldEvidence(value, "unavailable" if value is None else "available", source,
                         timestamp, 0. if value is None else 1., WORLD_FRAME, units, origin)


def inferred_world_estimate(estimates: tuple[PoseEstimate, ...], timestamp: float) -> WorldEstimate:
    """RGB-only adapter: no truth, task id, geometry lookup or telemetry argument."""
    source = INFERRED_PERCEPTION
    return WorldEstimate(
        ESTIMATE_VERSION, timestamp, WORLD_FRAME, source,
        tuple(EntityEstimate(e.track.track_id, evidence_from_pose(e)) for e in estimates),
        _world_field(None, source, timestamp, "m,wxyz_unit", "named_robot_bodies"),
        _world_field(None, source, timestamp, "relation", "named_entities"),
        _world_field(None, source, timestamp, "m,bool", "fixture_rest_axis"),
        _world_field(None, source, timestamp, "id_pairs", "named_entities"),
    )


def evaluation_world_estimate(world: WorldState) -> WorldEstimate:
    """Evaluation oracle only; shares the typed shape, never used by inference."""
    source = SIMULATOR_GROUND_TRUTH
    entities = []
    for item in world.entities:
        if item.source != source:
            raise ObjectObservationError("Evaluation adapter requires explicitly marked truth.")

        def field(value, frame, units, origin):
            return FieldEvidence(value, "unavailable" if value is None else "available", source,
                                 item.timestamp, item.confidence, frame, units, origin, item.valid)

        evidence = ObjectEvidence(
            field(item.object_id, "semantic", "id", "semantic_identity"),
            field(item.class_id, "semantic", "id", "semantic_identity"),
            field(item.position, WORLD_FRAME, "m", BODY_ORIGIN),
            field(item.orientation_wxyz, WORLD_FRAME, "wxyz_unit", BODY_ORIGIN),
            field(item.linear_velocity, WORLD_FRAME, "m/s", BODY_ORIGIN),
            field(item.angular_velocity, WORLD_FRAME, "rad/s", BODY_ORIGIN),
            "evaluation_only_truth",
        )
        entities.append(EntityEstimate(item.object_id, evidence))
    return WorldEstimate(
        ESTIMATE_VERSION, world.timestamp, world.frame, source, tuple(entities),
        _world_field(world.robot, source, world.timestamp, "m,wxyz_unit", "named_robot_bodies"),
        _world_field(world.relations, source, world.timestamp, "relation", "named_entities"),
        _world_field(world.fixtures, source, world.timestamp, "m,bool", "fixture_rest_axis"),
        _world_field(world.contacts, source, world.timestamp, "id_pairs", "named_entities"),
    )


def world_estimate_from_dict(raw: dict) -> WorldEstimate:
    """Validate serialized truth/inferred envelopes through the same constructors."""
    def field(value: dict, kind: str | None = None):
        payload = value["value"]
        if payload is not None:
            if kind == "robot":
                payload = RobotObservation(**{name: value if name == "safe_neutral" else tuple(value)
                                               for name, value in payload.items()})
            elif kind == "relations":
                payload = tuple(EntityRelation(**r) for r in payload)
            elif kind == "fixtures":
                payload = tuple(FixtureObservation(**f) for f in payload)
            elif kind == "contacts":
                payload = tuple(tuple(p) for p in payload)
            elif isinstance(payload, list):
                payload = tuple(payload)
        return FieldEvidence(**{**value, "value": payload})

    entities = []
    for entity in raw["entities"]:
        evidence = entity["evidence"]
        kwargs = {name: field(evidence[name]) for name in (
            "object_id", "class_id", "position", "orientation_wxyz", "linear_velocity", "angular_velocity")}
        entities.append(EntityEstimate(entity["observation_id"], ObjectEvidence(
            **kwargs, diagnostic_reason=evidence["diagnostic_reason"])))
    return WorldEstimate(raw["schema_version"], raw["timestamp"], raw["frame"], raw["source"], tuple(entities),
                         field(raw["robot"], "robot"), field(raw["relations"], "relations"),
                         field(raw["fixtures"], "fixtures"), field(raw["contacts"], "contacts"))
