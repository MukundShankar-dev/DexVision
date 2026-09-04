"""Typed Level 4 workcell state shared across tasks and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, sqrt

from dexvision.perception.object_observations import (
    ObjectObservation,
    ObjectObservationError,
)


WORLD_STATE_VERSION = "level4/world-state-v1"


class WorldStateError(ValueError):
    """Raised when a world-state snapshot is incomplete or ambiguous."""


@dataclass(frozen=True)
class EntityRelation:
    """Known physical relationships for one rigid object."""

    object_id: str
    supported_by: str | None
    held_by: str | None
    receptacle_id: str | None


@dataclass(frozen=True)
class FixtureObservation:
    """Scalar state for one named workcell fixture."""

    fixture_id: str
    press_depth_m: float
    pressed: bool

    def __post_init__(self) -> None:
        if not self.fixture_id:
            raise WorldStateError("fixture_id must be non-empty.")
        if not isfinite(self.press_depth_m) or self.press_depth_m < 0.0:
            raise WorldStateError(
                f"Fixture '{self.fixture_id}' press_depth_m must be finite and non-negative."
            )


@dataclass(frozen=True)
class RobotObservation:
    """Robot pose needed by the Level 4.1 task contracts."""

    base_position: tuple[float, float, float]
    base_orientation_wxyz: tuple[float, float, float, float]
    end_effector_position: tuple[float, float, float]
    end_effector_orientation_wxyz: tuple[float, float, float, float]
    safe_neutral: bool

    def __post_init__(self) -> None:
        _finite_vector(self.base_position, 3, "robot base_position")
        _unit_quaternion(self.base_orientation_wxyz, "robot base_orientation_wxyz")
        _finite_vector(self.end_effector_position, 3, "robot end_effector_position")
        _unit_quaternion(
            self.end_effector_orientation_wxyz,
            "robot end_effector_orientation_wxyz",
        )


@dataclass(frozen=True)
class WorldState:
    """One immutable, timestamped workcell snapshot."""

    schema_version: str
    timestamp: float
    frame: str
    entities: tuple[ObjectObservation, ...]
    relations: tuple[EntityRelation, ...]
    fixtures: tuple[FixtureObservation, ...]
    robot: RobotObservation
    contacts: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != WORLD_STATE_VERSION:
            raise WorldStateError(
                f"Unsupported world-state schema '{self.schema_version}'; "
                f"expected '{WORLD_STATE_VERSION}'."
            )
        if not isfinite(self.timestamp) or self.timestamp < 0.0:
            raise WorldStateError("World-state timestamp must be finite and non-negative.")
        if not self.frame:
            raise WorldStateError("World-state coordinate frame must be explicit.")
        entity_ids = [entity.object_id for entity in self.entities]
        duplicate_entities = _duplicates(entity_ids)
        if duplicate_entities:
            raise WorldStateError(
                f"Ambiguous duplicate entity ids: {sorted(duplicate_entities)}."
            )
        for entity in self.entities:
            if entity.frame != self.frame:
                raise WorldStateError(
                    f"Entity '{entity.object_id}' uses frame '{entity.frame}', "
                    f"expected '{self.frame}'."
                )
            if entity.timestamp > self.timestamp + 1e-9:
                raise WorldStateError(
                    f"Entity '{entity.object_id}' timestamp is newer than its world state."
                )
        relation_ids = [relation.object_id for relation in self.relations]
        if _duplicates(relation_ids):
            raise WorldStateError("Each object may have only one relationship record.")
        unknown_relations = set(relation_ids) - set(entity_ids)
        if unknown_relations:
            raise WorldStateError(
                f"Relations reference unknown entities: {sorted(unknown_relations)}."
            )
        fixture_ids = [fixture.fixture_id for fixture in self.fixtures]
        if _duplicates(fixture_ids):
            raise WorldStateError("Each fixture may have only one scalar-state record.")

    def require_entity(
        self, object_id: str, *, maximum_age_s: float = 0.1
    ) -> ObjectObservation:
        """Return one fresh, valid entity or raise an actionable error."""

        matches = [entity for entity in self.entities if entity.object_id == object_id]
        if not matches:
            available = ", ".join(sorted(entity.object_id for entity in self.entities))
            raise WorldStateError(
                f"Unknown workcell entity id '{object_id}'. Available ids: {available}."
            )
        if len(matches) != 1:  # Defensive even though __post_init__ rejects this.
            raise WorldStateError(f"Ambiguous workcell entity id '{object_id}'.")
        try:
            matches[0].require_fresh(now=self.timestamp, maximum_age_s=maximum_age_s)
        except ObjectObservationError as exc:
            raise WorldStateError(str(exc)) from exc
        return matches[0]

    def relation_for(self, object_id: str) -> EntityRelation:
        """Return one object's relationships, defaulting to unknown relationships."""

        self.require_entity(object_id)
        for relation in self.relations:
            if relation.object_id == object_id:
                return relation
        return EntityRelation(
            object_id=object_id,
            supported_by=None,
            held_by=None,
            receptacle_id=None,
        )

    def require_fixture(self, fixture_id: str) -> FixtureObservation:
        """Return one unambiguous fixture scalar state."""

        matches = [fixture for fixture in self.fixtures if fixture.fixture_id == fixture_id]
        if not matches:
            available = ", ".join(sorted(item.fixture_id for item in self.fixtures))
            raise WorldStateError(
                f"Unknown workcell fixture id '{fixture_id}'. Available ids: {available}."
            )
        if len(matches) != 1:
            raise WorldStateError(f"Ambiguous workcell fixture id '{fixture_id}'.")
        return matches[0]

    def replace_entity(self, replacement: ObjectObservation) -> "WorldState":
        """Return a test/evaluation snapshot with one entity replaced by stable id."""

        self.require_entity(replacement.object_id)
        entities = tuple(
            replacement if item.object_id == replacement.object_id else item
            for item in self.entities
        )
        return replace(self, entities=entities)


def _duplicates(values: list[str]) -> set[str]:
    return {value for value in values if values.count(value) > 1}


def _finite_vector(values: tuple[float, ...], length: int, name: str) -> None:
    if len(values) != length or any(not isfinite(float(value)) for value in values):
        raise WorldStateError(f"{name} must contain {length} finite values.")


def _unit_quaternion(values: tuple[float, ...], name: str) -> None:
    _finite_vector(values, 4, name)
    norm = sqrt(sum(float(value) ** 2 for value in values))
    if abs(norm - 1.0) > 1e-6:
        raise WorldStateError(f"{name} must be a unit wxyz quaternion.")
