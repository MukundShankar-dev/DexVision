"""Typed object observations shared by simulation truth and perception.

Level 4 deliberately gives simulator state and future inferred perception the
same outer schema.  The ``source`` and ``confidence`` fields preserve the
important distinction between those two producers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Final, Iterable


SIMULATOR_GROUND_TRUTH: Final = "simulator_ground_truth"
INFERRED_PERCEPTION: Final = "inferred_perception"
SUPPORTED_SOURCES: Final = frozenset(
    {SIMULATOR_GROUND_TRUTH, INFERRED_PERCEPTION}
)


class ObjectObservationError(ValueError):
    """Raised when an object observation is invalid, stale, or unusable."""


@dataclass(frozen=True)
class ObjectObservation:
    """Pose and optional velocity for one stable workcell entity.

    Position and velocity use metres and metres per second. Orientation uses
    the MuJoCo ``wxyz`` unit-quaternion convention. ``frame`` names the
    coordinate frame explicitly; Level 4.1 uses ``mujoco_world``.
    """

    object_id: str
    class_id: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float] | None
    source: str
    confidence: float
    timestamp: float
    frame: str
    angular_velocity: tuple[float, float, float] | None = None
    valid: bool = True

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ObjectObservationError("object_id must be a non-empty stable id.")
        if not self.class_id:
            raise ObjectObservationError(
                f"Object '{self.object_id}' must have a non-empty class_id."
            )
        if not self.frame:
            raise ObjectObservationError(
                f"Object '{self.object_id}' must name its coordinate frame."
            )
        if self.source not in SUPPORTED_SOURCES:
            raise ObjectObservationError(
                f"Object '{self.object_id}' has unsupported source '{self.source}'; "
                f"expected one of {sorted(SUPPORTED_SOURCES)}."
            )
        _require_finite_vector(self.position, 3, "position", self.object_id)
        _require_finite_vector(
            self.orientation_wxyz, 4, "orientation_wxyz", self.object_id
        )
        if self.linear_velocity is not None:
            _require_finite_vector(
                self.linear_velocity, 3, "linear_velocity", self.object_id
            )
        if self.angular_velocity is not None:
            _require_finite_vector(
                self.angular_velocity, 3, "angular_velocity", self.object_id
            )
        if not isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ObjectObservationError(
                f"Object '{self.object_id}' timestamp must be finite and non-negative."
            )
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ObjectObservationError(
                f"Object '{self.object_id}' confidence must be within [0, 1]."
            )
        quat_norm = sqrt(sum(value * value for value in self.orientation_wxyz))
        if abs(quat_norm - 1.0) > 1e-6:
            raise ObjectObservationError(
                f"Object '{self.object_id}' orientation_wxyz must be a unit "
                f"quaternion; norm={quat_norm:.9f}."
            )
        if self.valid and self.confidence <= 0.0:
            raise ObjectObservationError(
                f"Valid object '{self.object_id}' must have positive confidence."
            )

    def require_fresh(self, *, now: float, maximum_age_s: float) -> None:
        """Raise a clear error when this observation cannot cross a skill boundary."""

        if not isfinite(now) or now < self.timestamp:
            raise ObjectObservationError(
                f"Current time {now!r} precedes '{self.object_id}' timestamp "
                f"{self.timestamp:.6f}."
            )
        if not isfinite(maximum_age_s) or maximum_age_s < 0.0:
            raise ObjectObservationError("maximum_age_s must be finite and non-negative.")
        if not self.valid:
            raise ObjectObservationError(
                f"Object '{self.object_id}' observation is marked invalid."
            )
        age = now - self.timestamp
        if age > maximum_age_s:
            raise ObjectObservationError(
                f"Object '{self.object_id}' observation is stale: age={age:.6f}s, "
                f"maximum={maximum_age_s:.6f}s."
            )


def make_object_observation(
    *,
    object_id: str,
    class_id: str,
    position: Iterable[float],
    orientation_wxyz: Iterable[float],
    linear_velocity: Iterable[float] | None,
    source: str,
    confidence: float,
    timestamp: float,
    frame: str,
    angular_velocity: Iterable[float] | None = None,
    valid: bool = True,
) -> ObjectObservation:
    """Build an observation while normalizing a finite quaternion.

    This constructor is shared by simulator and future perception adapters so
    neither producer gets a privileged or structurally different schema.
    """

    quaternion = tuple(float(value) for value in orientation_wxyz)
    _require_finite_vector(quaternion, 4, "orientation_wxyz", object_id)
    norm = sqrt(sum(value * value for value in quaternion))
    if norm <= 0.0:
        raise ObjectObservationError(
            f"Object '{object_id}' orientation_wxyz must be non-zero."
        )
    return ObjectObservation(
        object_id=object_id,
        class_id=class_id,
        position=_float_tuple(position, 3, "position", object_id),
        orientation_wxyz=tuple(value / norm for value in quaternion),
        linear_velocity=(
            None
            if linear_velocity is None
            else _float_tuple(linear_velocity, 3, "linear_velocity", object_id)
        ),
        angular_velocity=(
            None
            if angular_velocity is None
            else _float_tuple(angular_velocity, 3, "angular_velocity", object_id)
        ),
        source=source,
        confidence=float(confidence),
        timestamp=float(timestamp),
        frame=frame,
        valid=bool(valid),
    )


def _float_tuple(
    values: Iterable[float], length: int, field: str, object_id: str
) -> tuple:
    result = tuple(float(value) for value in values)
    _require_finite_vector(result, length, field, object_id)
    return result


def _require_finite_vector(
    values: Iterable[float], length: int, field: str, object_id: str
) -> None:
    vector = tuple(values)
    if len(vector) != length or any(not isfinite(float(value)) for value in vector):
        raise ObjectObservationError(
            f"Object '{object_id}' {field} must contain {length} finite values."
        )
