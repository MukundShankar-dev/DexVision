"""Fail-closed 5.7 boundary for the unchanged accepted control bundles.

Only RGB and acquisition/delivery times enter this adapter. There is deliberately
no task, simulator, truth callback, semantic-id lookup or default-state argument.
The accepted 5.6 visual baseline cannot yet supply any full control bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import time
from typing import Protocol

import numpy as np

from dexvision.perception.object_observations import INFERRED_PERCEPTION, ObjectObservationError
from dexvision.perception.world_estimate import WorldEstimate, inferred_world_estimate

SKILLS = ("reach", "pick", "place", "push", "press")


class RGBBaseline(Protocol):
    def reset(self) -> None: ...
    def infer(self, rgb: np.ndarray, timestamp: float) -> tuple: ...


@dataclass(frozen=True)
class PerceptionDecision:
    """A rejection is terminal and has no proposed or applied actuator action."""

    terminal_result: str
    reason: str
    unavailable_fields: tuple[str, ...]
    estimate: WorldEstimate
    delivery_timestamp: float
    action: None = None


def unavailable_fields(world: WorldEstimate, *, now: float) -> tuple[str, ...]:
    """Expose every failing field, including uncertified hypotheses and age.

    This diagnostic never promotes a hypothesis or manufactures an empty
    measured set. The existing full boundary remains the conversion authority.
    """
    problems = []
    if world.source != INFERRED_PERCEPTION:
        problems.append("world.source:evaluation_truth_prohibited")
    if not isfinite(now) or now < world.timestamp or now - world.timestamp > .1:
        problems.append("world.timestamp:stale_or_future")
    if not world.entities:
        problems.append("entities:missing")
    fields = [(name, getattr(world, name)) for name in ("robot", "relations", "fixtures", "contacts")]
    for entity in world.entities:
        for name in ("object_id", "class_id", "position", "orientation_wxyz",
                     "linear_velocity", "angular_velocity"):
            fields.append((f"{entity.observation_id}.{name}", getattr(entity.evidence, name)))
    for name, field in fields:
        if field.availability != "available" or not field.valid:
            problems.append(f"{name}:{field.availability if field.valid else 'invalid_or_ambiguous'}")
        if not isfinite(now) or now < field.timestamp or now - field.timestamp > .1:
            problems.append(f"{name}:stale_or_future")
        if field.confidence < .8:
            problems.append(f"{name}:low_confidence")
    return tuple(problems)


class PerceptionPolicy:
    """Guard an accepted bundle without invoking it on unsupported evidence.

    This version is intentionally rejection-only: 5.6 explicitly declares all
    five bundles incompatible. A complete synthetic WorldEstimate cannot qualify
    a real producer or remove place V25's privileged collision preview. A later
    versioned producer/integration amendment must establish that evidence before
    an accepted controller can be connected here.
    """

    def __init__(self, baseline: RGBBaseline, skill: str, *, incompatibility: str):
        if skill not in SKILLS or not incompatibility:
            raise ValueError("A required skill and its accepted compatibility blocker are required.")
        self.baseline = baseline
        self.skill = skill
        self.incompatibility = incompatibility

    def reset(self) -> None:
        self.baseline.reset()

    def observe(self, rgb: np.ndarray, *, timestamp: float, now: float) -> PerceptionDecision:
        """RGB [480, 640, 3] uint8; timestamps in simulator seconds, no truth."""
        if (rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8
                or not isfinite(timestamp) or timestamp < 0 or not isfinite(now) or now < timestamp):
            raise ValueError("Expected calibrated uint8 RGB and finite acquisition/delivery times.")
        started = time.perf_counter()
        world = inferred_world_estimate(self.baseline.infer(rgb, timestamp), timestamp)
        now += time.perf_counter() - started
        missing = unavailable_fields(world, now=now)
        try:
            world.require_control_state(now=now, required_motion=("linear_velocity", "angular_velocity"))
        except ObjectObservationError as exc:
            boundary_reason = str(exc)
        else:
            boundary_reason = "producer_qualification_required"
        return PerceptionDecision(
            "rejected_before_execution",
            f"unsupported_accepted_bundle:{self.skill}: {self.incompatibility}; {boundary_reason}",
            missing, world, now,
        )
