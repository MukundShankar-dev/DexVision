"""Causal Level 4 phase labels, intervals, and pick/place segments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class PhaseLabelError(ValueError):
    """Raised when phase rules or saved phase labels are invalid."""


@dataclass(frozen=True)
class PhaseTransition:
    """One ordered state-machine edge evaluated from current/prior state only."""

    source: str
    target: str
    predicate: str


@dataclass(frozen=True)
class PhaseInterval:
    """Inclusive-exclusive interval for one phase in an episode."""

    phase: str
    start_frame: int
    end_frame: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "phase": self.phase,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


@dataclass(frozen=True)
class SkillSegment:
    """Inclusive-exclusive training segment derived from a complete sequence."""

    skill_name: str
    start_frame: int
    end_frame: int
    source_phases: tuple[str, ...]

    def to_dict(self) -> dict[str, str | int | list[str]]:
        return {
            "skill_name": self.skill_name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "source_phases": list(self.source_phases),
        }


DEFAULT_PICK_PLACE_TRANSITIONS = (
    PhaseTransition("approach", "acquire", "approach_success_dwell_passed"),
    PhaseTransition("acquire", "lift", "requested_object_held_now"),
    PhaseTransition("lift", "stabilize", "requested_object_above_lift_height_now"),
    PhaseTransition("stabilize", "transport", "pick_success_dwell_passed"),
    PhaseTransition("transport", "place", "requested_object_inside_target_now"),
    PhaseTransition("place", "release", "release_started_now"),
    PhaseTransition("release", "settle", "requested_object_not_held_now"),
    PhaseTransition("settle", "retract", "place_success_dwell_passed"),
)


class CausalPhaseTracker:
    """Evaluate one ordered phase transition per sample without future frames."""

    def __init__(
        self,
        *,
        initial_phase: str,
        transitions: Sequence[PhaseTransition],
        vocabulary: Sequence[str],
    ) -> None:
        self.vocabulary = tuple(vocabulary)
        if not self.vocabulary or len(set(self.vocabulary)) != len(self.vocabulary):
            raise PhaseLabelError("phase vocabulary must contain unique non-empty phases.")
        if initial_phase not in self.vocabulary:
            raise PhaseLabelError(f"initial phase '{initial_phase}' is not in the vocabulary.")
        self.transitions = tuple(transitions)
        for transition in self.transitions:
            if transition.source not in self.vocabulary or transition.target not in self.vocabulary:
                raise PhaseLabelError("phase transition contains a phase outside the vocabulary.")
            if not transition.predicate:
                raise PhaseLabelError("phase transition predicates must be non-empty.")
        self.current_phase = initial_phase
        self._prior_state: Mapping[str, Any] | None = None

    def update(self, current_state: Mapping[str, Any]) -> str:
        """Return the online phase after evaluating this sample causally."""

        if not isinstance(current_state, Mapping):
            raise PhaseLabelError("current phase state must be a mapping.")
        for transition in self.transitions:
            if transition.source != self.current_phase:
                continue
            if _predicate_value(
                transition.predicate,
                current=current_state,
                prior=self._prior_state,
            ):
                self.current_phase = transition.target
                break
        self._prior_state = dict(current_state)
        return self.current_phase


def phase_tracker_from_config(
    config: Mapping[str, Any],
    *,
    skill_name: str,
) -> CausalPhaseTracker:
    """Build a tracker from the frozen Level 4 phase configuration."""

    phase_contract = _mapping(config, "phase_contract")
    vocabulary = _string_sequence(phase_contract.get("vocabulary"), "phase vocabulary")
    machine_config = _mapping(config, "online_phase_state_machine")
    if machine_config.get("causal") is not True or machine_config.get("future_frames_allowed"):
        raise PhaseLabelError("online phase state machine must be causal and forbid future frames.")
    machines = _mapping(machine_config, "machines")
    if skill_name == "pick_place_sequence":
        return CausalPhaseTracker(
            initial_phase="approach",
            transitions=DEFAULT_PICK_PLACE_TRANSITIONS,
            vocabulary=vocabulary,
        )
    machine = _mapping(machines, skill_name)
    initial_phase = machine.get("initial_phase")
    if not isinstance(initial_phase, str) or not initial_phase:
        raise PhaseLabelError(f"phase machine '{skill_name}' needs an initial_phase.")
    transitions = []
    raw_transitions = machine.get("transitions")
    if not isinstance(raw_transitions, Sequence) or isinstance(raw_transitions, str):
        raise PhaseLabelError(f"phase machine '{skill_name}' transitions must be a sequence.")
    for raw_transition in raw_transitions:
        if not isinstance(raw_transition, Mapping):
            raise PhaseLabelError("each phase transition must be a mapping.")
        transitions.append(
            PhaseTransition(
                source=_required_string(raw_transition, "from"),
                target=_required_string(raw_transition, "to"),
                predicate=_required_string(raw_transition, "predicate"),
            )
        )
    return CausalPhaseTracker(
        initial_phase=initial_phase,
        transitions=transitions,
        vocabulary=vocabulary,
    )


def phases_to_intervals(phases: Sequence[str]) -> tuple[PhaseInterval, ...]:
    """Run-length encode frame phases as monotonic inclusive-exclusive intervals."""

    values = tuple(phases)
    if not values:
        raise PhaseLabelError("at least one online phase is required.")
    if any(not isinstance(phase, str) or not phase for phase in values):
        raise PhaseLabelError("online phases must be non-empty strings.")
    intervals: list[PhaseInterval] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            intervals.append(PhaseInterval(values[start], start, index))
            start = index
    return tuple(intervals)


def validate_phase_intervals(
    intervals: Sequence[PhaseInterval | Mapping[str, Any]],
    *,
    frame_count: int,
    phases: Sequence[str] | None = None,
) -> tuple[PhaseInterval, ...]:
    """Validate full, gap-free, non-overlapping phase coverage."""

    if frame_count <= 0:
        raise PhaseLabelError("frame_count must be positive.")
    coerced = tuple(_coerce_interval(value) for value in intervals)
    if not coerced:
        raise PhaseLabelError("phase intervals must not be empty.")
    expected_start = 0
    reconstructed: list[str] = []
    for interval in coerced:
        if interval.start_frame != expected_start:
            raise PhaseLabelError(
                "phase intervals must be monotonic, non-overlapping, and gap-free; "
                f"expected start {expected_start}, got {interval.start_frame}."
            )
        if interval.end_frame <= interval.start_frame:
            raise PhaseLabelError("phase interval end_frame must exceed start_frame.")
        if interval.end_frame > frame_count:
            raise PhaseLabelError("phase interval extends beyond the episode frame count.")
        reconstructed.extend([interval.phase] * (interval.end_frame - interval.start_frame))
        expected_start = interval.end_frame
    if expected_start != frame_count:
        raise PhaseLabelError("phase intervals must cover every episode frame.")
    if phases is not None and tuple(reconstructed) != tuple(phases):
        raise PhaseLabelError("phase intervals do not reconstruct saved online phases.")
    return coerced


def phase_disagreement_report(
    online_phases: Sequence[str],
    audited_phases: Sequence[str | None],
) -> dict[str, Any]:
    """Report audited-vs-online disagreement without changing online labels."""

    online = tuple(online_phases)
    audited = tuple(audited_phases)
    if len(online) != len(audited):
        raise PhaseLabelError("online and audited phase arrays must have equal length.")
    audited_indices = [index for index, phase in enumerate(audited) if phase not in {None, ""}]
    disagreements = [index for index in audited_indices if audited[index] != online[index]]
    count = len(audited_indices)
    return {
        "audited_frame_count": count,
        "disagreement_count": len(disagreements),
        "disagreement_fraction": (len(disagreements) / count) if count else 0.0,
        "disagreement_frames": disagreements,
    }


def derive_pick_place_segments(
    intervals: Sequence[PhaseInterval | Mapping[str, Any]],
    *,
    frame_count: int,
) -> tuple[SkillSegment, SkillSegment, SkillSegment]:
    """Derive compatible reach, pick, and place segments from one complete episode."""

    values = validate_phase_intervals(intervals, frame_count=frame_count)
    phase_bounds = {interval.phase: interval for interval in values}
    required = ("approach", "acquire", "lift", "stabilize", "transport", "place", "release", "settle")
    missing = [phase for phase in required if phase not in phase_bounds]
    if missing:
        raise PhaseLabelError(
            "complete pick/place episode is missing phases: " + ", ".join(missing)
        )
    ordered = [phase_bounds[phase].start_frame for phase in required]
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise PhaseLabelError("pick/place phases are not in the required causal order.")
    return (
        SkillSegment(
            "reach_object",
            phase_bounds["approach"].start_frame,
            phase_bounds["acquire"].start_frame,
            ("approach",),
        ),
        SkillSegment(
            "pick_object",
            phase_bounds["acquire"].start_frame,
            phase_bounds["transport"].start_frame,
            ("acquire", "lift", "stabilize"),
        ),
        SkillSegment(
            "place_held_object",
            phase_bounds["transport"].start_frame,
            phase_bounds["settle"].end_frame,
            ("transport", "place", "release", "settle"),
        ),
    )


def _predicate_value(
    predicate: str,
    *,
    current: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
) -> bool:
    if predicate in current:
        return current[predicate] is True
    if predicate.endswith("_started_now"):
        state_name = predicate.removesuffix("_started_now")
        current_value = current.get(state_name) is True
        prior_value = prior is not None and prior.get(state_name) is True
        return current_value and not prior_value
    return False


def _coerce_interval(value: PhaseInterval | Mapping[str, Any]) -> PhaseInterval:
    if isinstance(value, PhaseInterval):
        return value
    if not isinstance(value, Mapping):
        raise PhaseLabelError("phase intervals must be PhaseInterval values or mappings.")
    phase = _required_string(value, "phase")
    start = value.get("start_frame")
    end = value.get("end_frame")
    if not isinstance(start, int) or isinstance(start, bool):
        raise PhaseLabelError("phase interval start_frame must be an integer.")
    if not isinstance(end, int) or isinstance(end, bool):
        raise PhaseLabelError("phase interval end_frame must be an integer.")
    return PhaseInterval(phase, start, end)


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise PhaseLabelError(f"'{key}' must be a mapping.")
    return result


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise PhaseLabelError(f"'{key}' must be a non-empty string.")
    return result


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise PhaseLabelError(f"{name} must be a sequence of strings.")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise PhaseLabelError(f"{name} must contain non-empty strings.")
    return result
