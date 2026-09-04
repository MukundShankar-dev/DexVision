from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dexvision.logging.phase_labels import (
    CausalPhaseTracker,
    PhaseInterval,
    PhaseLabelError,
    PhaseTransition,
    derive_pick_place_segments,
    phase_disagreement_report,
    phase_tracker_from_config,
    phases_to_intervals,
    validate_phase_intervals,
)


ROOT = Path(__file__).resolve().parents[1]


def test_causal_tracker_uses_only_current_and_prior_state() -> None:
    tracker = CausalPhaseTracker(
        initial_phase="approach",
        transitions=(
            PhaseTransition("approach", "push_contact", "requested_object_contact_started_now"),
            PhaseTransition("push_contact", "settle", "inside_target"),
        ),
        vocabulary=("approach", "push_contact", "settle"),
    )

    assert tracker.update({"requested_object_contact": False}) == "approach"
    assert tracker.update({"requested_object_contact": True, "inside_target": True}) == (
        "push_contact"
    )
    assert tracker.update({"requested_object_contact": True, "inside_target": True}) == "settle"


def test_frozen_config_builds_every_required_phase_machine() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "level4_dataset.yaml").read_text(encoding="utf-8")
    )

    for skill_name in (
        "reach_object",
        "pick_object",
        "place_held_object",
        "push_object_to_target",
        "press_button",
        "pick_place_sequence",
    ):
        tracker = phase_tracker_from_config(config, skill_name=skill_name)
        assert tracker.current_phase in config["phase_contract"]["vocabulary"]


def test_phase_intervals_round_trip_and_reject_overlap() -> None:
    phases = ("approach", "approach", "acquire", "lift", "lift")
    intervals = phases_to_intervals(phases)

    assert intervals == (
        PhaseInterval("approach", 0, 2),
        PhaseInterval("acquire", 2, 3),
        PhaseInterval("lift", 3, 5),
    )
    assert validate_phase_intervals(intervals, frame_count=5, phases=phases) == intervals
    with pytest.raises(PhaseLabelError, match="expected start 2, got 1"):
        validate_phase_intervals(
            (
                PhaseInterval("approach", 0, 2),
                PhaseInterval("acquire", 1, 5),
            ),
            frame_count=5,
        )


def test_audited_disagreement_is_reported_without_relabeling_online_phase() -> None:
    online = ["approach", "acquire", "lift", "stabilize"]
    audited = ["approach", "approach", "lift", None]

    report = phase_disagreement_report(online, audited)

    assert online == ["approach", "acquire", "lift", "stabilize"]
    assert report == {
        "audited_frame_count": 3,
        "disagreement_count": 1,
        "disagreement_fraction": pytest.approx(1 / 3),
        "disagreement_frames": [1],
    }


def test_complete_pick_place_yields_reach_pick_and_place_segments() -> None:
    phases = (
        "approach",
        "acquire",
        "lift",
        "stabilize",
        "transport",
        "place",
        "release",
        "settle",
        "retract",
    )
    intervals = phases_to_intervals(phases)

    segments = derive_pick_place_segments(intervals, frame_count=len(phases))

    assert [segment.skill_name for segment in segments] == [
        "reach_object",
        "pick_object",
        "place_held_object",
    ]
    assert [(segment.start_frame, segment.end_frame) for segment in segments] == [
        (0, 1),
        (1, 4),
        (4, 8),
    ]
