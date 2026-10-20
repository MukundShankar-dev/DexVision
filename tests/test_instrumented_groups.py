"""Rejected requests retain their predeclared object/goal family."""

import pytest

from dexvision.evaluation.instrumented_groups import (
    canonical_family_metrics,
    physical_success,
)


def test_missing_legacy_target_label_does_not_invent_failure_family():
    request = {
        "entity_id": "block",
        "typed_goal": {"object_id": "block", "target_id": "slot"},
    }
    success = {
        "terminal_result": "succeeded",
        "target_type": "placement_slot",
        "physical_audit": {"passed": True},
    }
    rejected = {
        "terminal_result": "rejected_before_execution",
        "target_type": "approach_or_fixture",
    }
    measured = canonical_family_metrics(
        [(request, success), (request, rejected)],
        object_families={"block": "cuboid"},
        target_types={"slot": "placement_slot"},
    )
    assert set(measured["groups"]) == {"object/cuboid", "goal/placement_slot"}
    assert measured["worst_family_success_rate"] == 0.5
    assert all(g["attempts"] == 2 for g in measured["groups"].values())


def test_unverified_success_cannot_pass_family_metric():
    with pytest.raises(ValueError, match="independent physical"):
        physical_success({"terminal_result": "succeeded"})
    assert not physical_success(
        {"terminal_result": "succeeded", "physical_audit": {"passed": False}}
    )
