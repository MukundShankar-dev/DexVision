"""Independent synthetic ID discontinuities, multiplicity, and coverage gates."""

import ast
from pathlib import Path

import numpy as np
import pytest

from dexvision.evaluation.perception_identity_coverage import identity_coverage
from dexvision.perception.identity_localization import ConservativeIdentityTracker
from dexvision.perception.object_detector import Detection


def tracker():
    return ConservativeIdentityTracker(
        maximum_distance_px=60, maximum_gap_s=0.51, ambiguity_margin_px=3,
    )


def detection(x, category="rigid_cuboid"):
    mask = np.zeros((100, 300), bool)
    mask[30:40, x:x + 10] = True
    return Detection(category, (x, 30, x + 10, 40), mask, 1.0)


def test_continuous_motion_keeps_visual_identity():
    t = tracker()
    initial = t.update((detection(10),), 0)[0]
    for i in range(1, 12):
        result = t.update((detection(10 + 4 * i),), i * 0.034)[0]
        assert not result.ambiguous and result.track_id == initial.track_id


def test_jump_cannot_certify_new_id_or_poison_history():
    t = tracker()
    initial = t.update((detection(10),), 0)[0]
    jumped = t.update((detection(200),), 0.17)[0]
    assert jumped.ambiguous and jumped.track_id != initial.track_id
    assert list(t.previous) == [initial.track_id]
    assert t.previous[initial.track_id].timestamp == 0
    assert not t.velocities
    returned = t.update((detection(14),), 0.34)[0]
    assert returned.track_id == initial.track_id and not returned.ambiguous


def test_expired_identity_is_declared_reacquired_without_borrowed_velocity():
    t = tracker()
    initial = t.update((detection(10),), 0)[0]
    t.update((), 0.17)
    result = t.update((detection(200),), 0.52)[0]
    assert not result.ambiguous and result.reacquired
    assert result.track_id != initial.track_id
    assert result.track_id not in t.velocities


def test_same_class_multiplicity_is_explicitly_unsupported():
    t = tracker()
    initial = t.update((detection(10),), 0)[0]
    result = t.update((detection(12), detection(200)), 0.17)
    assert all(r.ambiguous for r in result)
    assert t.previous[initial.track_id].timestamp == 0
    recovered = t.update((detection(14),), 0.34)[0]
    assert not recovered.ambiguous and recovered.track_id == initial.track_id


def test_different_classes_remain_independently_trackable():
    result = tracker().update((detection(10), detection(200, "rigid_cylinder")), 0)
    assert len(result) == 2 and all(not r.ambiguous for r in result)


def test_reset_removes_identity_anchors():
    t = tracker()
    t.update((detection(10),), 2)
    t.reset()
    result = t.update((detection(200),), 0)[0]
    assert not result.ambiguous and not result.reacquired


@pytest.mark.parametrize("timestamp", [float("nan"), -1, 0])
def test_invalid_or_nonincreasing_time_remains_rejected(timestamp):
    t = tracker()
    t.update((detection(10),), 0)
    with pytest.raises(ValueError):
        t.update((detection(12),), timestamp)


def instance(ambiguous=False, phase="release", matched=True):
    return {
        "primary_object": True, "eligible": True, "matched": matched,
        "class_id": "rigid_cuboid", "phase_stratum": phase,
        "rejection_reason": "ambiguous_identity_position_hypothesis" if ambiguous else "full_orientation_semantic_identity_and_motion_unavailable",
    }


def test_identity_coverage_cannot_qualify_by_rejecting_everything():
    assert not identity_coverage([instance(True)] * 100)["passed"]
    assert not identity_coverage([instance(matched=False)] * 100)["passed"]
    assert identity_coverage([instance()] * 100)["passed"]


def test_phase_coverage_cannot_hide_behind_large_global_denominator():
    rows = [instance(phase="transport")] * 1000 + [instance(True)] + [instance()] * 8
    report = identity_coverage(rows)
    assert report["primary"]["passed"]
    assert not report["supported_primary_class_phase"]["rigid_cuboid/release"]["passed"]
    assert not report["passed"]


def test_original_numerical_scoring_is_unchanged_except_additional_coverage():
    def body(path):
        tree = ast.parse(Path(path).read_text())
        f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "score_context")
        f.body = [n for n in f.body if not (
            isinstance(n, ast.ImportFrom) and n.module == "dexvision.evaluation.perception_identity_coverage"
        ) and not (
            isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                and t.slice.value == "identity_coverage" for t in n.targets
            )
        )]
        return ast.dump(f, include_attributes=False)

    assert body("dexvision/evaluation/perception_identity.py") == body("dexvision/evaluation/perception_current.py")
