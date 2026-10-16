"""Body-point association must preserve RGB evidence and reject missing poses."""

import numpy as np

from dexvision.perception.body_tracking_localization import associate_body_centers
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_pose import PoseEstimate
from dexvision.perception.object_tracker import Track
from dexvision.perception.predictive_tracker import PredictiveTracker


def tracker():
    return PredictiveTracker(maximum_distance_px=60, maximum_gap_s=0.51, ambiguity_margin_px=3)


def pose(box_x, position, timestamp=0):
    mask = np.zeros((100, 400), bool)
    mask[30:40, box_x:box_x + 10] = True
    detection = Detection("rigid_cuboid", (box_x, 30, box_x + 10, 40), mask, 1.0)
    return PoseEstimate(
        Track("provisional", detection, timestamp, False, False), position,
        None, None, None, None, None,
        "full_orientation_semantic_identity_and_motion_unavailable",
    )


def projected(points):
    return points[:, :2]


def test_mask_center_jump_does_not_change_stable_inferred_body_identity():
    t = tracker()
    first = associate_body_centers(t, (pose(10, (40., 40., .03)),), 0, projected)[0]
    current = pose(200, (42., 40., .03), .17)
    result = associate_body_centers(t, (current,), .17, projected)[0]
    assert result.track.track_id == first.track.track_id and not result.track.ambiguous
    assert result.track.detection is current.track.detection
    assert result.position == current.position and result.linear_velocity is None


def test_multiple_body_tracks_follow_current_positions_when_visible_centers_cross():
    t = tracker()
    initial = associate_body_centers(t, (pose(10, (40., 40., 0.)), pose(200, (240., 40., 0.))), 0, projected)
    later = associate_body_centers(t, (pose(200, (42., 40., 0.)), pose(10, (238., 40., 0.))), .17, projected)
    assert [p.track.track_id for p in later] == [p.track.track_id for p in initial]
    assert all(not p.track.ambiguous for p in later)


def test_missing_localization_never_changes_association_history():
    t = tracker()
    first = associate_body_centers(t, (pose(10, (40., 40., 0.)),), 0, projected)[0]
    invalid = associate_body_centers(t, (pose(200, None),), .17, projected)[0]
    assert invalid.track.ambiguous and invalid.position is None
    assert t.previous[first.track.track_id].timestamp == 0
    restored = associate_body_centers(t, (pose(200, (41., 40., 0.)),), .34, projected)[0]
    assert restored.track.track_id == first.track.track_id


def test_indistinguishable_body_positions_still_reject_identity():
    results = associate_body_centers(tracker(), (pose(10, (40., 40., 0.)), pose(200, (41., 40., 0.))), 0, projected)
    assert all(p.track.ambiguous for p in results)
    assert all(p.rejection_reason.startswith("ambiguous_identity") for p in results)


def test_nonfinite_projection_is_invalid_without_history_poisoning():
    t = tracker()
    result = associate_body_centers(t, (pose(10, (40., 40., 0.)),), 0, lambda p: np.full((1, 2), np.nan))[0]
    assert result.track.ambiguous and not t.previous


def test_other_classes_keep_original_visible_component_association():
    from dataclasses import replace

    values = []
    for x in (10, 200):
        p = pose(x, (40., 40., 0.))
        values.append(replace(p, track=replace(p.track, detection=replace(p.track.detection, class_id="rigid_flat_puck"))))
    result = associate_body_centers(tracker(), tuple(values), 0, projected)
    assert all(not p.track.ambiguous for p in result)
    assert len({p.track.track_id for p in result}) == 2


def test_body_tracking_preserves_original_numerical_scoring():
    import ast
    from pathlib import Path

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

    assert body("dexvision/evaluation/perception_body_tracking.py") == body("dexvision/evaluation/perception_current.py")
