"""RGB context remains causal; predictive association never reads truth ids."""

from dataclasses import replace

import numpy as np
import pytest

from dexvision.perception.context_baseline import (
    context_features,
    ContextPose,
    FEATURE_COUNT,
    FEATURE_VERSION,
)
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_tracker import Track
from dexvision.perception.predictive_tracker import PredictiveTracker


def detection(x=10, category="rigid_cuboid", confidence=1.0):
    mask = np.zeros((80, 100), bool)
    mask[38:42, x - 2 : x + 2] = True
    return Detection(category, (x - 2, 38, x + 2, 42), mask, confidence)


def test_context_sees_visible_hand_evidence_without_state_or_labels():
    rgb = np.zeros((80, 100, 3), np.uint8)
    d = detection(50)
    rgb[d.mask] = [200, 90, 20]
    changed = rgb.copy()
    changed[4:24, 10:35] = 100
    a, current = context_features(rgb, d, 1.0, None, maximum_gap_s=0.51)
    b, _ = context_features(changed, d, 1.0, None, maximum_gap_s=0.51)
    assert a.shape == (FEATURE_COUNT,) == (370,)
    np.testing.assert_array_equal(a[:34], b[:34])
    assert np.any(a[34:] != b[34:])
    alternate, _ = context_features(
        rgb,
        replace(d, class_id="withheld", confidence=0.1),
        1.0,
        None,
        maximum_gap_s=0.51,
    )
    np.testing.assert_array_equal(a, alternate)
    populated, _ = context_features(rgb, d, 1.17, (1.0, current), maximum_gap_s=0.51)
    assert populated[16] == 1.0
    expired, _ = context_features(rgb, d, 2.0, (1.0, current), maximum_gap_s=0.51)
    np.testing.assert_array_equal(expired[16:34], np.zeros(18))


def test_context_contract_and_reset_reject_wrong_model_features():
    with pytest.raises(ValueError, match="contract"):
        ContextPose(
            {}, {"rigid_cuboid": {"feature_version": "old34"}}, maximum_gap_s=0.51
        )
    pose = ContextPose(
        {}, {"rigid_cuboid": {"feature_version": FEATURE_VERSION}}, maximum_gap_s=0.51
    )
    pose.previous["visual/a"] = (1.0, np.ones(3))
    pose.feature_history["visual/a"] = (1.0, np.zeros(16))
    pose.reset()
    assert not pose.previous and not pose.feature_history


def test_predictive_tracker_preserves_shared_type_and_crossing_identity():
    tracker = PredictiveTracker(
        maximum_distance_px=30, maximum_gap_s=0.51, ambiguity_margin_px=0.25
    )
    first = tracker.update((detection(10), detection(40)), 0.0)
    second = tracker.update((detection(18), detection(32)), 0.1)
    third = tracker.update((detection(26), detection(24)), 0.2)
    assert all(type(t) is Track for t in third)
    assert (
        [t.track_id for t in first]
        == [t.track_id for t in second]
        == [t.track_id for t in third]
    )
    assert not any(t.ambiguous for t in third)
    tracker.reset()
    assert not tracker.velocities and not tracker.previous


def test_predictive_tracker_gap_future_and_low_confidence_bounds():
    tracker = PredictiveTracker(
        maximum_distance_px=30, maximum_gap_s=0.51, ambiguity_margin_px=1.0
    )
    first = tracker.update((detection(10),), 0.0)[0]
    tracker.update((detection(15, confidence=0.1),), 0.1)
    assert not tracker.velocities
    last = tracker.update((detection(15),), 1.0)[0]
    assert last.track_id != first.track_id and not tracker.velocities
    with pytest.raises(ValueError, match="increase"):
        tracker.update((), 1.0)
    with pytest.raises(ValueError, match="bounds"):
        PredictiveTracker(
            maximum_distance_px=30,
            maximum_gap_s=0.51,
            ambiguity_margin_px=1.0,
            velocity_alpha=0.0,
        )


def test_strict_validation_requires_physical_phase_coverage_and_zero_switches():
    from dexvision.learning.perception_context import strict_validation

    group = {
        "visible_eligible_instances": 8,
        "position_error_p95_m": 0.005,
        "position_error_median_m": 0.001,
        "recall": 1.0,
    }
    report = {
        "all_numerical_gates_pass": True,
        "all_primary_gates_pass": True,
        "primary_object_metrics": {
            "identity_switches": 0,
            "position_hypothesis_count": 100,
            "detections_matched_iou_0_5": 100,
        },
        "primary_class_phase": {
            f"{c}/{p}": dict(group)
            for c in ("rigid_cuboid", "rigid_cylinder")
            for p in ("transport", "place")
        },
    }
    assert strict_validation(report)
    report["primary_object_metrics"]["identity_switches"] = 1
    assert not strict_validation(report)
    report["primary_object_metrics"]["identity_switches"] = 0
    report["primary_class_phase"]["rigid_cuboid/place"][
        "visible_eligible_instances"
    ] = 7
    assert not strict_validation(report)


def test_consumed_context_protocol_rejects_before_data_access(tmp_path, monkeypatch):
    import dexvision.logging.perception_development as logging
    from dexvision.evaluation.perception_context import confirm_context

    monkeypatch.setattr(
        logging,
        "load_protocol",
        lambda _: {"dataset_role": "confirmation", "selected_engine": FEATURE_VERSION},
    )
    monkeypatch.setattr(
        logging,
        "verify_dataset",
        lambda *_: pytest.fail("Consumed data must not be read"),
    )
    (tmp_path / "confirmation_attempt.json").write_text("consumed")
    with pytest.raises(ValueError, match="consumed"):
        confirm_context(
            tmp_path / "protocol.json",
            tmp_path / "absent",
            tmp_path / "different-output",
        )


def test_review_slots_include_worst_cases_and_preserve_full_manifest(tmp_path):
    import json
    from dexvision.evaluation.perception_context import select_context_review

    cases = [{"episode_id": f"e{i}", "seed": 100 + i} for i in range(8)]
    videos = [{"episode": f"e{i}", "path": f"v{i}.avi"} for i in range(8)]
    (tmp_path / "manual_review.json").write_text(
        json.dumps({"videos": videos, "scope": "fresh_confirmation"})
    )
    metrics = [
        {
            "episode": f"e{i}",
            "primary_object": True,
            "eligible": True,
            "class_id": "rigid_cuboid",
            "position_error_m": i * 0.001,
        }
        for i in range(8)
    ]
    (tmp_path / "instance_metrics.json").write_text(json.dumps(metrics))
    select_context_review(
        tmp_path,
        {
            "cases": cases,
            "manual_review_sampling": {
                "repetitions": [0, 1],
                "include_worst_primary_pose_per_class": True,
            },
        },
    )
    selected = json.loads((tmp_path / "manual_review.json").read_text())
    assert {v["episode"] for v in selected["videos"]} == {"e0", "e1", "e7"}
    assert len(json.loads((tmp_path / "all_sequences.json").read_text())["videos"]) == 8


def test_shared_frame_preprocessing_is_exact_and_reset_clears_cache():
    from dexvision.perception.context_baseline import prepare_context

    rgb = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
    prepared = prepare_context(rgb)
    for x in (20, 50, 80):
        d = detection(x)
        a, _ = context_features(rgb, d, 1.0, None, maximum_gap_s=0.51)
        b, _ = context_features(
            rgb, d, 1.0, None, maximum_gap_s=0.51, prepared=prepared
        )
        np.testing.assert_array_equal(a, b)
    pose = ContextPose(
        {}, {"rigid_cuboid": {"feature_version": FEATURE_VERSION}}, maximum_gap_s=0.51
    )
    pose._context_rgb = rgb
    pose._context_maps = prepared
    pose.reset()
    assert pose._context_rgb is None and pose._context_maps is None
