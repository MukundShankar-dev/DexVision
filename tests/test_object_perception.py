"""Camera/GPU/GUI-free checkpoint 5.6 isolation and temporal boundary tests."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from dexvision.evaluation.perception_audit import truth_observation
from dexvision.evaluation.evaluate_perception import serialize_estimate, synthetic_rejection_check
from dexvision.perception.object_detector import ColorDetector, Detection, fit_palette
from dexvision.perception.object_observations import INFERRED_PERCEPTION, ObjectObservation, make_object_observation
from dexvision.perception.object_pose import MetricPoseEstimator, validate_delivery, world_state_from_observations
from dexvision.perception.object_tracker import ObjectTracker, Track
from dexvision.sim.world_state import RobotObservation


def detection(x=10, *, category="rigid_cuboid", confidence=1.):
    mask = np.zeros((480, 640), bool)
    mask[10:25, x:x+20] = True
    return Detection(category, (x, 10, x+20, 25), mask, confidence)


def tracker():
    return ObjectTracker(maximum_distance_px=50., maximum_gap_s=.5, ambiguity_margin_px=2.)


def calibration():
    return {"version": "level4/visual-camera-v1", "world_frame": "mujoco_world", "length_units": "m",
            "intrinsics": [[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]],
            "camera_from_world": [[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 1.], [0, 0, 0, 1.]]}


def estimator():
    return MetricPoseEstimator(calibration(), {"rigid_cuboid": {"shape": "box", "half_size_m": [.02]*3}},
                               maximum_gap_s=.5, maximum_fit_evaluations=30, maximum_residual_px=3.)


def test_palette_fit_deterministic_and_no_pixel_mutation():
    samples = {"rigid_cuboid": [np.array([[255, 100, 10]], np.uint8)],
               "rigid_cylinder": [np.array([[0, 200, 0]], np.uint8)],
               "rigid_flat_puck": [np.array([[0, 100, 240]], np.uint8)]}
    palette = fit_palette(samples)
    assert palette == fit_palette(samples)
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[100:120, 100:120] = [255, 100, 10]
    original = rgb.copy()
    detector = ColorDetector(palette, hue_margin=10, minimum_pixels=5,
                             minimum_saturation=50, minimum_value=20, button_roi=(400, 0, 500, 100))
    found = detector.detect(rgb)
    assert len(found) == 1
    assert found[0].box == (100, 100, 120, 120)
    np.testing.assert_array_equal(rgb, original)
    with pytest.raises(ValueError, match="uint8"):
        detector.detect(rgb.astype(float))


def test_training_histogram_separates_board_and_puck_with_same_hue():
    counts = np.zeros((4, 180, 32), int)
    counts[0, 107, 15] = 10000
    counts[3, 107, 30] = 100
    hsv = np.zeros((480, 640, 3), np.uint8)
    hsv[:] = [107, 120, 200]
    hsv[100:120, 100:120, 1] = 240
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    detector = ColorDetector({"rigid_cuboid": 11, "rigid_cylinder": 68, "rigid_flat_puck": 107},
                             hue_margin=8, minimum_pixels=8, minimum_saturation=90,
                             minimum_value=25, button_roi=(390, 136, 450, 196),
                             color_counts=counts.tolist())
    output = detector.detect(rgb)
    assert len(output) == 1
    assert output[0].class_id == "rigid_flat_puck"
    assert output[0].box == (100, 100, 120, 120)


def test_continuous_silhouette_moments_translate_in_pixel_units():
    from dexvision.perception.object_pose import silhouette_features
    hull = np.array([[0., 0], [20, 0], [20, 10], [0, 10]], np.float32)
    first = silhouette_features(hull)
    # OpenCV contour moments use float32 coordinates (float64 Nx2 is an image).
    second = silhouette_features(hull+np.array([13., 7.], np.float32))
    np.testing.assert_allclose(second-first, [13., 7., 0, 0, 0], atol=1e-10)


def test_tracking_occlusion_reacquisition_and_expiry():
    subject = tracker()
    first = subject.update((detection(),), 0.)[0]
    subject.update((), .17)
    reacquired = subject.update((detection(12),), .34)[0]
    assert reacquired.track_id == first.track_id and reacquired.reacquired
    expired = subject.update((detection(12),), 1.)[0]
    assert expired.track_id != first.track_id
    subject.reset()
    assert subject.update((detection(),), 0.)[0].track_id == first.track_id


def test_ambiguity_rejects_tied_association_without_corrupting_history():
    subject = tracker()
    first = subject.update((detection(10), detection(30)), 0.)
    result = subject.update((detection(20),), .1)
    assert result[0].ambiguous
    assert set(subject.previous) == {t.track_id for t in first}
    with pytest.raises(ValueError, match="ambiguous"):
        validate_delivery(result[0], now=.1, minimum_confidence=.8, maximum_age_s=.1)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -1.])
def test_invalid_timestamps(timestamp):
    with pytest.raises(ValueError, match="timestamp"):
        tracker().update((detection(),), timestamp)


def test_timestamp_order_requires_explicit_reset():
    subject = tracker()
    subject.update((), 1.)
    with pytest.raises(ValueError, match="increase"):
        subject.update((), 1.)
    with pytest.raises(ValueError, match="increase"):
        subject.update((), .5)


@pytest.mark.parametrize("age", [.10001, .17, .34])
def test_delivery_rejects_export_gap_ages(age):
    track = Track("visual/0", detection(), 1., False, False)
    with pytest.raises(ValueError, match="stale"):
        validate_delivery(track, now=1+age, minimum_confidence=.8, maximum_age_s=.1)


def test_synthetic_stale_ambiguity_confidence_rejection_is_complete():
    assert synthetic_rejection_check()["rejection_fraction"] == 1.
    track = Track("visual/0", detection(), 1., False, False)
    validate_delivery(track, now=1.099, minimum_confidence=.8, maximum_age_s=.1)
    for confidence in (float("nan"), float("inf"), 1.1, -.1):
        with pytest.raises(ValueError):
            detection(confidence=confidence)


def test_metric_projection_body_center_and_real_elapsed_velocity():
    subject = estimator()
    def sample(x, timestamp):
        # The synthetic extent is a *known* projected primitive; no GT enters
        # the estimator. Metric body centre differs from its visible surface.
        from dexvision.perception.object_pose import primitive_points
        points = primitive_points("box", np.array([.02]*3)) + [x, 0, .02]
        pixels = subject.project(points)
        mask = np.zeros((480, 640), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pixels.astype(np.int32)), 1)
        y, xs = np.nonzero(mask)
        d = Detection("rigid_cuboid", (int(xs.min()), int(y.min()), int(xs.max()+1), int(y.max()+1)), mask.astype(bool), 1.)
        return subject.estimate(Track("visual/0", d, timestamp, False, False))
    first = sample(0, 0.)
    second = sample(.01, .34)
    assert first.linear_velocity is None
    np.testing.assert_allclose(second.linear_velocity, (np.asarray(second.position)-first.position)/.34)
    assert abs(first.position[2]-.02) < .04
    assert first.orientation_wxyz is None and first.angular_velocity is None
    with pytest.raises(ValueError, match="uncertified"):
        first.require_observation(now=0., minimum_confidence=.8, maximum_age_s=.1)
    subject.reset()
    assert sample(.01, .5).linear_velocity is None


def test_missing_orientation_velocity_and_geometry_never_fabricated():
    track = Track("visual/0", detection(category="placement_target"), 1., False, False)
    estimate = estimator().estimate(track)
    assert estimate.position is None and estimate.orientation_wxyz is None
    assert estimate.linear_velocity is None and estimate.angular_velocity is None
    assert estimate.rejection_reason == "unsupported_geometry"
    with pytest.raises(ValueError, match="unsupported"):
        estimate.require_observation(now=1., minimum_confidence=.8, maximum_age_s=.1)


def test_withheld_truth_labels_do_not_change_inferred_observations_or_history():
    # Two isolated inferred histories receive identical RGB components and
    # timestamps while the evaluator's withheld pose/id/outcome changes.
    left, right = estimator(), estimator()
    annotation = {"object_id": "original", "class_id": "rigid_cuboid", "pose_6d":
                  {"translation_m": [0, 0, 0], "rotation_matrix": np.eye(3).tolist()}}
    for timestamp in (0., .17, .51):
        a = left.estimate(Track("visual/0", detection(), timestamp, False, False))
        changed = {**annotation, "object_id": "withheld_changed", "pose_6d":
                   {"translation_m": [9, 8, 7], "rotation_matrix": np.diag([-1, -1, 1]).tolist()}}
        truth_observation(changed, timestamp)
        b = right.estimate(Track("visual/0", detection(), timestamp, False, False))
        assert serialize_estimate(a) == serialize_estimate(b)
        for value in (a, b):
            with pytest.raises(ValueError):
                value.require_observation(now=timestamp, minimum_confidence=.8, maximum_age_s=.1)


def test_shared_schema_truth_and_explicit_inferred_provenance():
    annotation = {"object_id": "item", "class_id": "rigid_cuboid", "visible": False, "pose_6d":
                  {"translation_m": [.1, 0, .02], "rotation_matrix": np.eye(3).tolist()}}
    truth = truth_observation(annotation, 1.)
    inferred = make_object_observation(object_id="item", class_id="rigid_cuboid", position=[.1, 0, .02],
        orientation_wxyz=[1, 0, 0, 0], linear_velocity=None, source=INFERRED_PERCEPTION,
        confidence=.9, timestamp=1., frame="mujoco_world")
    assert type(truth) is type(inferred) is ObjectObservation
    assert truth.source != inferred.source
    assert inferred.linear_velocity is None


@pytest.mark.parametrize("missing", ["robot", "relations", "fixtures", "contacts"])
def test_missing_world_fields_block_inferred_consumers(missing):
    kwargs = dict(timestamp=1., entities=(), robot=None, relations=(), fixtures=(), contacts=(),
                  field_sources={k: "declared_robot_sensor" for k in ("robot", "relations", "fixtures", "contacts")})
    kwargs[missing] = None
    with pytest.raises(ValueError, match="missing"):
        world_state_from_observations(**kwargs)


def test_world_boundary_blocks_truth_missing_sources_and_explicit_invalidity():
    robot = RobotObservation((0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (1, 0, 0, 0), True)
    kwargs = dict(timestamp=1., entities=(), robot=robot, relations=(), fixtures=(), contacts=(),
                  field_sources={k: "declared_robot_sensor" for k in ("robot", "relations", "fixtures", "contacts")})
    assert world_state_from_observations(**kwargs).contacts == ()
    with pytest.raises(ValueError, match="provenance"):
        world_state_from_observations(**{**kwargs, "field_sources": {}})
    item = make_object_observation(object_id="item", class_id="rigid_cuboid", position=[0, 0, 0],
        orientation_wxyz=[1, 0, 0, 0], linear_velocity=None, source=INFERRED_PERCEPTION,
        confidence=.9, timestamp=1., frame="mujoco_world")
    for invalid in (replace(item, valid=False), replace(item, source="simulator_ground_truth")):
        with pytest.raises(ValueError, match="Truth or invalid"):
            world_state_from_observations(**{**kwargs, "entities": (invalid,)})
