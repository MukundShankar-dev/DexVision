"""Checkpoint 5.6 float64 geometry, learned ray decoding and test isolation."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from dexvision.perception.body_center_model import (
    FEATURE_NAMES, component_features, decode_metric, fit_kernel, metric_target, predict_kernel,
)
from dexvision.perception.metric_geometry import convex_polygon, polygon_features
from dexvision.perception.object_detector import ColorDetector, Detection


def test_polygon_centroid_variance_and_small_optimizer_steps():
    points = np.array([[300., 200.], [320., 200.], [320., 210.], [300., 210.]])
    expected = np.array([310., 205., np.sqrt(400/12), np.sqrt(100/12), 0.])
    np.testing.assert_allclose(polygon_features(convex_polygon(points)), expected, atol=1e-11)
    delta = np.array([1e-7, -2e-7])
    shifted = polygon_features(convex_polygon(points+delta))
    np.testing.assert_allclose(shifted[:2]-expected[:2], delta, atol=1e-12)
    np.testing.assert_allclose(shifted[2:], expected[2:], atol=1e-10)


def test_polygon_invariant_to_large_coordinate_offsets_and_vertex_order():
    points = np.array([[0, 0], [20, 0], [20, 10], [0, 10]], float)
    left = polygon_features(points)
    right = polygon_features(points[::-1]+[10000, 20000])
    np.testing.assert_allclose(right-left, [10000, 20000, 0, 0, 0], atol=1e-10)


def test_calibrated_body_center_depth_round_trip_and_no_board_plane():
    calibration = {"intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                   "camera_from_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]}
    features = np.zeros(len(FEATURE_NAMES))
    features[:2] = [.5, .5]
    for body in ([.02, -.03, .15], [-.1, .04, -.08]):
        target = metric_target(np.asarray(body), features, calibration)
        np.testing.assert_allclose(decode_metric(target, features, calibration), body, atol=1e-12)
    with pytest.raises(ValueError, match="depth"):
        decode_metric(np.array([0., 0., -1.]), features, calibration)


def test_kernel_fit_is_deterministic_and_pixels_define_features():
    inputs = np.zeros((20, len(FEATURE_NAMES)))
    inputs[:, 0] = np.linspace(0, 1, 20)
    targets = np.column_stack((inputs[:, 0]*.1, inputs[:, 0]*-.1, 1+inputs[:, 0]*.05))
    model = fit_kernel(inputs, targets, bandwidth=2, regularization=1e-5, maximum_centres=12,
                       weights=np.ones(20))
    assert model == fit_kernel(inputs, targets, bandwidth=2, regularization=1e-5,
                               maximum_centres=12, weights=np.ones(20))
    np.testing.assert_allclose(predict_kernel(model, inputs), targets, atol=5e-4)
    image = np.zeros((80, 100, 3), np.uint8)
    image[20:35, 40:60] = [10, 100, 240]
    mask = np.zeros((80, 100), bool)
    mask[20:35, 40:60] = True
    detection = Detection("rigid_flat_puck", (40, 20, 60, 35), mask, 1.)
    left = component_features(image, detection)
    # Class/id/confidence cannot change the image features; no truth field exists.
    right = component_features(image, replace(detection, class_id="withheld_changed", confidence=.5))
    np.testing.assert_array_equal(left, right)
    assert left.shape == (len(FEATURE_NAMES),)


def test_class_saturation_prevents_puck_board_merge():
    hsv = np.zeros((100, 100, 3), np.uint8)
    hsv[:] = [103, 169, 139]
    hsv[30:45, 40:60] = [107, 187, 240]
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    detector = ColorDetector({"rigid_cuboid": 11, "rigid_cylinder": 68, "rigid_flat_puck": 107, "button": 64},
                             hue_margin=8, minimum_pixels=8, minimum_saturation=90, minimum_value=25,
                             button_roi=(0, 0, 10, 10), class_minimum_saturation={"rigid_flat_puck": 175})
    output = detector.detect(rgb)
    assert len(output) == 1 and output[0].box == (40, 30, 60, 45)


def test_repair_test_split_blocked_before_model_or_images(monkeypatch, tmp_path):
    from dexvision.evaluation import evaluate_perception as module
    monkeypatch.setattr(module, "load_recipe", lambda _: {"allowed_evaluation_splits": ["validation"]})
    monkeypatch.setattr(module, "load_model", lambda _: pytest.fail("Test must be rejected before model/data access"))
    with pytest.raises(ValueError, match="held-out access is blocked"):
        module.evaluate_perception("repair.yaml", tmp_path/"must_not_exist", split="test")
    assert not (tmp_path/"must_not_exist").exists()


def test_learned_height_decoder_uses_variable_body_height():
    from dexvision.perception.height_model import decode_height, height_target
    calibration = {"intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                   "camera_from_world": [[1, 0, 0, 0], [0, 0, -1, .52], [0, 1, 0, 1], [0, 0, 0, 1]]}
    features = np.zeros(len(FEATURE_NAMES))
    features[:2] = [.5, .5]
    for position in ([.1, .1, .15], [-.1, .05, -.08]):
        target = height_target(np.asarray(position), features, calibration)
        np.testing.assert_allclose(decode_height(target, features, calibration), position, atol=1e-12)


def test_tree_generalizes_piecewise_height_without_truth_at_inference():
    from dexvision.perception.height_model import fit_tree, predict_tree
    inputs = np.zeros((40, len(FEATURE_NAMES)))
    inputs[:, 0] = np.linspace(0, 1, 40)
    targets = np.zeros((40, 3))
    targets[:, 2] = np.where(inputs[:, 0] < .5, .18, -.82)
    weights = np.ones(40)
    model = fit_tree(inputs, targets, weights, maximum_depth=8, minimum_leaf_samples=4)
    assert model == fit_tree(inputs, targets, weights, maximum_depth=8, minimum_leaf_samples=4)
    queries = inputs[[4, 34]].copy()
    queries[:, 4] = 99.  # unrelated silhouette moment cannot override selected split
    np.testing.assert_allclose(predict_tree(model, queries)[:, 2], [.18, -.82])


def test_temporal_rgb_context_is_causal_gap_masked_and_reset():
    from dexvision.perception.body_center_model import BodyCenterEstimator, temporal_features
    current, old = np.ones(16), np.arange(16.)
    populated = temporal_features(current, 1.2, (1., old), maximum_gap_s=.5)
    np.testing.assert_array_equal(populated[:16], current)
    np.testing.assert_allclose(populated[16:18], [1., .4])
    np.testing.assert_array_equal(populated[18:], old)
    for prior in (None, (1.3, old), (1.2, old), (.6, old)):
        missing = temporal_features(current, 1.2, prior, maximum_gap_s=.5)
        np.testing.assert_array_equal(missing[16:], np.zeros(18))
    estimator = BodyCenterEstimator({}, {}, maximum_gap_s=.5)
    estimator.feature_history['visual/0'] = 1., old
    estimator.previous['visual/0'] = 1., np.ones(3)
    estimator.reset()
    assert estimator.feature_history == {} and estimator.previous == {}


def test_reducing_detection_area_does_not_inflate_confidence():
    rgb = np.zeros((30, 30, 3), np.uint8)
    hsv = np.array([[[11, 220, 240]]], np.uint8)
    rgb[10:12, 10:12] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    options = dict(hue_margin=3, minimum_saturation=90, minimum_value=25,
                   button_roi=(0, 0, 2, 2), confidence_area_pixels=32)
    small = ColorDetector({'rigid_cuboid': 11, 'rigid_cylinder': 68, 'rigid_flat_puck': 107}, minimum_pixels=1, **options).detect(rgb)
    larger = ColorDetector({'rigid_cuboid': 11, 'rigid_cylinder': 68, 'rigid_flat_puck': 107}, minimum_pixels=3, **options).detect(rgb)
    assert len(small) == len(larger) == 1
    assert small[0].confidence == larger[0].confidence < .8


def test_repair_filters_test_annotations_before_json_decode(monkeypatch, tmp_path):
    import json
    from dexvision.perception import perception_baseline as module
    root = tmp_path/'visual'
    root.mkdir()
    release_dir = tmp_path/'release'
    (release_dir/'splits').mkdir(parents=True)
    owner = {'episode_id': 'train/1', 'split': 'train', 'recording_session_id': 's1', 'source_id': 'e1'}
    row = {'split': 'train', 'source_episode_id': 'train/1', 'recording_session_id': 's1',
           'source_id': 'e1', 'image_sha256': 'unique', 'source_frame_index': 0}
    # The TEST row is deliberately invalid JSON after its split field. Decoding
    # it would fail: repair must skip the entire row before annotation decoding.
    (root/'frames.jsonl').write_text(json.dumps(row)+'\n'+ '{"split":"test",INVALID_PRIVILEGED_LABELS}\n')
    (root/'calibration.json').write_text('{}')
    manifest = {'files': {str(root/name): {'sha256': module.digest_file(root/name)}
                         for name in ('frames.jsonl', 'calibration.json')}}
    (release_dir/'manifest.json').write_text(json.dumps(manifest))
    for split, episodes in (('train', [owner]), ('validation', [])):
        (release_dir/'splits'/f'{split}.json').write_text(json.dumps({'episodes': episodes}))
    recipe = {'visual_root': str(root), 'source_manifest': str(release_dir/'manifest.json'),
              'input_digests': {str(p): module.digest_file(p) for p in (release_dir/'splits').iterdir()}}
    rows = module.visual_rows(recipe, allowed_splits=('train', 'validation'))
    assert len(rows) == 1 and rows[0]['split'] == 'train'
    assert not (release_dir/'splits'/'test.json').exists()


def test_tree_adjacent_float_split_never_produces_empty_child():
    from dexvision.perception.height_model import fit_tree, predict_tree
    high = np.nextafter(1., np.inf)
    x = np.array([[1.], [high]])
    y = np.array([[0., 0., 0.], [0., 0., 1.]])
    fitted = fit_tree(x, y, np.ones(2), maximum_depth=20, minimum_leaf_samples=1)
    assert fitted['root']['left']['training_rows'] == 1
    assert fitted['root']['right']['training_rows'] == 1
    np.testing.assert_array_equal(predict_tree(fitted, x), y)


def test_rgb_area_association_avoids_background_fragment_stealing_target_id():
    from dexvision.perception.object_tracker import ObjectTracker
    def component(box, area):
        mask = np.zeros((30, 30), bool)
        mask.flat[:area] = True
        return Detection('rigid_flat_puck', box, mask, 1.)
    tracker = ObjectTracker(maximum_distance_px=60., maximum_gap_s=.51,
                            ambiguity_margin_px=3., area_ratio_weight_px=20.)
    initial = tracker.update((component((256, 265, 280, 275), 40),
                              component((256, 252, 281, 262), 87)), 1.)
    enlarged = tracker.update((component((252, 257, 282, 282), 640),), 1.17)
    assert enlarged[0].track_id == initial[1].track_id
    assert not enlarged[0].ambiguous


def test_puck_value_filter_rejects_dark_board_without_changing_other_class():
    hsv = np.full((50, 60, 3), [103, 180, 139], np.uint8)
    hsv[20:30, 20:40] = [107, 190, 240]
    hsv[5:10, 5:10] = [11, 210, 100]
    detector = ColorDetector({'rigid_cuboid': 11, 'rigid_cylinder': 68, 'rigid_flat_puck': 107},
        hue_margin=8, minimum_pixels=3, minimum_saturation=90, minimum_value=25,
        button_roi=(0, 0, 2, 2), class_minimum_value={'rigid_flat_puck': 140})
    components = detector.detect(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    assert {d.class_id for d in components} == {'rigid_cuboid', 'rigid_flat_puck'}
    assert next(d for d in components if d.class_id == 'rigid_flat_puck').box == (20, 20, 40, 30)


def test_rgb_fragment_union_restores_occluded_box_with_bounded_extent():
    from dexvision.perception.object_detector import merge_fragments
    def fragment(box):
        mask = np.zeros((60, 90), bool)
        x0,y0,x1,y1 = box
        mask[y0:y1, x0:x1] = True
        return Detection('rigid_flat_puck', box, mask, .2)
    detections = (fragment((10,10,22,20)), fragment((10,21,25,28)), fragment((70,10,80,20)))
    merged = merge_fragments(detections, {'rigid_flat_puck': {'maximum_gap_px': 2, 'maximum_extent_px': 32}}, 32)
    assert len(merged) == 2 and merged[0].box == (10,10,25,28)
    assert merged[0].mask.sum() == 225 and not merged[0].mask[20].any()
    assert merged[0].confidence == 1.
    bounded = merge_fragments(detections, {'rigid_flat_puck': {'maximum_gap_px': 60, 'maximum_extent_px': 32}}, 32)
    assert len(bounded) == 2


def test_temporal_elapsed_feature_ignores_numerical_jitter_but_keeps_real_gap():
    from dexvision.perception.body_center_model import temporal_features
    feature = np.zeros(16)
    first = temporal_features(feature,.17,(0.,feature),maximum_gap_s=.51)
    jitter = temporal_features(feature,.17000000000003,(0.,feature),maximum_gap_s=.51)
    np.testing.assert_array_equal(first,jitter)
    different = temporal_features(feature,.34,(0.,feature),maximum_gap_s=.51)
    assert different[17] != first[17]


def test_height_nearest_and_extended_kernel_use_rgb_feature_dimensions():
    from dexvision.perception.height_model import fit_nearest, predict_nearest
    x = np.zeros((12,34))
    x[:,33]=np.arange(12.)
    y = np.column_stack((x[:,33]*.01,x[:,33]*.02,x[:,33]*.03))
    model=fit_nearest(x,y,np.ones(12),neighbours=1)
    np.testing.assert_allclose(predict_nearest(model,x[[3,8]]),y[[3,8]])
    kernel=fit_kernel(x,y,bandwidth=.5,regularization=1e-8,maximum_centres=12,weights=np.ones(12))
    np.testing.assert_allclose(predict_kernel(kernel,x),y,atol=1e-3)
