"""Localization availability must never certify identity or fabricate motion."""

from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from dexvision.perception.context_motion import FEATURE_VERSION, context_features
from dexvision.perception.dual_context import (
    FeatureHistory,
    DualContextPose,
    INFERENCE_VARIANT,
    current_frame_features,
    compose_history,
)
from dexvision.evaluation.perception_dual_queries import predict_batch
from dexvision.perception.context_motion_network import predict_network
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_tracker import Track
from dexvision.perception.object_observations import ObjectObservationError
from dexvision.perception.world_estimate import inferred_world_estimate
from dexvision.perception.observation_boundary import evidence_from_pose


def sample():
    rgb = np.zeros((80, 100, 3), np.uint8)
    mask = np.zeros((80, 100), bool)
    mask[38:42, 48:52] = True
    rgb[mask] = [200, 90, 20]
    return rgb, Detection("rigid_cuboid", (48, 38, 52, 42), mask, 1.0)


def head(height):
    return {
        "kind": "rgb_motion_world_height_mlp",
        "feature_version": FEATURE_VERSION,
        "mean": [0.0] * 706,
        "spread": [1.0] * 706,
        "class_index": 0,
        "layers": [{"weight": np.zeros((3, 710)).tolist(), "bias": [0.0, 0.0, height]}],
        "target_mean": [0.0] * 3,
        "target_spread": [1.0] * 3,
    }


def pose_model():
    return {
        "inference_variant": INFERENCE_VARIANT,
        "body_center_models": {"rigid_cuboid": head(2.0)},
        "current_frame_models": {"rigid_cuboid": head(1.0)},
    }


def pose():
    return DualContextPose(
        {"intrinsics": np.eye(3).tolist(), "camera_from_world": np.eye(4).tolist()},
        pose_model(),
        0.51,
    )


def test_cached_current_composition_is_exact_old_runtime_with_actual_gaps():
    rgb, d = sample()
    prior = None
    state = FeatureHistory(0.51)
    for t in (1.0, 1.17, 1.34, 2.0, 2.17):
        rgb = rgb.copy()
        rgb[1:8, 1:9] = round(t * 30)
        expected, current = context_features(rgb, d, t, prior, maximum_gap_s=0.51)
        _, independent = context_features(rgb, d, t, None, maximum_gap_s=0.51)
        np.testing.assert_array_equal(
            compose_history(independent, t, prior, 0.51), expected
        )
        x, temporal = state.observe(rgb, (Track("visual/0", d, t, False, False),))[0]
        assert temporal == bool(expected[16])
        np.testing.assert_array_equal(
            x, expected if temporal else current_frame_features(expected)
        )
        prior = t, current


def test_current_only_removes_all_history_without_changing_current_pixels():
    x = np.arange(706, dtype=float)
    y = current_frame_features(x)
    np.testing.assert_array_equal(y[:16], x[:16])
    np.testing.assert_array_equal(y[34:370], x[34:370])
    assert not y[16:34].any() and not y[370:].any()
    assert x[16] != 0
    with pytest.raises(ValueError):
        current_frame_features(np.zeros(370))


def test_ambiguous_identity_has_current_position_invalid_evidence_and_no_motion():
    rgb, d = sample()
    p = pose()
    track = Track("visual/0", d, 1.0, True, False)
    p.features.history["visual/0"] = (0.9, np.ones(370) * 100)
    p.features.classes["visual/0"] = "rigid_cuboid"
    p.previous["visual/0"] = (0.9, np.ones(3) * 100)
    result = p.estimate_frame(rgb, (track,))[0]
    assert result.track.ambiguous and result.position[2] == pytest.approx(0.1)
    assert result.linear_velocity is None and result.orientation_wxyz is None
    e = evidence_from_pose(result)
    assert (
        e.position.value is not None
        and not e.position.valid
        and e.position.availability == "hypothesis"
    )
    assert e.object_id.value is None and not p.features.history and not p.previous
    with pytest.raises(ObjectObservationError):
        e.require_observation(now=1.0)
    with pytest.raises(ObjectObservationError):
        inferred_world_estimate((result,), 1.0).require_control_state(now=1.0)
    # Arbitrary withheld prior position/history cannot change current localization.
    q = pose()
    other = q.estimate_frame(rgb, (track,))[0]
    assert other.position == result.position


def test_ambiguity_clears_competing_same_class_history_and_recovery_is_cold():
    rgb, d = sample()
    h = FeatureHistory(0.51)
    h.observe(rgb, (Track("a", d, 1.0, False, False), Track("b", d, 1.0, False, False)))
    xs = h.observe(
        rgb, (Track("a", d, 1.1, True, False), Track("b", d, 1.1, False, False))
    )
    assert not any(t for _, t in xs) and not h.history
    xs = h.observe(rgb, (Track("b", d, 1.2, False, False),))
    assert not xs[0][1]
    assert h.observe(rgb, (Track("b", d, 1.3, False, False),))[0][1]


def test_temporal_route_gap_switch_and_reset_never_fabricate_velocity():
    rgb, d = sample()
    p = pose()
    result = []
    for t in (1.0, 1.1, 1.2, 2.0):
        result.append(p.estimate_frame(rgb, (Track("a", d, t, False, False),))[0])
    assert [r.position[2] for r in result] == pytest.approx([0.1, 0.2, 0.2, 0.1])
    assert result[0].linear_velocity is None and result[1].linear_velocity is None
    assert result[2].linear_velocity == pytest.approx((0.0, 0.0, 0.0))
    assert result[3].linear_velocity is None
    p.reset()
    assert not p.features.history and not p.previous


def test_current_stress_cannot_use_temporal_head_or_prior_geometry():
    rgb, d = sample()
    p = pose()
    for t in (1.0, 1.1, 1.2):
        r = p.estimate_frame(
            rgb, (Track("a", d, t, False, False),), force_current=True
        )[0]
        assert r.position[2] == pytest.approx(0.1) and r.linear_velocity is None


def test_portable_dual_heads_are_not_mutated_and_missing_variant_rejects():
    model = pose_model()
    frozen = deepcopy(model)
    p = DualContextPose(
        {"intrinsics": np.eye(3).tolist(), "camera_from_world": np.eye(4).tolist()},
        model,
        0.51,
    )
    rgb, d = sample()
    p.estimate_frame(rgb, (Track("a", d, 1.0, False, False),))
    assert model == frozen
    model.pop("inference_variant")
    with pytest.raises(ValueError, match="contract"):
        DualContextPose({}, model, 0.51)


def test_batch_query_predictor_matches_portable_scalar():
    rng = np.random.default_rng(91)
    m = head(1.0)
    m["layers"] = [
        {
            "weight": rng.normal(size=(8, 710)).tolist(),
            "bias": rng.normal(size=8).tolist(),
        },
        {
            "weight": rng.normal(size=(3, 8)).tolist(),
            "bias": rng.normal(size=3).tolist(),
        },
    ]
    x = rng.normal(size=(9, 706))
    np.testing.assert_allclose(
        predict_batch(m, x),
        np.stack([predict_network(m, v) for v in x]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_query_phases_require_support_recall_and_pose_coverage(tmp_path):
    import json
    from dexvision.evaluation.perception_dual_queries import score_query_cache

    cal = {"intrinsics": np.eye(3).tolist(), "camera_from_world": np.eye(4).tolist()}
    (tmp_path / "calibration.json").write_text(json.dumps(cal))
    cache = tmp_path / "queries.npz"

    def save(matched, phase="transport"):
        np.savez(
            cache,
            features=np.zeros((8, 706)),
            targets=np.tile([0.0, 0.0, 1.0], (8, 1)),
            matched=matched,
            phase=np.full(8, phase),
            source=np.array([str(i % 4) for i in range(8)]),
            temporal=np.zeros(8, bool),
            ambiguous=np.zeros(8, bool),
        )

    recipe = {
        "visual_root": str(tmp_path),
        "validation_query_cache": str(cache),
        "query_phases": ["transport"],
    }
    save(np.ones(8, bool))
    assert score_query_cache(recipe, pose_model())["passed"]
    save(np.array([False] + [True] * 7))
    assert not score_query_cache(recipe, pose_model())["passed"]
    save(np.ones(8, bool), "release")
    assert not score_query_cache(recipe, pose_model())["passed"]


def test_cadence_selection_uses_only_original_frame_index_and_declared_phase():
    from dexvision.logging.perception_cadence_windows import query_indices

    phases = np.array(["transport"] * 11 + ["release"] * 5 + ["unselected"] * 5)
    assert query_indices(phases, ["transport", "release"], 5) == [0, 5, 10, 15]
    assert query_indices(phases, ["release"], 5) == [15]
    with pytest.raises(ValueError):
        query_indices(phases, ["release"], True)


@pytest.mark.parametrize("failure", ["current_stress", "incomplete_second_head"])
def test_fresh_freeze_blocks_failed_stress_or_partial_budget_before_pixels(
    tmp_path, monkeypatch, failure
):
    import json
    import yaml
    import dexvision.evaluation.perception_dual as module
    import dexvision.learning.perception_context as gate

    report = {
        "strict_validation_pass": True,
        "additional_motion_phase_validation": {"passed": True},
        "current_frame_stress_validation": {"passed": failure != "current_stress"},
    }
    (tmp_path / "validation.json").write_text(json.dumps(report))
    recipe = {
        "model_dir": str(tmp_path),
        "engine": FEATURE_VERSION,
        "network_candidates": [{"seed": s, "epochs": 3000} for s in (3, 7, 11)],
    }
    for seed in (3, 7, 11):
        h = {"rigid_cuboid": {"seed": seed, "epochs": 3000}}
        (tmp_path / f"candidate_seed{seed}.json").write_text(
            json.dumps(
                {
                    "temporal_train_loss": [0.0] * 3000,
                    "current_train_loss": [0.0]
                    * (
                        2999
                        if failure == "incomplete_second_head" and seed == 7
                        else 3000
                    ),
                    "model": {"body_center_models": h, "current_frame_models": h},
                }
            )
        )
    monkeypatch.setattr(module, "selected_context", lambda config: (recipe, {}))
    monkeypatch.setattr(gate, "strict_validation", lambda report: True)
    spec = {
        "selected_recipe": "unused",
        "selected_engine": FEATURE_VERSION,
        "manual_review_sampling": {
            "repetitions": [0, 1],
            "include_worst_primary_pose_per_class": True,
        },
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="stress|Incomplete"):
        module.freeze_context_confirmation(config, tmp_path / "never_created")
    assert not (tmp_path / "never_created").exists()


def test_repeated_development_views_cannot_inflate_validation_coverage():
    from dexvision.learning.perception_dual_examples import unique_queries

    value = (
        "rigid_cuboid",
        np.zeros(706),
        np.zeros(3),
        "physical/1",
        "release",
        True,
        False,
        False,
        51,
    )
    assert len(unique_queries([value, value])) == 1
    changed = list(value)
    changed[1] = np.ones(706)
    with pytest.raises(ValueError, match="changed exact"):
        unique_queries([value, tuple(changed)])
    changed = list(value)
    changed[3] = "physical/2"
    assert len(unique_queries([value, tuple(changed)])) == 2


def test_display_projection_does_not_change_pose_or_history():
    from types import SimpleNamespace
    from dexvision.evaluation.perception_projection import (
        review_dual_overlay,
        CalibratedProjection,
    )

    rgb, d = sample()
    before = rgb.copy()
    left, right = pose(), pose()
    track = Track("a", d, 1.0, False, False)
    a = left.estimate_frame(rgb, (track,))
    right.estimate_frame(rgb, (track,))
    baseline = SimpleNamespace(pose=left, model=pose_model())
    row = {
        "source_episode_id": "synthetic",
        "source_frame_index": 0,
        "timestamp": 1.0,
        "condition": "nominal",
    }
    overlay = review_dual_overlay(rgb, a, row, 0.01, baseline)
    assert overlay.shape == rgb.shape and np.array_equal(rgb, before)
    np.testing.assert_allclose(
        CalibratedProjection(left.calibration).project(np.asarray([a[0].position])),
        [[49.5, 39.5]],
    )
    track = replace(track, timestamp=1.1)
    a = left.estimate_frame(rgb, (track,))[0]
    b = right.estimate_frame(rgb, (track,))[0]
    assert a.position == b.position and a.linear_velocity == b.linear_velocity


@pytest.mark.parametrize("engine", ["dual", "current"])
def test_complete_dual_review_export_smoke_without_gui(tmp_path, engine):
    import json
    import cv2
    from dexvision.evaluation.perception_dual_review import score_context
    from dexvision.logging.visual_stream import digest_file

    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[238:242, 318:322] = [200, 90, 20]
    path = tmp_path / "rgb.png"
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cal = {
        "intrinsics": np.eye(3).tolist(),
        "camera_from_world": np.eye(4).tolist(),
        "image_size": [640, 480],
    }
    (tmp_path / "calibration.json").write_text(json.dumps(cal))
    tracker = {
        "maximum_distance_px": 60.0,
        "maximum_gap_s": 0.51,
        "ambiguity_margin_px": 3.0,
    }
    model = {
        **pose_model(),
        "palette": {
            "rigid_cuboid": 11.0,
            "rigid_cylinder": 100.0,
            "rigid_flat_puck": 40.0,
            "button": 100.0,
        },
        "hue_margin": 8.0,
        "minimum_pixels": 1,
        "confidence_area_pixels": 32,
        "tracker_config": tracker,
    }
    if engine == "current":
        from dexvision.evaluation.perception_current import score_context
        from dexvision.perception.current_localization import ENGINE

        model["body_center_models"] = model.pop("current_frame_models")
        model.update(
            feature_version=ENGINE,
            inference_variant=ENGINE,
            localization_mode="current_only_no_motion",
        )
    recipe = {
        "visual_root": str(tmp_path),
        "minimum_pixels": 1,
        "minimum_saturation": 60,
        "minimum_value": 10,
        "button_roi_xyxy": [389, 136, 449, 196],
        "tracker": tracker,
        "predictive_tracker": {},
        "tracking_mode": "predictive",
        "review_fps": 6.0,
        "conditions": ["nominal"],
        "maximum_age_s": 0.1,
        "diagnostics": {"upright_cosine_min": 0.98},
        "coverage_limits": ["synthetic smoke only"],
        "review_scope": "validation",
    }
    label = {
        "class_id": "rigid_cuboid",
        "object_id": "cube",
        "visible": True,
        "box_xyxy": [318, 238, 322, 242],
        "pose_6d": {
            "translation_m": [31.95, 23.95, 0.1],
            "rotation_matrix": np.eye(3).tolist(),
        },
    }
    row = {
        "source_episode_id": "synthetic",
        "source_frame_index": 0,
        "timestamp": 1.0,
        "condition": "nominal",
        "split": "validation",
        "rgb": "rgb.png",
        "rgb_sha256": digest_file(path),
        "annotations": [label],
        "evaluation_only_source_owner": {
            "skill": "synthetic",
            "segments": [],
            "entity_id": "cube",
            "phase_intervals": [
                {"phase": "transport", "start_frame": 0, "end_frame": 1}
            ],
        },
    }
    out = tmp_path / "review"
    out.mkdir()
    report = score_context(recipe, model, [row], "validation", output_dir=out)
    assert (
        report["frame_count"] == 1
        and report["primary_object_metrics"]["position_hypothesis_count"] == 1
    )
    cap = cv2.VideoCapture(str(out / "sequence_00.avi"))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (480, 640)
    typed = json.loads((out / "typed_inferred_frames.json").read_text())
    assert typed[0]["accepted_control_state_blocked"]
    assert cv2.imread(str(out / "validation_contact_sheet.png")) is not None


@pytest.mark.parametrize("ambiguous", [False, True, "reacquired"])
def test_current_pose_is_exact_frozen_current_branch_without_motion(ambiguous):
    from dexvision.perception.current_localization import CurrentPose

    rgb, d = sample()
    old = pose()
    current = CurrentPose(old.calibration, pose_model()["current_frame_models"], 0.51)
    for timestamp in (1.0, 1.17, 2.0):
        track = Track("a", d, timestamp, ambiguous is True, ambiguous == "reacquired")
        a = old.estimate_frame(rgb, (track,), force_current=True)[0]
        b = current.estimate_frame(rgb, (track,))[0]
        assert (
            a.position == b.position
            and b.linear_velocity is None
            and b.orientation_wxyz is None
        )
        assert evidence_from_pose(b).position.valid == (not track.ambiguous)
    assert not current.previous and not current.feature_history


def test_current_feature_cache_is_only_verified_current_pixels_and_mask():
    from dexvision.perception.current_localization import CurrentPose

    rgb, d = sample()
    cache = {}
    cal = pose().calibration
    cached = CurrentPose(
        cal, pose_model()["current_frame_models"], 0.51, feature_cache=cache
    )
    plain = CurrentPose(cal, pose_model()["current_frame_models"], 0.51)
    for timestamp in (1.0, 2.0):
        t = Track("other", d, timestamp, True, False)
        a = cached.estimate_frame(rgb, (t,))[0]
        b = plain.estimate_frame(rgb, (t,))[0]
        assert a.position == b.position and a.linear_velocity is None
    assert len(cache) == 1 and all(not x.flags.writeable for x in cache.values())
    changed = rgb.copy()
    changed[0, 0] = 20
    cached.reset()
    cached.estimate_frame(changed, (Track("a", d, 3.0, False, False),))
    assert len(cache) == 2
