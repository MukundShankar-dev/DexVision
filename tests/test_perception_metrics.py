"""Checkpoint 5.6 omissions, metric denominators and frozen artifacts."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dexvision.evaluation.perception_metrics import box_iou, match_detections, summarize_instances
from dexvision.logging.visual_stream import digest_file, write_json
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_pose import PoseEstimate
from dexvision.perception.object_tracker import Track
from dexvision.perception.perception_baseline import load_model, load_recipe, model_digest, read_rgb, visual_rows


def metric_row(**changes):
    return {"eligible": True, "matched": True, "position_error_m": .005,
            "position_delta_m": [.005, 0, 0], "velocity_error_mps": None,
            "symmetry_yaw_error_rad": None, "identity_switch": False,
            "reacquired": False, "certified_pose": False, "rejection_reason": "uncertified", **changes}


def test_iou_exclusive_edges_and_one_to_one_matching():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.
    assert box_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.
    detection = Detection("button", (0, 0, 10, 10), np.ones((10, 10), bool), 1.)
    estimate = PoseEstimate(Track("visual/0", detection, 0., False, False), None, None, None, None, None, None, "missing")
    labels = [{"visible": True, "class_id": "button", "box_xyxy": [0, 0, 10, 10]}]*2
    assert len(match_detections(labels, (estimate,))) == 1
    assert match_detections([{**labels[0], "visible": False}], (estimate,)) == {}


def test_omissions_cannot_inflate_recall_or_pose_availability():
    rows = [metric_row(), metric_row(matched=False, position_error_m=None, position_delta_m=None)]
    report = summarize_instances(rows, isolation_rejection_fraction=1.)
    assert report["recall"] == .5
    assert report["position_hypothesis_count"] == 1
    assert report["certified_typed_pose_count"] == 0
    assert not report["all_numerical_gates_pass"]
    assert report["missing_detections"] == 1


def test_empty_groups_and_no_pose_fail_numerical_gates():
    report = summarize_instances([], isolation_rejection_fraction=1.)
    assert report["recall"] is None
    assert not report["all_numerical_gates_pass"]
    report = summarize_instances([metric_row(position_error_m=None, position_delta_m=None)], isolation_rejection_fraction=1.)
    assert not report["numerical_gates"]["median_position"]


def test_identity_denominator_and_missing_frames_remain_visible():
    rows = [metric_row(identity_switch=True)] + [metric_row(matched=False, position_error_m=None,
                                                            position_delta_m=None)]*999
    report = summarize_instances(rows, isolation_rejection_fraction=1.)
    assert report["identity_switches_per_1000_visible_instances"] == 1.
    assert report["identity_matched_coverage"] == .001
    assert not report["numerical_gates"]["recall"]


def test_percentiles_and_velocity_units_are_separate():
    rows = [metric_row(position_error_m=.001*i, position_delta_m=[.001*i, 0, 0],
                       velocity_error_mps=.02) for i in range(100)]
    report = summarize_instances(rows, isolation_rejection_fraction=1.)
    assert report["position_error_p95_m"] == pytest.approx(.09405)
    assert report["linear_velocity_error_p95_mps"] == .02
    assert not report["numerical_gates"]["p95_position"]


def test_only_training_masks_can_fit_detector(tmp_path):
    with pytest.raises(ValueError, match="Only training"):
        read_rgb(tmp_path, {"split": "test"}, with_training_mask=True)


def test_recipe_lock_and_implementation_pins(tmp_path):
    implementation = tmp_path/"code.py"
    implementation.write_text("frozen", encoding="utf-8")
    config = tmp_path/"perception.yaml"
    recipe = {"version": "level5/perception-baseline-v1", "minimum_confidence": .8,
              "maximum_age_s": .1, "input_digests": {},
              "implementation_digests": {str(implementation): digest_file(implementation)}}
    config.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    config.with_suffix(".sha256").write_text(digest_file(config)+"  perception.yaml\n", encoding="utf-8")
    assert load_recipe(config)["version"] == recipe["version"]
    implementation.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation changed"):
        load_recipe(config)
    config.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_recipe(config)


def test_selected_artifact_and_exact_recipe_binding(tmp_path):
    recipe = {"model_dir": str(tmp_path), "input_digests": {}, "implementation_digests": {}, "_recipe_sha256": "recipe"}
    model = {"hue_margin": 10}
    write_json(tmp_path/"selected.json", model)
    selection = {"model_sha256": model_digest(model), "recipe_input_digests": {},
                 "implementation_digests": {}, "recipe_sha256": "recipe"}
    write_json(tmp_path/"selection.json", selection)
    assert load_model(recipe) == model
    with pytest.raises(ValueError, match="recipe changed"):
        load_model({**recipe, "_recipe_sha256": "new_recipe"})
    write_json(tmp_path/"selected.json", {"hue_margin": 11})
    with pytest.raises(ValueError, match="model changed"):
        load_model(recipe)


def test_visual_rows_enforce_released_episode_and_session_ownership(tmp_path):
    root = tmp_path/"visual"
    root.mkdir()
    release = tmp_path/"release"
    (release/"splits").mkdir(parents=True)
    row = {"source_episode_id": "e", "source_frame_index": 0, "recording_session_id": "s", "split": "train"}
    (root/"frames.jsonl").write_text(json.dumps(row)+"\n", encoding="utf-8")
    (root/"calibration.json").write_text("{}", encoding="utf-8")
    manifest = {"files": {str(root/name): {"sha256": digest_file(root/name)}
                          for name in ("frames.jsonl", "calibration.json")}}
    write_json(release/"manifest.json", manifest)
    digests = {}
    for split in ("train", "validation", "test"):
        path = release/"splits"/f"{split}.json"
        write_json(path, {"episodes": [{"episode_id": "e", "split": "validation", "recording_session_id": "s"}]
                                     if split == "validation" else []})
        digests[str(path)] = digest_file(path)
    with pytest.raises(ValueError, match="released episode"):
        visual_rows({"visual_root": str(root), "source_manifest": str(release/"manifest.json"), "input_digests": digests})


def test_heldout_reservation_cannot_be_reused_under_another_output(tmp_path, monkeypatch):
    from dexvision.evaluation import evaluate_perception as module
    recipe = {"model_dir": str(tmp_path/"model")}
    Path(str(tmp_path/"model")+"_heldout_attempt.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "load_recipe", lambda _: recipe)
    monkeypatch.setattr(module, "load_model", lambda _: {})
    monkeypatch.setattr(module, "visual_rows", lambda _: [])
    with pytest.raises(FileExistsError):
        module.evaluate_perception("unused", tmp_path/"different_output")
    assert not (tmp_path/"different_output").exists()
