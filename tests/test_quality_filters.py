from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexvision.apps import filter_demos
from dexvision.logging.quality_filters import (
    DEFAULT_REPORT_NAME,
    QUALITY_REPORT_VERSION,
    QUALITY_THRESHOLDS_VERSION,
    QualityThresholds,
    evaluate_episode_quality,
    filter_demo_dataset,
)


def _write_episode(
    dataset: Path,
    name: str,
    *,
    tracking_confidence: float = 0.95,
    feature_confidence: float = 0.95,
    missing_frames: tuple[int, ...] = (),
    high_feature_jitter: bool = False,
    high_action_jerk: bool = False,
    joint_limit_hits: bool = False,
    workspace_limit_hits: bool = False,
    task_success: bool = True,
    skill_name: str = "reach_touch_target",
    task_id: str = "reach_touch_target",
) -> Path:
    episode = dataset / name
    episode.mkdir(parents=True)
    frame_count = 12

    metadata = {
        "episode_id": f"episode-{name}",
        "skill_name": skill_name,
        "task_id": task_id,
        "success": task_success,
        "tracking_quality_fields": [
            "detected",
            "handedness_code",
            "hand_tracking_confidence",
            "feature_confidence",
            "dropped_frame",
            "reacquired",
        ],
        "feature_fields": ["index_bend", "middle_bend", "ring_bend", "pinky_bend"],
        "finger_target_names": ["joint_a", "joint_b"],
        "action_schema": {
            "base_position_target": [0, 3],
            "base_orientation_target": [3, 7],
            "finger_actuator_targets": [7, 9],
        },
        "teleop_config": {
            "base_control": {
                "workspace_limits": {
                    "min": [-0.2, -0.2, 0.08],
                    "max": [0.2, 0.2, 0.24],
                }
            },
            "retargeting": {
                "static_targets": {
                    "joint_a": {"min": 0.0, "max": 1.0, "value": 0.5},
                    "joint_b": {"min": 0.0, "max": 1.0, "value": 0.5},
                },
                "fingers": {},
            },
        },
        "task_config": {},
    }
    if task_id == "push_cube_to_target":
        metadata["task_config"] = {
            "success_metric_inputs": [
                "object_position",
                "target_position",
                "distance_to_target",
                "dwell_steps",
            ],
            "success_dwell_steps": 5,
            "target_radius": 0.035,
            "resolved_object_id": "push_cube",
            "resolved_target_source": "push_target_center",
        }
    else:
        metadata["task_config"] = {
            "success_metric_inputs": [
                "target_position",
                "touch_position",
                "distance_to_target",
                "palm_contact",
            ]
        }
    (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    tracking = np.tile(
        [1.0, 1.0, tracking_confidence, feature_confidence, 0.0, 0.0],
        (frame_count, 1),
    )
    if missing_frames:
        tracking[list(missing_frames), 0] = 0.0
        tracking[list(missing_frames), 4] = 1.0
    np.save(episode / "tracking_quality.npy", tracking)

    features = np.full((frame_count, 4), 0.25, dtype=np.float64)
    if high_feature_jitter:
        features[1::2] = 1.0
        features[::2] = 0.0
    np.save(episode / "features.npy", features)

    actions = np.zeros((frame_count, 9), dtype=np.float64)
    actions[:, :3] = [0.0, 0.0, 0.16]
    actions[:, 3] = 1.0
    actions[:, 7:] = 0.5
    if high_action_jerk:
        actions[1::2, 7:] = 0.9
        actions[::2, 7:] = 0.1
    if joint_limit_hits:
        actions[:, 7:] = 0.0
    if workspace_limit_hits:
        actions[:, 0] = 0.2
    np.save(episode / "actions.npy", actions)

    if task_id == "push_cube_to_target":
        target = np.asarray([0.1, 0.0, -0.015], dtype=np.float64)
        distance = 0.01 if task_success else 0.05
        cube = target + np.asarray([distance, 0.0, 0.0])
        dwell = np.arange(1, frame_count + 1) if task_success else np.zeros(frame_count)
        task_states = np.stack(
            [
                np.concatenate((cube, target, [distance, float(dwell[index])]))
                for index in range(frame_count)
            ]
        )
        object_row = np.concatenate(
            (cube, [1.0, 0.0, 0.0, 0.0], np.zeros(6, dtype=np.float64))
        )
        np.save(
            episode / "object_states.npy",
            np.tile(object_row, (frame_count, 1)),
        )
        np.save(episode / "task_states.npy", task_states)
    else:
        target = np.asarray([0.1, 0.0, 0.5], dtype=np.float64)
        touch = target + np.asarray([0.01 if task_success else 0.05, 0.0, 0.0])
        distance = float(np.linalg.norm(touch - target))
        contact = 1.0 if task_success else 0.0
        task_row = np.concatenate((target, touch, [distance, contact]))
        np.save(episode / "task_states.npy", np.tile(task_row, (frame_count, 1)))
    return episode


def test_good_synthetic_demo_passes(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path, "good")

    result = evaluate_episode_quality(episode)

    assert result.passed is True
    assert result.failed_filters == ()
    assert result.metrics["recomputed_task_success"] is True


def test_low_confidence_and_missing_frames_fail(tmp_path: Path) -> None:
    episode = _write_episode(
        tmp_path,
        "low-confidence",
        tracking_confidence=0.4,
        missing_frames=(0, 1, 2),
    )

    result = evaluate_episode_quality(episode)

    assert result.passed is False
    assert "low_tracking_confidence" in result.failed_filters
    assert "too_many_missing_frames" in result.failed_filters


def test_high_feature_jitter_fails(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path, "jitter", high_feature_jitter=True)

    result = evaluate_episode_quality(episode)

    assert result.passed is False
    assert result.failed_filters == ("high_feature_jitter",)


def test_high_action_jerk_fails(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path, "jerk", high_action_jerk=True)

    result = evaluate_episode_quality(episode)

    assert result.passed is False
    assert result.failed_filters == ("high_action_jerk",)


def test_joint_and_workspace_limit_hits_fail(tmp_path: Path) -> None:
    episode = _write_episode(
        tmp_path,
        "limits",
        joint_limit_hits=True,
        workspace_limit_hits=True,
    )

    result = evaluate_episode_quality(episode)

    assert result.passed is False
    assert "too_many_joint_limit_hits" in result.failed_filters
    assert "workspace_limit_hits" in result.failed_filters


def test_fixed_workspace_axis_is_not_treated_as_a_limit_hit(tmp_path: Path) -> None:
    episode = _write_episode(
        tmp_path,
        "fixed-height",
        skill_name="push_cube_to_target",
        task_id="push_cube_to_target",
    )
    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["teleop_config"]["base_control"]["workspace_limits"] = {
        "min": [-0.22, -0.10, 0.16],
        "max": [0.22, 0.10, 0.16],
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = evaluate_episode_quality(episode)

    assert result.passed is True
    assert result.metrics["workspace_limit_hit_fraction"] == 0.0


def test_failed_task_demo_is_flagged(tmp_path: Path) -> None:
    episode = _write_episode(tmp_path, "failed-task", task_success=False)

    result = evaluate_episode_quality(episode)

    assert result.passed is False
    assert result.failed_filters == ("recomputed_task_failure",)


def test_report_is_grouped_versioned_and_preserves_raw_files(tmp_path: Path) -> None:
    first = _write_episode(tmp_path, "first", skill_name="reach")
    _write_episode(
        tmp_path,
        "second",
        task_success=False,
        skill_name="reach",
    )
    metadata_before = (first / "metadata.json").read_bytes()
    actions_before = (first / "actions.npy").read_bytes()

    exit_code = filter_demos.main(["--dataset", str(tmp_path)])

    saved = json.loads((tmp_path / DEFAULT_REPORT_NAME).read_text(encoding="utf-8"))
    assert exit_code == 0
    assert saved["version"] == QUALITY_REPORT_VERSION
    assert saved["threshold_version"] == QUALITY_THRESHOLDS_VERSION
    assert saved["raw_episodes_modified"] is False
    assert saved["episode_count"] == 2
    assert saved["pass_count"] == 1
    assert saved["fail_count"] == 1
    assert saved["groups"] == [
        {
            "episode_count": 2,
            "fail_count": 1,
            "pass_count": 1,
            "skill_name": "reach",
            "task_id": "reach_touch_target",
        }
    ]
    assert (first / "metadata.json").read_bytes() == metadata_before
    assert (first / "actions.npy").read_bytes() == actions_before


def test_thresholds_are_configurable(tmp_path: Path) -> None:
    _write_episode(tmp_path, "jitter", high_feature_jitter=True)
    thresholds = QualityThresholds(max_feature_jitter_p95=3.0)

    report = filter_demo_dataset(tmp_path, thresholds=thresholds)

    assert report.threshold_version == QUALITY_THRESHOLDS_VERSION
    assert report.thresholds["max_feature_jitter_p95"] == 3.0
    assert report.episodes[0].passed is True


def test_push_cube_demo_dispatches_through_quality_filters(tmp_path: Path) -> None:
    episode = _write_episode(
        tmp_path,
        "push-cube",
        skill_name="push_cube_to_target",
        task_id="push_cube_to_target",
    )

    result = evaluate_episode_quality(episode)

    assert result.task_id == "push_cube_to_target"
    assert result.passed is True
    assert result.metrics["recomputed_task_success"] is True
