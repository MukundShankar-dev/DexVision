from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dexvision.apps import record_demo
from dexvision.logging.dataset_schema import validate_demo
from dexvision.logging.demo_logger import (
    DemoLogger,
    DemoLoggerError,
    DemoStepData,
    build_level1_action_schema,
    build_level2_observation_schema,
    load_logged_demo,
)


ROOT = Path(__file__).resolve().parents[1]
TELEOP_CONFIG_PATH = ROOT / "configs" / "level1_teleop.yaml"


def _schemas(target_names: tuple[str, ...] = ("a0", "a1")):
    action_schema = build_level1_action_schema(target_names)
    observation_schema = build_level2_observation_schema(
        robot_qpos_dim=4,
        robot_qvel_dim=4,
        finger_target_dim=len(target_names),
        tracking_quality_dim=6,
    )
    return action_schema, observation_schema


def _metadata() -> dict:
    return {
        "skill_name": "free_space_gesture",
        "task_name": "Free Space Gesture",
        "task_id": "free_space_gesture",
        "episode_id": "demo_0001",
        "robot_model": "assets/mujoco/hand_scene.xml",
        "retargeter_config": "configs/level1_teleop.yaml",
        "control_rate_hz": 30.0,
        "teleop_config": {"base_control": {"enable_base_control": False}},
        "task_config": {
            "required_objects": (),
            "requires_task_state": False,
            "requires_success_metric_inputs": False,
            "required_observation_fields": (),
        },
    }


def _step(index: int, action_dim: int, target_dim: int) -> DemoStepData:
    action = np.zeros(action_dim, dtype=np.float64)
    action[3] = 1.0
    action[7 : 7 + target_dim] = index + 0.5
    return DemoStepData(
        features=np.full(14, index, dtype=np.float64),
        action=action,
        robot_state=np.full(4 + 4 + target_dim + 3 + 4, index, dtype=np.float64),
        tracking_quality=np.asarray([1.0, 0.0, 0.9, 0.8, 0.0, 0.0], dtype=np.float64),
        timestamp=float(index) / 30.0,
        landmarks=np.full((21, 3), index, dtype=np.float64),
    )


def test_demo_logger_saves_required_episode_files(tmp_path: Path) -> None:
    action_schema, observation_schema = _schemas()
    logger = DemoLogger(
        tmp_path / "demo",
        action_schema=action_schema,
        observation_schema=observation_schema,
    )
    logger.start_episode(_metadata())
    for index in range(3):
        logger.append(_step(index, action_schema.action_dim, target_dim=2))

    episode = logger.close(success=True)

    demo_dir = tmp_path / "demo"
    assert (demo_dir / "metadata.json").exists()
    assert (demo_dir / "features.npy").exists()
    assert (demo_dir / "actions.npy").exists()
    assert (demo_dir / "robot_states.npy").exists()
    assert (demo_dir / "tracking_quality.npy").exists()
    assert (demo_dir / "timestamps.npy").exists()
    assert (demo_dir / "landmarks.npy").exists()
    assert not (demo_dir / "camera.mp4").exists()

    metadata = json.loads((demo_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["success"] is True
    assert metadata["num_steps"] == 3
    assert metadata["action_schema"]["finger_actuator_targets"] == [7, 9]

    loaded = load_logged_demo(demo_dir)
    validate_demo(
        loaded,
        action_schema=action_schema,
        observation_schema=observation_schema,
    )
    assert episode.actions.shape == (3, action_schema.action_dim)
    assert loaded.features.shape[0] == loaded.actions.shape[0] == loaded.timestamps.shape[0]


def test_demo_logger_refuses_mixed_optional_landmarks(tmp_path: Path) -> None:
    action_schema, observation_schema = _schemas()
    logger = DemoLogger(
        tmp_path / "demo",
        action_schema=action_schema,
        observation_schema=observation_schema,
    )
    logger.start_episode(_metadata())
    logger.append(_step(0, action_schema.action_dim, target_dim=2))
    second = _step(1, action_schema.action_dim, target_dim=2)
    logger.append(
        DemoStepData(
            features=second.features,
            action=second.action,
            robot_state=second.robot_state,
            tracking_quality=second.tracking_quality,
            timestamp=second.timestamp,
            landmarks=None,
        )
    )

    with pytest.raises(DemoLoggerError, match="landmarks must be provided"):
        logger.close(success=None)


def test_demo_logger_rejects_empty_episode(tmp_path: Path) -> None:
    action_schema, observation_schema = _schemas()
    logger = DemoLogger(
        tmp_path / "demo",
        action_schema=action_schema,
        observation_schema=observation_schema,
    )
    logger.start_episode(_metadata())

    with pytest.raises(DemoLoggerError, match="empty demo episode"):
        logger.close(success=None)


def test_record_demo_parser_accepts_progress_command() -> None:
    parser = record_demo.build_parser()

    args = parser.parse_args(
        [
            "--task",
            "free_space_gesture",
            "--retargeter",
            "curl",
            "--show-camera-window",
            "--viewer",
            "--viewer-sleep",
            "0.01",
            "--gesture-label",
            "peace sign",
            "--require-hand-detected",
            "--min-hand-detected-frames",
            "2",
            "--output",
            "data/demos/free_space_gesture",
        ]
    )

    assert args.task == "free_space_gesture"
    assert args.skill_name is None
    assert args.gesture_label == "peace sign"
    assert args.retargeter == "curl"
    assert args.output == Path("data/demos/free_space_gesture")
    assert args.show_camera_window is True
    assert args.viewer is True
    assert args.viewer_sleep == pytest.approx(0.01)
    assert args.require_hand_detected is True
    assert args.min_hand_detected_frames == 2
    assert args.synthetic is False


def test_record_demo_normalizes_free_space_gesture_label() -> None:
    parser = record_demo.build_parser()
    args = parser.parse_args(["--task", "free_space_gesture", "--gesture-label", "Open Palm"])

    record_demo._validate_recording_args(args)

    assert args.gesture_label == "open_palm"


def test_record_demo_rejects_gesture_label_for_other_tasks() -> None:
    parser = record_demo.build_parser()
    args = parser.parse_args(["--task", "reach_touch_target", "--gesture-label", "fist"])

    with pytest.raises(ValueError, match="--gesture-label"):
        record_demo._validate_recording_args(args)


def test_level1_13_full_preset_enables_interactive_full_teleop_recording() -> None:
    parser = record_demo.build_parser()
    args = parser.parse_args(["--task", "free_space_gesture", "--level1-13-full"])

    record_demo._apply_recording_presets(args)
    record_demo._validate_recording_args(args)

    assert args.show_camera_window is True
    assert args.viewer is True
    assert args.enable_base_control is True
    assert args.enable_base_orientation is True
    assert args.auto_calibrate_base is False
    assert args.start_on_calibration is True
    assert args.enable_depth_control is True
    assert args.require_hand_detected is True
    assert args.min_hand_detected_frames == 10
    assert args.max_frames == 0


def test_start_on_calibration_requires_camera_preview() -> None:
    parser = record_demo.build_parser()
    args = parser.parse_args(["--task", "free_space_gesture", "--start-on-calibration"])

    with pytest.raises(ValueError, match="requires --show-camera-window"):
        record_demo._validate_recording_args(args)


def test_calibration_command_starts_recording_only_after_successful_base_calibration() -> None:
    assert (
        record_demo._calibration_started_recording(
            commands=("calibrate_base",),
            base_status=SimpleNamespace(neutral_captured=True),
            base_control_enabled=True,
        )
        is True
    )
    assert (
        record_demo._calibration_started_recording(
            commands=("calibrate_base",),
            base_status=SimpleNamespace(neutral_captured=False),
            base_control_enabled=True,
        )
        is False
    )
    assert (
        record_demo._calibration_started_recording(
            commands=(),
            base_status=None,
            base_control_enabled=True,
        )
        is False
    )


def test_recording_status_tells_operator_to_press_c_until_started() -> None:
    assert record_demo._format_recording_status(
        recording_started=False,
        recorded_frames=0,
        gesture_label="wave",
    ) == "recording=armed press-c-to-calibrate gesture=wave"
    assert record_demo._format_recording_status(
        recording_started=True,
        recorded_frames=12,
        gesture_label="wave",
    ) == "recording=on frames=12 gesture=wave"


def test_record_demo_synthetic_smoke_writes_episode(tmp_path: Path) -> None:
    output = tmp_path / "free_space_gesture"

    result = record_demo.main(
        [
            "--task",
            "free_space_gesture",
            "--retargeter",
            "curl",
            "--config",
            str(TELEOP_CONFIG_PATH),
            "--output",
            str(output),
            "--episode-id",
            "synthetic_0001",
            "--gesture-label",
            "fist",
            "--synthetic",
            "--max-frames",
            "3",
        ]
    )

    assert result == 0
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    actions = np.load(output / "actions.npy")
    timestamps = np.load(output / "timestamps.npy")
    tracking_quality = np.load(output / "tracking_quality.npy")

    assert metadata["task_id"] == "free_space_gesture"
    assert metadata["gesture_label"] == "fist"
    assert metadata["task_config"]["gesture_labels"] == [
        "open_palm",
        "fist",
        "point",
        "pinch",
        "peace_sign",
        "wave",
    ]
    assert metadata["recording"]["synthetic"] is True
    assert metadata["recording"]["start_on_calibration"] is False
    assert metadata["num_steps"] == 3
    assert actions.shape == (3, 27)
    assert timestamps.shape == (3,)
    assert tracking_quality.shape == (3, 6)
    assert np.allclose(np.linalg.norm(actions[:, 3:7], axis=1), 1.0)


def test_action_vector_preserves_full_base_and_actuator_targets() -> None:
    targets = {"a0": 0.2, "a1": 0.8}

    action = record_demo.action_vector(
        base_position=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        base_orientation=np.asarray([2.0, 0.0, 0.0, 0.0], dtype=np.float64),
        targets=targets,
        target_names=("a0", "a1"),
    )

    assert action.shape == (9,)
    assert action[:3] == pytest.approx([0.1, 0.2, 0.3])
    assert action[3:7] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert action[7:] == pytest.approx([0.2, 0.8])


def test_detection_guard_rejects_empty_manual_recording() -> None:
    parser = record_demo.build_parser()
    args = parser.parse_args(
        [
            "--task",
            "free_space_gesture",
            "--require-hand-detected",
            "--min-hand-detected-frames",
            "2",
        ]
    )
    summary = record_demo.RecordingSummary(frames=120, detected_frames=0)

    with pytest.raises(DemoLoggerError, match="required 2, got 0"):
        record_demo._validate_detection_guard(args=args, summary=summary)


def test_detection_guard_allows_enough_detected_frames() -> None:
    parser = record_demo.build_parser()
    args = parser.parse_args(
        [
            "--task",
            "free_space_gesture",
            "--require-hand-detected",
            "--min-hand-detected-frames",
            "2",
        ]
    )
    summary = record_demo.RecordingSummary(frames=120, detected_frames=2)

    record_demo._validate_detection_guard(args=args, summary=summary)


def test_preview_keys_map_to_level1_13_recording_commands() -> None:
    assert record_demo._preview_event_from_key(ord("q")).should_stop is True
    assert record_demo._preview_event_from_key(ord("c")).base_commands == ("calibrate_base",)
    assert record_demo._preview_event_from_key(ord("r")).base_commands == ("reset_base",)
    assert record_demo._preview_event_from_key(-1) == record_demo.RecordingPreviewEvent()
