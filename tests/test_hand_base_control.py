from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import check_hand_base_control, run_level1_teleop
from dexvision.features.hand_base import (
    HandBaseTarget,
    HandBaseTargetSmoother,
    ImagePalmCenterTarget,
    estimate_hand_base_target,
    estimate_hand_scale,
    estimate_image_palm_center_target,
    no_hand_base_target,
    no_image_palm_center_target,
    quaternion_angle,
    quaternion_multiply,
)
from dexvision.retargeting.curl_retargeter import load_curl_retargeter_config
from dexvision.sim.hand_base_control import (
    HandBaseControlConfig,
    HandBaseMocapController,
    WorkspaceLimits,
    format_hand_base_status,
    hand_base_config_from_mapping,
    hand_base_config_from_teleop_config,
    map_hand_scale_to_depth_target,
    map_image_delta_to_base_position,
    map_orientation_delta_to_robot,
    palm_center_delta,
    rotation_vector_to_quaternion,
)


ROOT = Path(__file__).resolve().parents[1]
TELEOP_CONFIG_PATH = ROOT / "configs" / "level1_teleop.yaml"
HAND_MODEL_PATH = ROOT / "assets" / "mujoco" / "hand_scene.xml"


class FakeMocapEnv:
    def __init__(self) -> None:
        self.position = np.zeros(3, dtype=np.float64)
        self.orientation_quat = np.asarray([0.0, 0.70710678, 0.0, 0.70710678])

    def get_mocap_pose(self, _body_name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.position.copy(), self.orientation_quat.copy()

    def set_mocap_pose(
        self,
        _body_name: str,
        *,
        position: np.ndarray,
        orientation_quat: np.ndarray,
    ) -> None:
        self.position = np.asarray(position, dtype=np.float64)
        self.orientation_quat = np.asarray(orientation_quat, dtype=np.float64)


def _open_hand_landmarks() -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [0.50, 0.70, 0.0]
    landmarks[5] = [0.38, 0.48, 0.0]
    landmarks[9] = [0.50, 0.42, -0.02]
    landmarks[17] = [0.65, 0.50, 0.0]
    landmarks[6] = [0.38, 0.35, 0.0]
    landmarks[7] = [0.38, 0.25, 0.0]
    landmarks[8] = [0.38, 0.15, 0.0]
    landmarks[10] = [0.50, 0.30, -0.02]
    landmarks[11] = [0.50, 0.20, -0.02]
    landmarks[12] = [0.50, 0.10, -0.02]
    landmarks[18] = [0.65, 0.38, 0.0]
    landmarks[19] = [0.65, 0.28, 0.0]
    landmarks[20] = [0.65, 0.18, 0.0]
    return landmarks


def _image_target(
    palm_center: tuple[float, float] = (0.5, 0.5),
    *,
    hand_scale: float = 0.25,
    confidence: float = 1.0,
    valid: bool = True,
) -> ImagePalmCenterTarget:
    return ImagePalmCenterTarget(
        palm_center=np.asarray(palm_center, dtype=np.float64),
        confidence=confidence,
        valid=valid,
        hand_scale=hand_scale,
    )


def _body_position(env: object, body_name: str) -> np.ndarray:
    body_id = env._mujoco.mj_name2id(  # type: ignore[attr-defined]
        env.model,  # type: ignore[attr-defined]
        env._mujoco.mjtObj.mjOBJ_BODY,  # type: ignore[attr-defined]
        body_name,
    )
    return np.asarray(env.data.xpos[body_id], dtype=np.float64).copy()  # type: ignore[attr-defined]


def _body_quaternion(env: object, body_name: str) -> np.ndarray:
    body_id = env._mujoco.mj_name2id(  # type: ignore[attr-defined]
        env.model,  # type: ignore[attr-defined]
        env._mujoco.mjtObj.mjOBJ_BODY,  # type: ignore[attr-defined]
        body_name,
    )
    return np.asarray(env.data.xquat[body_id], dtype=np.float64).copy()  # type: ignore[attr-defined]


def test_palm_pose_estimator_returns_valid_normalized_quaternion_and_axes() -> None:
    target = estimate_hand_base_target(_open_hand_landmarks(), confidence=0.87)

    assert target.valid
    assert target.confidence == pytest.approx(0.87)
    assert target.position.shape == (3,)
    assert target.orientation_quat.shape == (4,)
    assert np.linalg.norm(target.orientation_quat) == pytest.approx(1.0)

    rotation = target.orientation_matrix
    assert rotation.shape == (3, 3)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    for axis_index in range(3):
        assert np.linalg.norm(rotation[:, axis_index]) == pytest.approx(1.0)


def test_palm_pose_estimator_invalid_landmarks_fail_safely() -> None:
    target = estimate_hand_base_target(np.full((21, 3), np.nan), confidence=float("nan"))

    assert target == no_hand_base_target(confidence=0.0)
    assert not target.valid
    assert np.all(np.isfinite(target.position))
    assert np.all(np.isfinite(target.orientation_quat))


def test_palm_pose_estimator_uses_wrist_position_by_default() -> None:
    target = estimate_hand_base_target(_open_hand_landmarks(), confidence=1.0)

    assert target.valid
    assert target.position.tolist() == pytest.approx([0.0, 0.0, -0.2])


def test_palm_center_position_source_remains_available() -> None:
    landmarks = _open_hand_landmarks()
    palm_center = np.mean(landmarks[[0, 5, 9, 17]], axis=0)
    expected_position = np.asarray(
        [-palm_center[2], palm_center[0] - 0.5, -(palm_center[1] - 0.5)],
        dtype=np.float64,
    )

    target = estimate_hand_base_target(
        landmarks,
        confidence=1.0,
        position_source="palm_center",
    )

    assert target.valid
    assert np.allclose(target.position, expected_position)


def test_image_palm_center_estimator_returns_normalized_palm_mean() -> None:
    landmarks = _open_hand_landmarks()
    expected = np.mean(landmarks[[0, 5, 9, 17], :2], axis=0)

    target = estimate_image_palm_center_target(landmarks, confidence=0.8)

    assert target.valid
    assert target.confidence == pytest.approx(0.8)
    assert target.palm_center.tolist() == pytest.approx(expected.tolist())
    assert target.hand_scale == pytest.approx(
        estimate_hand_scale(landmarks, depth_source="robust_palm_scale")
    )


def test_hand_scale_estimator_supports_palm_width_and_robust_scale() -> None:
    landmarks = _open_hand_landmarks()
    palm_width = float(np.linalg.norm(landmarks[5, :2] - landmarks[17, :2]))
    wrist_middle = float(np.linalg.norm(landmarks[0, :2] - landmarks[9, :2]))
    mcp_chain = float(np.linalg.norm(landmarks[5, :2] - landmarks[9, :2])) + float(
        np.linalg.norm(landmarks[9, :2] - landmarks[17, :2])
    )

    assert estimate_hand_scale(landmarks, depth_source="palm_width") == pytest.approx(palm_width)
    assert estimate_hand_scale(landmarks, depth_source="robust_palm_scale") == pytest.approx(
        np.mean([palm_width, wrist_middle, mcp_chain])
    )


def test_image_palm_center_invalid_landmarks_fail_safely() -> None:
    landmarks = _open_hand_landmarks()
    landmarks[5, 0] = np.nan

    target = estimate_image_palm_center_target(landmarks, confidence=0.6)

    assert target == no_image_palm_center_target(confidence=0.6)
    assert not target.valid
    assert np.all(np.isfinite(target.palm_center))


def test_palm_center_delta_calculation() -> None:
    delta = palm_center_delta(
        np.asarray([0.62, 0.42], dtype=np.float64),
        np.asarray([0.50, 0.47], dtype=np.float64),
    )

    assert delta.tolist() == pytest.approx([0.12, -0.05])


def test_image_2d_mapping_uses_height_axis_by_default() -> None:
    config = HandBaseControlConfig(
        base_fixed_z=0.14,
        base_position_scale_x=0.5,
        base_position_scale_y=0.25,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-1.0, -1.0, 0.0]),
            maximum=np.asarray([1.0, 1.0, 1.0]),
        ),
    )

    position, clamped = map_image_delta_to_base_position(
        np.asarray([0.2, -0.4], dtype=np.float64),
        config,
    )

    assert not clamped
    assert position.tolist() == pytest.approx([0.0, 0.1, 0.24])


def test_image_2d_mapping_can_use_approach_axis() -> None:
    config = HandBaseControlConfig(
        base_fixed_z=0.14,
        base_position_scale_x=0.5,
        base_position_scale_y=0.25,
        base_image_y_axis="approach",
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-1.0, -1.0, 0.0]),
            maximum=np.asarray([1.0, 1.0, 1.0]),
        ),
    )

    position, clamped = map_image_delta_to_base_position(
        np.asarray([0.2, -0.4], dtype=np.float64),
        config,
    )

    assert not clamped
    assert position.tolist() == pytest.approx([0.1, 0.1, 0.14])


def test_image_2d_mapping_clamps_workspace() -> None:
    config = HandBaseControlConfig(
        base_fixed_z=0.14,
        base_position_scale_x=2.0,
        base_position_scale_y=2.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-0.05, -0.05, 0.1]),
            maximum=np.asarray([0.05, 0.05, 0.12]),
        ),
    )

    position, clamped = map_image_delta_to_base_position(
        np.asarray([0.5, -0.5], dtype=np.float64),
        config,
    )

    assert clamped
    assert position.tolist() == pytest.approx([0.0, 0.05, 0.12])


def test_orientation_delta_mapping_applies_axis_signs() -> None:
    config = HandBaseControlConfig(
        base_orientation_axis_signs=np.asarray([-1.0, 1.0, 1.0], dtype=np.float64)
    )
    human_delta = rotation_vector_to_quaternion(
        np.asarray([np.deg2rad(30.0), 0.0, 0.0], dtype=np.float64)
    )
    expected = rotation_vector_to_quaternion(
        np.asarray([np.deg2rad(-30.0), 0.0, 0.0], dtype=np.float64)
    )

    mapped = map_orientation_delta_to_robot(human_delta, config)

    assert quaternion_angle(mapped, expected) < 1e-6


def test_wrist_position_source_reduces_rotation_induced_translation_drift() -> None:
    open_landmarks = _open_hand_landmarks()
    rotated_or_curled = open_landmarks.copy()
    rotated_or_curled[[5, 9, 17]] += np.asarray([0.12, -0.08, 0.05], dtype=np.float32)

    open_target = estimate_hand_base_target(open_landmarks, confidence=1.0)
    moved_mcp_target = estimate_hand_base_target(rotated_or_curled, confidence=1.0)

    assert open_target.valid
    assert moved_mcp_target.valid
    assert np.allclose(moved_mcp_target.position, open_target.position)


def test_left_hand_orientation_is_canonicalized_to_right_hand_target() -> None:
    right_landmarks = _open_hand_landmarks()
    mirrored_left_landmarks = right_landmarks.copy()
    mirrored_left_landmarks[:, 0] = 1.0 - mirrored_left_landmarks[:, 0]

    right_target = estimate_hand_base_target(
        right_landmarks,
        confidence=1.0,
        handedness="Right",
    )
    left_target = estimate_hand_base_target(
        mirrored_left_landmarks,
        confidence=1.0,
        handedness="Left",
    )

    assert right_target.valid
    assert left_target.valid
    assert np.allclose(left_target.orientation_matrix, right_target.orientation_matrix)


def test_hand_base_target_smoother_smooths_position_and_orientation() -> None:
    first = HandBaseTarget(
        position=np.asarray([0.0, 0.0, 0.0]),
        orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
        confidence=1.0,
        valid=True,
    )
    second = HandBaseTarget(
        position=np.asarray([1.0, 0.0, 0.0]),
        orientation_quat=np.asarray([0.0, 1.0, 0.0, 0.0]),
        confidence=1.0,
        valid=True,
    )
    smoother = HandBaseTargetSmoother(alpha=0.5, min_confidence=0.2)

    assert smoother.update(first) == first
    smoothed = smoother.update(second)

    assert smoothed.position[0] == pytest.approx(0.5)
    assert np.linalg.norm(smoothed.orientation_quat) == pytest.approx(1.0)


def test_workspace_clamp_limits_position() -> None:
    limits = WorkspaceLimits(
        minimum=np.asarray([-1.0, -2.0, -3.0]),
        maximum=np.asarray([1.0, 2.0, 3.0]),
    )

    clamped, changed = limits.clamp(np.asarray([2.5, -5.0, 0.25]))

    assert changed
    assert clamped.tolist() == [1.0, -2.0, 0.25]


def test_base_control_config_defaults_preserve_fixed_base_behavior() -> None:
    raw_config = load_curl_retargeter_config(TELEOP_CONFIG_PATH)
    config = hand_base_config_from_teleop_config(raw_config)
    enabled_config = hand_base_config_from_teleop_config(raw_config, enable_override=True)
    parser = run_level1_teleop.build_parser()

    assert config.enabled is False
    assert enabled_config.enabled is True
    assert config.mocap_body_name == "dexvision_hand_base_target"
    assert config.base_control_mode == "image_2d"
    assert config.base_fixed_z == pytest.approx(0.14)
    assert config.base_position_scale_x == pytest.approx(0.45)
    assert config.base_position_scale_y == pytest.approx(0.35)
    assert config.base_image_y_axis == "height"
    assert config.enable_depth_control is True
    assert config.depth_source == "robust_palm_scale"
    assert config.depth_axis == "x"
    assert config.depth_scale == pytest.approx(0.35)
    assert config.depth_sign == pytest.approx(1.0)
    assert config.depth_min == pytest.approx(-0.12)
    assert config.depth_max == pytest.approx(0.16)
    assert config.depth_smoothing_alpha == pytest.approx(0.25)
    assert config.depth_deadband == pytest.approx(0.03)
    assert config.depth_hold_on_tracking_loss is True
    assert config.enable_base_orientation is False
    assert config.base_orientation_axis_signs.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert np.allclose(config.base_orientation_remap_matrix, np.eye(3))
    assert config.position_source == "wrist"
    assert config.position_mode == "absolute"
    assert config.rotation_mode == "fixed"
    assert config.base_smoothing_alpha == pytest.approx(0.25)
    assert config.max_position_step == pytest.approx(0.025)
    assert config.max_rotation_step_degrees == pytest.approx(3.0)
    assert parser.parse_args([]).enable_base_control is False
    assert parser.parse_args(["--enable-base-control"]).enable_base_control is True
    assert parser.parse_args([]).base_control_mode is None
    assert parser.parse_args(["--base-control-mode", "image_2d"]).base_control_mode == "image_2d"
    assert parser.parse_args([]).enable_base_orientation is False
    assert parser.parse_args(["--enable-base-orientation"]).enable_base_orientation is True
    assert parser.parse_args([]).enable_depth_control is None
    assert parser.parse_args(["--enable-depth-control"]).enable_depth_control is True
    assert parser.parse_args(["--disable-depth-control"]).enable_depth_control is False


def test_empty_base_control_mapping_uses_safe_image_2d_defaults() -> None:
    config = hand_base_config_from_mapping({})

    assert config.enabled is False
    assert config.base_control_mode == "image_2d"
    assert config.base_fixed_z == pytest.approx(0.14)
    assert config.base_image_y_axis == "height"
    assert config.enable_depth_control is False
    assert config.depth_source == "robust_palm_scale"
    assert config.depth_axis == "x"
    assert config.depth_scale == pytest.approx(0.35)
    assert config.depth_min == pytest.approx(-0.12)
    assert config.depth_max == pytest.approx(0.16)
    assert config.position_source == "wrist"
    assert config.position_mode == "absolute"
    assert config.rotation_mode == "fixed"
    assert config.enable_base_orientation is False
    assert config.base_smoothing_alpha == pytest.approx(0.25)
    assert config.max_position_step == pytest.approx(0.025)
    assert config.max_rotation_step_degrees == pytest.approx(3.0)


def test_base_control_config_rejects_unknown_position_source() -> None:
    with pytest.raises(ValueError, match="position_source"):
        hand_base_config_from_mapping({"position_source": "palm"})


def test_base_control_config_rejects_unknown_position_mode() -> None:
    with pytest.raises(ValueError, match="position_mode"):
        hand_base_config_from_mapping({"position_mode": "neutral"})


def test_base_control_config_rejects_unknown_base_control_mode() -> None:
    with pytest.raises(ValueError, match="base_control_mode"):
        hand_base_config_from_mapping({"base_control_mode": "world"})


def test_base_control_config_rejects_unknown_image_y_axis() -> None:
    with pytest.raises(ValueError, match="base_image_y_axis"):
        hand_base_config_from_mapping({"base_image_y_axis": "diagonal"})


def test_base_control_config_rejects_unknown_depth_source() -> None:
    with pytest.raises(ValueError, match="depth_source"):
        hand_base_config_from_mapping({"depth_source": "knuckle_area"})


def test_base_control_config_rejects_unknown_depth_axis() -> None:
    with pytest.raises(ValueError, match="depth_axis"):
        hand_base_config_from_mapping({"depth_axis": "diagonal"})


def test_base_control_config_rejects_invalid_orientation_axis_signs() -> None:
    with pytest.raises(ValueError, match="base_orientation_axis_signs"):
        hand_base_config_from_mapping({"base_orientation_axis_signs": [1.0, 0.0, 1.0]})


def test_check_hand_base_control_parser_enables_base_and_overlay_by_default() -> None:
    args = check_hand_base_control.build_parser().parse_args([])

    assert args.enable_base_control is True
    assert args.show_camera_window is True
    assert args.enable_depth_control is True


def test_image_2d_controller_requires_calibration_before_moving() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        base_fixed_z=0.14,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]

    status = controller.apply_image_2d(
        ImagePalmCenterTarget(
            palm_center=np.asarray([0.8, 0.2], dtype=np.float64),
            confidence=1.0,
            valid=True,
        )
    )

    assert status.tracking_valid
    assert not status.neutral_captured
    assert env.position.tolist() == pytest.approx([0.0, 0.0, 0.14])
    assert "cal=no" in format_hand_base_status(status)


def test_image_2d_controller_calibrates_and_maps_palm_delta() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        base_fixed_z=0.14,
        base_position_scale_x=0.5,
        base_position_scale_y=0.25,
        base_smoothing_alpha=1.0,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-1.0, -1.0, 0.0]),
            maximum=np.asarray([1.0, 1.0, 1.0]),
        ),
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]
    neutral = ImagePalmCenterTarget(
        palm_center=np.asarray([0.5, 0.5], dtype=np.float64),
        confidence=1.0,
        valid=True,
    )
    moved = ImagePalmCenterTarget(
        palm_center=np.asarray([0.7, 0.1], dtype=np.float64),
        confidence=1.0,
        valid=True,
    )

    assert controller.calibrate_image_2d(neutral)
    status = controller.apply_image_2d(moved)

    assert status.tracking_valid
    assert status.neutral_captured
    assert status.palm_delta is not None
    assert status.palm_delta.tolist() == pytest.approx([0.2, -0.4])
    assert env.position.tolist() == pytest.approx([0.0, 0.1, 0.24])
    assert "base=image_2d cal=yes tracking" in format_hand_base_status(status)


def test_image_2d_calibration_stores_neutral_hand_scale() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        enable_depth_control=True,
        base_smoothing_alpha=1.0,
        depth_smoothing_alpha=1.0,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]

    assert controller.calibrate_image_2d(_image_target(hand_scale=0.24))
    status = controller.apply_image_2d(_image_target(hand_scale=0.24))

    assert status.neutral_captured
    assert status.hand_scale == pytest.approx(0.24)
    assert status.neutral_hand_scale == pytest.approx(0.24)
    assert "scale=0.240 neutral=0.240" in format_hand_base_status(status)


def test_larger_hand_scale_moves_depth_in_configured_direction() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        enable_depth_control=True,
        depth_axis="x",
        depth_scale=0.4,
        depth_sign=1.0,
        depth_deadband=0.0,
        depth_min=-1.0,
        depth_max=1.0,
        base_smoothing_alpha=1.0,
        depth_smoothing_alpha=1.0,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-1.0, -1.0, -1.0]),
            maximum=np.asarray([1.0, 1.0, 1.0]),
        ),
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]

    assert controller.calibrate_image_2d(_image_target(hand_scale=0.20))
    status = controller.apply_image_2d(_image_target(hand_scale=0.30))

    assert status.depth_delta == pytest.approx(0.5)
    assert status.depth_target == pytest.approx(0.2)
    assert env.position[0] == pytest.approx(0.2)


def test_smaller_hand_scale_moves_depth_opposite_direction() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        enable_depth_control=True,
        depth_axis="x",
        depth_scale=0.4,
        depth_sign=1.0,
        depth_deadband=0.0,
        depth_min=-1.0,
        depth_max=1.0,
        base_smoothing_alpha=1.0,
        depth_smoothing_alpha=1.0,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-1.0, -1.0, -1.0]),
            maximum=np.asarray([1.0, 1.0, 1.0]),
        ),
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]

    assert controller.calibrate_image_2d(_image_target(hand_scale=0.20))
    status = controller.apply_image_2d(_image_target(hand_scale=0.10))

    assert status.depth_delta == pytest.approx(-0.5)
    assert status.depth_target == pytest.approx(-0.2)
    assert env.position[0] == pytest.approx(-0.2)


def test_depth_deadband_suppresses_tiny_scale_noise() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        enable_depth_control=True,
        depth_scale=1.0,
        depth_deadband=0.05,
        depth_min=-1.0,
        depth_max=1.0,
        base_smoothing_alpha=1.0,
        depth_smoothing_alpha=1.0,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-1.0, -1.0, -1.0]),
            maximum=np.asarray([1.0, 1.0, 1.0]),
        ),
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]

    assert controller.calibrate_image_2d(_image_target(hand_scale=0.20))
    status = controller.apply_image_2d(_image_target(hand_scale=0.208))

    assert status.depth_delta == pytest.approx(0.0)
    assert env.position[0] == pytest.approx(0.0)


def test_depth_clamp_limits_scale_derived_target() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        enable_depth_control=True,
        depth_scale=2.0,
        depth_deadband=0.0,
        depth_min=-0.05,
        depth_max=0.12,
        base_smoothing_alpha=1.0,
        depth_smoothing_alpha=1.0,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-1.0, -1.0, -1.0]),
            maximum=np.asarray([1.0, 1.0, 1.0]),
        ),
    )
    target, delta, clamped = map_hand_scale_to_depth_target(
        0.40,
        0.20,
        np.zeros(3, dtype=np.float64),
        config,
    )

    assert delta == pytest.approx(1.0)
    assert clamped
    assert target == pytest.approx(0.12)


def test_missing_tracking_holds_last_depth_without_nan_or_jump() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        enable_depth_control=True,
        depth_scale=0.4,
        depth_deadband=0.0,
        depth_min=-1.0,
        depth_max=1.0,
        base_smoothing_alpha=1.0,
        depth_smoothing_alpha=1.0,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-1.0, -1.0, -1.0]),
            maximum=np.asarray([1.0, 1.0, 1.0]),
        ),
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]

    assert controller.calibrate_image_2d(_image_target(hand_scale=0.20))
    controller.apply_image_2d(_image_target(hand_scale=0.30))
    last_position = env.position.copy()
    status = controller.apply_image_2d(no_image_palm_center_target())

    assert not status.tracking_valid
    assert np.all(np.isfinite(status.applied_target.position))
    assert env.position.tolist() == pytest.approx(last_position.tolist())


def test_image_2d_orientation_calibration_maps_relative_palm_delta() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        enable_base_orientation=True,
        base_smoothing_alpha=1.0,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]
    palm_center = ImagePalmCenterTarget(
        palm_center=np.asarray([0.5, 0.5], dtype=np.float64),
        confidence=1.0,
        valid=True,
    )
    calibrated_human = HandBaseTarget(
        position=np.zeros(3),
        orientation_quat=rotation_vector_to_quaternion(
            np.asarray([0.0, np.deg2rad(45.0), 0.0], dtype=np.float64)
        ),
        confidence=1.0,
        valid=True,
    )
    current_human = HandBaseTarget(
        position=np.zeros(3),
        orientation_quat=quaternion_multiply(
            rotation_vector_to_quaternion(
                np.asarray([0.0, 0.0, np.deg2rad(30.0)], dtype=np.float64)
            ),
            calibrated_human.orientation_quat,
        ),
        confidence=1.0,
        valid=True,
    )
    expected_robot_quat = quaternion_multiply(
        rotation_vector_to_quaternion(
            np.asarray([0.0, 0.0, np.deg2rad(30.0)], dtype=np.float64)
        ),
        env.orientation_quat,
    )

    assert controller.calibrate_image_2d(palm_center, orientation_target=calibrated_human)
    status = controller.apply_image_2d(palm_center, orientation_target=current_human)

    assert status.tracking_valid
    assert status.neutral_captured
    assert status.orientation_enabled
    assert status.orientation_calibrated
    assert status.orientation_delta_rpy_degrees is not None
    assert status.orientation_delta_rpy_degrees[2] == pytest.approx(30.0)
    assert quaternion_angle(env.orientation_quat, expected_robot_quat) < 1e-6
    assert "ori r=" in format_hand_base_status(status)


def test_image_2d_orientation_requires_orientation_target_when_enabled() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        enable_base_orientation=True,
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]
    palm_center = ImagePalmCenterTarget(
        palm_center=np.asarray([0.5, 0.5], dtype=np.float64),
        confidence=1.0,
        valid=True,
    )

    assert not controller.calibrate_image_2d(palm_center)


def test_image_2d_orientation_axis_signs_flip_relative_rotation() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        enable_base_orientation=True,
        base_orientation_axis_signs=np.asarray([-1.0, 1.0, 1.0], dtype=np.float64),
        base_smoothing_alpha=1.0,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]
    palm_center = ImagePalmCenterTarget(
        palm_center=np.asarray([0.5, 0.5], dtype=np.float64),
        confidence=1.0,
        valid=True,
    )
    neutral_orientation = HandBaseTarget(
        position=np.zeros(3),
        orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        confidence=1.0,
        valid=True,
    )
    current_orientation = HandBaseTarget(
        position=np.zeros(3),
        orientation_quat=rotation_vector_to_quaternion(
            np.asarray([np.deg2rad(30.0), 0.0, 0.0], dtype=np.float64)
        ),
        confidence=1.0,
        valid=True,
    )
    expected_robot_quat = quaternion_multiply(
        rotation_vector_to_quaternion(
            np.asarray([np.deg2rad(-30.0), 0.0, 0.0], dtype=np.float64)
        ),
        env.orientation_quat,
    )

    assert controller.calibrate_image_2d(palm_center, orientation_target=neutral_orientation)
    controller.apply_image_2d(palm_center, orientation_target=current_orientation)

    assert quaternion_angle(env.orientation_quat, expected_robot_quat) < 1e-6


def test_image_2d_controller_holds_last_target_when_tracking_is_lost() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        base_smoothing_alpha=1.0,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]
    controller.calibrate_image_2d(
        ImagePalmCenterTarget(
            palm_center=np.asarray([0.5, 0.5], dtype=np.float64),
            confidence=1.0,
            valid=True,
        )
    )
    controller.apply_image_2d(
        ImagePalmCenterTarget(
            palm_center=np.asarray([0.6, 0.4], dtype=np.float64),
            confidence=1.0,
            valid=True,
        )
    )
    last_position = env.position.copy()

    status = controller.apply_image_2d(no_image_palm_center_target())

    assert not status.tracking_valid
    assert status.neutral_captured
    assert env.position.tolist() == pytest.approx(last_position.tolist())


def test_image_2d_controller_reset_returns_to_neutral_and_clears_calibration() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="image_2d",
        base_smoothing_alpha=1.0,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]

    assert controller.calibrate_image_2d(
        ImagePalmCenterTarget(
            palm_center=np.asarray([0.5, 0.5], dtype=np.float64),
            confidence=1.0,
            valid=True,
        )
    )
    controller.apply_image_2d(
        ImagePalmCenterTarget(
            palm_center=np.asarray([0.7, 0.2], dtype=np.float64),
            confidence=1.0,
            valid=True,
        )
    )
    status = controller.reset_to_neutral(clear_image_calibration=True)

    assert not status.neutral_captured
    assert env.position.tolist() == pytest.approx([0.0, 0.0, 0.14])


def test_default_orientation_control_is_disabled() -> None:
    config = HandBaseControlConfig(enabled=True)
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]
    neutral_quat = env.orientation_quat.copy()

    controller.calibrate_image_2d(
        ImagePalmCenterTarget(
            palm_center=np.asarray([0.5, 0.5], dtype=np.float64),
            confidence=1.0,
            valid=True,
        )
    )
    status = controller.apply_image_2d(
        ImagePalmCenterTarget(
            palm_center=np.asarray([0.6, 0.4], dtype=np.float64),
            confidence=1.0,
            valid=True,
        ),
        orientation_target=HandBaseTarget(
            position=np.zeros(3),
            orientation_quat=np.asarray([0.0, 1.0, 0.0, 0.0]),
            confidence=1.0,
            valid=True,
        ),
    )

    assert status.orientation_enabled is False
    assert env.orientation_quat.tolist() == pytest.approx(neutral_quat.tolist())


def test_mocap_controller_applies_clamped_base_pose() -> None:
    pytest.importorskip("mujoco")

    from dexvision.sim.mujoco_env import MujocoEnv

    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="pose_3d",
        base_fixed_z=0.0,
        position_scale=np.asarray([1.0, 1.0, 1.0]),
        rotation_mode="fixed",
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-0.05, -0.05, -0.05]),
            maximum=np.asarray([0.05, 0.05, 0.05]),
        ),
    )
    neutral = HandBaseTarget(
        position=np.asarray([0.0, 0.0, 0.0]),
        orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
        confidence=1.0,
        valid=True,
    )
    moved = HandBaseTarget(
        position=np.asarray([0.5, 0.0, 0.0]),
        orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
        confidence=1.0,
        valid=True,
    )

    with MujocoEnv(HAND_MODEL_PATH) as env:
        env.reset()
        controller = HandBaseMocapController(env, config)
        controller.apply(neutral)
        status = controller.apply(moved)
        mocap_position, mocap_quat = env.get_mocap_pose(config.mocap_body_name)

    assert status.tracking_valid
    assert status.clamped
    assert mocap_position[0] == pytest.approx(0.05)
    assert np.linalg.norm(mocap_quat) == pytest.approx(1.0)
    assert "base=tracking,clamped" in format_hand_base_status(status)


def test_mocap_controller_absolute_position_mode_uses_current_pose_directly() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="pose_3d",
        position_mode="absolute",
        position_scale=np.asarray([1.0, 1.0, 1.0]),
        rotation_mode="fixed",
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-2.0, -2.0, -2.0]),
            maximum=np.asarray([2.0, 2.0, 2.0]),
        ),
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]

    controller.apply(
        HandBaseTarget(
            position=np.asarray([0.1, 0.0, 0.0]),
            orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
            confidence=1.0,
            valid=True,
        )
    )
    status = controller.apply(
        HandBaseTarget(
            position=np.asarray([0.3, 0.0, 0.0]),
            orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
            confidence=1.0,
            valid=True,
        )
    )

    assert status.tracking_valid
    assert env.position.tolist() == pytest.approx([0.3, 0.0, 0.0])


def test_pose_3d_controller_holds_on_missing_tracking_without_jump() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="pose_3d",
        position_mode="absolute",
        position_scale=np.asarray([1.0, 1.0, 1.0]),
        rotation_mode="fixed",
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-2.0, -2.0, -2.0]),
            maximum=np.asarray([2.0, 2.0, 2.0]),
        ),
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]
    controller.apply(
        HandBaseTarget(
            position=np.asarray([0.3, 0.0, 0.0]),
            orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
            confidence=1.0,
            valid=True,
        )
    )
    last_position = env.position.copy()

    status = controller.apply(no_hand_base_target())

    assert not status.tracking_valid
    assert not status.rate_limited
    assert env.position.tolist() == pytest.approx(last_position.tolist())


def test_mocap_controller_relative_position_mode_uses_first_valid_pose_as_neutral() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="pose_3d",
        position_mode="relative",
        position_scale=np.asarray([1.0, 1.0, 1.0]),
        rotation_mode="fixed",
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-2.0, -2.0, -2.0]),
            maximum=np.asarray([2.0, 2.0, 2.0]),
        ),
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]

    first_status = controller.apply(
        HandBaseTarget(
            position=np.asarray([0.1, 0.0, 0.0]),
            orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
            confidence=1.0,
            valid=True,
        )
    )
    second_status = controller.apply(
        HandBaseTarget(
            position=np.asarray([0.3, 0.0, 0.0]),
            orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
            confidence=1.0,
            valid=True,
        )
    )

    assert first_status.applied_target.position.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert second_status.tracking_valid
    assert env.position.tolist() == pytest.approx([0.2, 0.0, 0.0])


def test_mocap_controller_rate_limits_large_pose_jumps() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="pose_3d",
        base_fixed_z=0.0,
        position_scale=np.asarray([1.0, 1.0, 1.0]),
        rotation_mode="palm_delta",
        enable_base_orientation=True,
        max_position_step=0.01,
        max_rotation_step_degrees=5.0,
        workspace_limits=WorkspaceLimits(
            minimum=np.asarray([-2.0, -2.0, -2.0]),
            maximum=np.asarray([2.0, 2.0, 2.0]),
        ),
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]
    neutral = HandBaseTarget(
        position=np.zeros(3),
        orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0]),
        confidence=1.0,
        valid=True,
    )
    jumped = HandBaseTarget(
        position=np.asarray([1.0, 0.0, 0.0]),
        orientation_quat=np.asarray([0.0, 1.0, 0.0, 0.0]),
        confidence=1.0,
        valid=True,
    )

    controller.apply(neutral)
    previous_orientation = env.orientation_quat.copy()
    status = controller.apply(jumped)

    assert status.rate_limited
    assert np.linalg.norm(status.applied_target.position) == pytest.approx(0.01)
    assert quaternion_angle(previous_orientation, status.applied_target.orientation_quat) <= (
        np.deg2rad(5.0) + 1e-6
    )
    assert ",limited" in format_hand_base_status(status)


def test_absolute_palm_rotation_maps_synthetic_upright_pose_to_fingers_up() -> None:
    config = replace(
        HandBaseControlConfig(enabled=True),
        base_control_mode="pose_3d",
        position_scale=np.zeros(3),
        rotation_mode="palm_absolute",
        enable_base_orientation=True,
        max_position_step=1.0,
        max_rotation_step_degrees=180.0,
    )
    env = FakeMocapEnv()
    controller = HandBaseMocapController(env, config)  # type: ignore[arg-type]
    target = estimate_hand_base_target(_open_hand_landmarks(), confidence=1.0)

    status = controller.apply(target)

    assert status.tracking_valid
    assert not status.rate_limited
    assert np.allclose(
        np.abs(status.applied_target.orientation_quat),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=1e-5,
    )


def test_hand_scene_contains_mocap_body_and_weld() -> None:
    text = HAND_MODEL_PATH.read_text(encoding="utf-8")
    shadow_hand_text = (
        ROOT / "assets" / "mujoco" / "menagerie" / "shadow_hand" / "right_hand_dexvision.xml"
    ).read_text(encoding="utf-8")

    assert 'body name="dexvision_hand_base_target" mocap="true"' in text
    assert (
        'body name="dexvision_hand_base_target" mocap="true" pos="0 0 0.14" '
        'quat="1 0 0 0"'
    ) in text
    assert 'body name="rh_forearm" childclass="right_hand" pos="0 0 0.14" quat="1 0 0 0"' in (
        shadow_hand_text
    )
    assert 'weld name="dexvision_hand_base_weld"' in text


def test_hand_scene_neutral_pose_starts_upright_with_palm_forward() -> None:
    pytest.importorskip("mujoco")

    from dexvision.sim.mujoco_env import MujocoEnv

    with MujocoEnv(HAND_MODEL_PATH) as env:
        env.reset()
        _, mocap_quat = env.get_mocap_pose("dexvision_hand_base_target")
        palm = _body_position(env, "rh_palm")
        middle_tip = _body_position(env, "rh_mfdistal")
        index_tip = _body_position(env, "rh_ffdistal")
        pinky_tip = _body_position(env, "rh_lfdistal")
        forearm_quat = _body_quaternion(env, "rh_forearm")

    finger_direction = middle_tip - palm
    spread_direction = index_tip - pinky_tip
    palm_normal = np.cross(spread_direction, finger_direction)
    palm_normal /= np.linalg.norm(palm_normal)

    assert mocap_quat.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert forearm_quat.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert finger_direction[2] > 0.12
    assert abs(finger_direction[2]) > (5.0 * max(abs(finger_direction[0]), abs(finger_direction[1])))
    assert palm_normal[0] > 0.95


def test_check_hand_base_control_help_runs_without_real_webcam() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.check_hand_base_control", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "--enable-base-control" in result.stdout
    assert "--base-control-mode" in result.stdout
    assert "--enable-base-orientation" in result.stdout
    assert "--enable-depth-control" in result.stdout
    assert "--disable-depth-control" in result.stdout
    assert "--show-camera-window" in result.stdout
