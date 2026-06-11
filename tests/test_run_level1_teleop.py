from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import run_level1_teleop
from dexvision.camera.opencv_camera import CameraFrame
from dexvision.features.hand_features import FingerState, HandFeatures, no_hand_features
from dexvision.features.smoothing import FeatureSmoother
from dexvision.perception.hand_tracker import HandTracker, HandTrackerError
from dexvision.retargeting.curl_retargeter import (
    CurlRetargeter,
    load_curl_retargeter_config,
)
from dexvision.sim.mujoco_env import MujocoState


ROOT = Path(__file__).resolve().parents[1]
TELEOP_CONFIG_PATH = ROOT / "configs" / "level1_teleop.yaml"
EXPECTED_TARGETS = (
    "rh_A_WRJ2",
    "rh_A_WRJ1",
    "rh_A_THJ5",
    "rh_A_THJ4",
    "rh_A_THJ3",
    "rh_A_THJ2",
    "rh_A_THJ1",
    "rh_A_FFJ4",
    "rh_A_FFJ3",
    "rh_A_FFJ0",
    "rh_A_MFJ4",
    "rh_A_MFJ3",
    "rh_A_MFJ0",
    "rh_A_RFJ4",
    "rh_A_RFJ3",
    "rh_A_RFJ0",
    "rh_A_LFJ5",
    "rh_A_LFJ4",
    "rh_A_LFJ3",
    "rh_A_LFJ0",
)


def _state(curl: float, extension: float) -> FingerState:
    return FingerState(
        curl=curl,
        extension=extension,
        abduction=None,
        is_up=extension >= 0.55,
        valid=True,
    )


def _full_hand_features(
    *,
    thumb: float = 0.0,
    index_extension: float = 1.0,
    middle_extension: float = 1.0,
    ring_extension: float = 1.0,
    pinky_extension: float = 1.0,
    confidence: float = 1.0,
) -> HandFeatures:
    return HandFeatures(
        thumb=_state(thumb, 1.0 - thumb),
        index=_state(0.9, index_extension),
        middle=_state(0.9, middle_extension),
        ring=_state(0.9, ring_extension),
        pinky=_state(0.9, pinky_extension),
        pinch_thumb_index=0.0,
        palm_roll_proxy=0.0,
        palm_pitch_proxy=0.0,
        confidence=confidence,
    )


def test_parser_exposes_level1_teleop_paths_and_camera_options() -> None:
    parser = run_level1_teleop.build_parser()

    args = parser.parse_args(
        [
            "--camera-id",
            "2",
            "--config",
            "custom.yaml",
            "--model",
            "hand.xml",
            "--show-camera-window",
            "--enable-base-control",
            "--base-control-mode",
            "image_2d",
            "--enable-base-orientation",
            "--orientation-dofs",
            "roll",
            "--max-roll-deg",
            "135",
            "--max-pitch-deg",
            "100",
            "--max-yaw-deg",
            "140",
            "--orientation-smoothing-alpha",
            "0.6",
            "--orientation-deadband-deg",
            "0.25",
            "--max-rotation-step-deg",
            "12",
            "--enable-depth-control",
            "--camera-window-name",
            "Demo Window",
        ]
    )

    assert args.camera_id == 2
    assert args.config == Path("custom.yaml")
    assert args.model == Path("hand.xml")
    assert args.width == run_level1_teleop.DEFAULT_CAMERA_WIDTH
    assert args.height == run_level1_teleop.DEFAULT_CAMERA_HEIGHT
    assert args.show_camera_window is True
    assert args.enable_base_control is True
    assert args.base_control_mode == "image_2d"
    assert args.enable_base_orientation is True
    assert args.orientation_dofs == "roll"
    assert args.max_roll_deg == pytest.approx(135.0)
    assert args.max_pitch_deg == pytest.approx(100.0)
    assert args.max_yaw_deg == pytest.approx(140.0)
    assert args.orientation_smoothing_alpha == pytest.approx(0.6)
    assert args.orientation_deadband_deg == pytest.approx(0.25)
    assert args.max_rotation_step_deg == pytest.approx(12.0)
    assert args.enable_depth_control is True
    assert args.camera_window_name == "Demo Window"


def test_default_target_names_cover_full_shadow_hand_config() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)

    assert run_level1_teleop.robot_target_names(retargeter) == EXPECTED_TARGETS
    assert run_level1_teleop.configured_control_fields(retargeter) == (
        "thumb:thumb_curl",
        "index:index_bend",
        "middle:middle_bend",
        "ring:ring_bend",
        "pinky:pinky_bend",
    )


def test_resolve_mujoco_model_path_defaults_to_teleop_config_model_path() -> None:
    raw_config = load_curl_retargeter_config(TELEOP_CONFIG_PATH)

    model_path = run_level1_teleop.resolve_mujoco_model_path(
        raw_config,
        config_path=TELEOP_CONFIG_PATH,
        override=None,
    )

    assert model_path == Path("assets/mujoco/hand_scene.xml")


def test_full_hand_targets_drive_all_configured_fingers() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    neutral = run_level1_teleop.build_full_hand_targets(retargeter, no_hand_features())
    fist = run_level1_teleop.build_full_hand_targets(
        retargeter,
        _full_hand_features(
            thumb=1.0,
            index_extension=0.0,
            middle_extension=0.0,
            ring_extension=0.0,
            pinky_extension=0.0,
        ),
    )

    assert fist["rh_A_THJ1"] > neutral["rh_A_THJ1"]
    assert fist["rh_A_FFJ0"] > neutral["rh_A_FFJ0"]
    assert fist["rh_A_MFJ0"] > neutral["rh_A_MFJ0"]
    assert fist["rh_A_RFJ0"] > neutral["rh_A_RFJ0"]
    assert fist["rh_A_LFJ0"] > neutral["rh_A_LFJ0"]


def test_point_and_peace_shape_use_long_finger_bend_fields() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    point = run_level1_teleop.build_full_hand_targets(
        retargeter,
        _full_hand_features(
            index_extension=1.0,
            middle_extension=0.0,
            ring_extension=0.0,
            pinky_extension=0.0,
        ),
    )
    peace = run_level1_teleop.build_full_hand_targets(
        retargeter,
        _full_hand_features(
            index_extension=1.0,
            middle_extension=1.0,
            ring_extension=0.0,
            pinky_extension=0.0,
        ),
    )

    assert point["rh_A_FFJ0"] < point["rh_A_MFJ0"]
    assert point["rh_A_FFJ0"] < point["rh_A_RFJ0"]
    assert peace["rh_A_FFJ0"] == pytest.approx(point["rh_A_FFJ0"])
    assert peace["rh_A_MFJ0"] < peace["rh_A_RFJ0"]
    assert peace["rh_A_MFJ0"] < peace["rh_A_LFJ0"]


def test_run_loop_maps_full_hand_features_without_real_camera_or_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OneFrameCamera:
        def read(self) -> CameraFrame:
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            return CameraFrame(success=True, frame=frame, timestamp=1.0)

    class SyntheticTracker:
        def process(self, frame: np.ndarray, *, timestamp: float):
            del frame
            return HandTracker.no_hand_result(timestamp)

    class FakeEnv:
        def __init__(self) -> None:
            self.targets: dict[str, float] | None = None

        def set_joint_targets(self, targets: dict[str, float]) -> None:
            self.targets = targets

        def step(self, *, n_steps: int) -> MujocoState:
            assert n_steps == 1
            return MujocoState(
                time=0.002,
                qpos=np.zeros(1, dtype=np.float64),
                qvel=np.zeros(1, dtype=np.float64),
                ctrl=np.zeros(1, dtype=np.float64),
            )

    class ClosingViewer:
        def __init__(self) -> None:
            self.sync_count = 0

        def sync(self) -> None:
            self.sync_count += 1

        def is_running(self) -> bool:
            return self.sync_count == 0

    synthetic_features = _full_hand_features(
        thumb=1.0,
        index_extension=0.0,
        middle_extension=0.0,
        ring_extension=0.0,
        pinky_extension=0.0,
    )
    monkeypatch.setattr(
        run_level1_teleop,
        "extract_hand_features",
        lambda _result: synthetic_features,
    )
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    viewer = ClosingViewer()
    env = FakeEnv()

    run_level1_teleop._run_loop(
        env=env,
        camera=OneFrameCamera(),
        tracker=SyntheticTracker(),
        smoother=FeatureSmoother(alpha=1.0),
        retargeter=retargeter,
        target_names=run_level1_teleop.robot_target_names(retargeter),
        camera_overlay=None,
        sim_steps_per_frame=1,
        viewer_handle=viewer,
        viewer_sleep=0.0,
        print_interval=1,
    )

    assert viewer.sync_count == 1
    assert env.targets is not None
    assert env.targets["rh_A_THJ1"] == pytest.approx(1.05)
    assert env.targets["rh_A_FFJ0"] == pytest.approx(2.25)
    assert env.targets["rh_A_MFJ0"] == pytest.approx(2.25)
    assert env.targets["rh_A_RFJ0"] == pytest.approx(2.25)
    assert env.targets["rh_A_LFJ0"] == pytest.approx(2.25)


def test_instability_guard_keeps_strict_finger_only_qvel_limit() -> None:
    state = MujocoState(
        time=0.0,
        qpos=np.asarray([0.5], dtype=np.float64),
        qvel=np.asarray([run_level1_teleop.MAX_ABS_QVEL + 1.0], dtype=np.float64),
        ctrl=np.zeros(1, dtype=np.float64),
    )

    with pytest.raises(run_level1_teleop.MujocoError, match="max_abs_qvel"):
        run_level1_teleop._raise_if_unstable(state)


def test_instability_guard_allows_bounded_base_control_qvel_transients() -> None:
    state = MujocoState(
        time=0.0,
        qpos=np.asarray([0.5], dtype=np.float64),
        qvel=np.asarray([run_level1_teleop.MAX_ABS_QVEL + 30.0], dtype=np.float64),
        ctrl=np.zeros(1, dtype=np.float64),
    )

    run_level1_teleop._raise_if_unstable(
        state,
        max_abs_qvel=run_level1_teleop.MAX_ABS_BASE_CONTROL_QVEL,
    )


def test_instability_guard_still_rejects_extreme_base_control_qvel() -> None:
    state = MujocoState(
        time=0.0,
        qpos=np.asarray([0.5], dtype=np.float64),
        qvel=np.asarray(
            [run_level1_teleop.MAX_ABS_BASE_CONTROL_QVEL + 1.0],
            dtype=np.float64,
        ),
        ctrl=np.zeros(1, dtype=np.float64),
    )

    with pytest.raises(run_level1_teleop.MujocoError, match="max_abs_qvel"):
        run_level1_teleop._raise_if_unstable(
            state,
            max_abs_qvel=run_level1_teleop.MAX_ABS_BASE_CONTROL_QVEL,
        )


def test_run_loop_sends_presentable_demo_overlay_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OneFrameCamera:
        def read(self) -> CameraFrame:
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            return CameraFrame(success=True, frame=frame, timestamp=1.0)

    class SyntheticTracker:
        def process(self, frame: np.ndarray, *, timestamp: float):
            del frame
            return HandTracker.no_hand_result(timestamp)

    class FakeEnv:
        def __init__(self) -> None:
            self.targets: dict[str, float] | None = None

        def set_joint_targets(self, targets: dict[str, float]) -> None:
            self.targets = targets

        def step(self, *, n_steps: int) -> MujocoState:
            assert n_steps == 1
            return MujocoState(
                time=0.002,
                qpos=np.zeros(1, dtype=np.float64),
                qvel=np.zeros(1, dtype=np.float64),
                ctrl=np.zeros(1, dtype=np.float64),
            )

    class ClosingViewer:
        def __init__(self) -> None:
            self.sync_count = 0

        def sync(self) -> None:
            self.sync_count += 1

        def is_running(self) -> bool:
            return self.sync_count == 0

    class RecordingOverlay:
        def __init__(self) -> None:
            self.payloads: list[run_level1_teleop.CameraOverlayFrame] = []

        def send(self, payload: run_level1_teleop.CameraOverlayFrame) -> None:
            self.payloads.append(payload)

        def should_stop(self) -> bool:
            return False

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        run_level1_teleop,
        "extract_hand_features",
        lambda _result: _full_hand_features(confidence=0.0),
    )
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    overlay = RecordingOverlay()

    run_level1_teleop._run_loop(
        env=FakeEnv(),
        camera=OneFrameCamera(),
        tracker=SyntheticTracker(),
        smoother=FeatureSmoother(
            min_confidence=run_level1_teleop.DEFAULT_MIN_SMOOTHING_CONFIDENCE,
            low_confidence_behavior=run_level1_teleop.DEFAULT_LOW_CONFIDENCE_BEHAVIOR,
        ),
        retargeter=retargeter,
        target_names=run_level1_teleop.robot_target_names(retargeter),
        camera_overlay=overlay,
        sim_steps_per_frame=1,
        viewer_handle=ClosingViewer(),
        viewer_sleep=0.0,
        print_interval=1,
    )

    assert len(overlay.payloads) == 1
    payload = overlay.payloads[0]
    assert payload.frame.shape == (16, 16, 3)
    assert payload.status_message == "TRACKING LOST - controls decaying to open"
    assert payload.target_names == EXPECTED_TARGETS


def test_tracking_loss_default_decays_full_hand_controls_toward_neutral() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    smoother = FeatureSmoother(
        alpha=1.0,
        min_confidence=run_level1_teleop.DEFAULT_MIN_SMOOTHING_CONFIDENCE,
        low_confidence_behavior=run_level1_teleop.DEFAULT_LOW_CONFIDENCE_BEHAVIOR,
        decay_alpha=run_level1_teleop.DEFAULT_DECAY_ALPHA,
    )
    stable = smoother.update(
        _full_hand_features(
            thumb=1.0,
            index_extension=0.0,
            middle_extension=0.0,
            ring_extension=0.0,
            pinky_extension=0.0,
        )
    )
    lost = smoother.update(no_hand_features())

    stable_targets = run_level1_teleop.build_full_hand_targets(retargeter, stable)
    lost_targets = run_level1_teleop.build_full_hand_targets(retargeter, lost)
    neutral_targets = run_level1_teleop.build_full_hand_targets(retargeter, no_hand_features())

    assert 0.0 < lost.index_bend < stable.index_bend
    assert 0.0 < lost.middle_bend < stable.middle_bend
    assert 0.0 < lost.ring_bend < stable.ring_bend
    assert 0.0 < lost.pinky_bend < stable.pinky_bend
    assert neutral_targets["rh_A_FFJ0"] < lost_targets["rh_A_FFJ0"] < stable_targets["rh_A_FFJ0"]
    assert neutral_targets["rh_A_MFJ0"] < lost_targets["rh_A_MFJ0"] < stable_targets["rh_A_MFJ0"]
    assert neutral_targets["rh_A_RFJ0"] < lost_targets["rh_A_RFJ0"] < stable_targets["rh_A_RFJ0"]
    assert neutral_targets["rh_A_LFJ0"] < lost_targets["rh_A_LFJ0"] < stable_targets["rh_A_LFJ0"]


def test_poll_overlay_commands_returns_camera_key_commands() -> None:
    class CommandOverlay:
        def poll_commands(self) -> tuple[str, ...]:
            return ("calibrate_base", "reset_base")

    assert run_level1_teleop._poll_overlay_commands(CommandOverlay()) == (
        "calibrate_base",
        "reset_base",
    )
    assert run_level1_teleop._poll_overlay_commands(None) == ()


def test_mjpython_command_preserves_orientation_tuning_overrides() -> None:
    command = run_level1_teleop._format_mjpython_command(
        camera_id=2,
        width=run_level1_teleop.DEFAULT_CAMERA_WIDTH,
        height=run_level1_teleop.DEFAULT_CAMERA_HEIGHT,
        config_path=run_level1_teleop.DEFAULT_CONFIG,
        model_path=None,
        hand_landmarker_model_path=None,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        assume_mirrored_input=False,
        smoothing_alpha=run_level1_teleop.DEFAULT_SMOOTHING_ALPHA,
        min_smoothing_confidence=run_level1_teleop.DEFAULT_MIN_SMOOTHING_CONFIDENCE,
        low_confidence_behavior=run_level1_teleop.DEFAULT_LOW_CONFIDENCE_BEHAVIOR,
        decay_alpha=run_level1_teleop.DEFAULT_DECAY_ALPHA,
        sim_steps_per_frame=run_level1_teleop.DEFAULT_SIM_STEPS_PER_FRAME,
        viewer_sleep=run_level1_teleop.DEFAULT_VIEWER_SLEEP,
        print_interval=run_level1_teleop.DEFAULT_PRINT_INTERVAL,
        show_camera_window=True,
        enable_base_control=True,
        base_control_mode="image_2d",
        enable_base_orientation=True,
        orientation_dofs="roll,pitch,yaw",
        max_roll_deg=135.0,
        max_pitch_deg=100.0,
        max_yaw_deg=140.0,
        orientation_smoothing_alpha=0.6,
        orientation_deadband_deg=0.25,
        max_rotation_step_degrees=12.0,
        enable_depth_control=True,
        camera_window_name=run_level1_teleop.DEFAULT_CAMERA_WINDOW_NAME,
    )

    assert "--enable-base-orientation" in command
    assert "--orientation-dofs" not in command
    assert "--max-roll-deg 135.0" in command
    assert "--max-pitch-deg 100.0" in command
    assert "--max-yaw-deg 140.0" in command
    assert "--orientation-smoothing-alpha 0.6" in command
    assert "--orientation-deadband-deg 0.25" in command
    assert "--max-rotation-step-deg 12.0" in command


def test_status_formatters_keep_console_output_compact() -> None:
    features = _full_hand_features(
        thumb=0.2,
        index_extension=0.9,
        middle_extension=0.8,
        ring_extension=0.7,
        pinky_extension=0.6,
    )
    targets = {name: float(index) for index, name in enumerate(EXPECTED_TARGETS)}

    assert run_level1_teleop._format_control_summary(features) == (
        "controls=Tcurl:0.20,Ibend:0.10,Mbend:0.20,Rbend:0.30,Pbend:0.40"
    )
    assert run_level1_teleop._format_target_summary(
        targets,
        EXPECTED_TARGETS,
        max_items=3,
    ) == "rh_A_WRJ2=0.00, rh_A_WRJ1=1.00, rh_A_THJ5=2.00, ...(17 more)"
    assert (
        run_level1_teleop._format_tracking_status(
            detected=False,
            confidence=0.0,
            min_confidence=0.2,
            low_confidence_behavior="decay",
        )
        == "TRACKING LOST - controls decaying to open"
    )
    assert (
        run_level1_teleop._format_tracking_status(
            detected=True,
            confidence=0.1,
            min_confidence=0.2,
            low_confidence_behavior="hold",
        )
        == "LOW CONFIDENCE 0.10 - controls holding last pose"
    )


def test_run_level1_teleop_help_runs_without_real_webcam() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.run_level1_teleop", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "--camera-id" in result.stdout
    assert "--config" in result.stdout
    assert "--model" in result.stdout
    assert "--hand-landmarker-model" in result.stdout
    assert "--show-camera-window" in result.stdout
    assert "--enable-base-control" in result.stdout
    assert "--base-control-mode" in result.stdout
    assert "--enable-base-orientation" in result.stdout
    assert "--orientation-dofs" in result.stdout
    assert "--max-roll-deg" in result.stdout
    assert "--max-pitch-deg" in result.stdout
    assert "--max-yaw-deg" in result.stdout
    assert "--orientation-smoothing-alpha" in result.stdout
    assert "--orientation-deadband-deg" in result.stdout
    assert "--max-rotation-step-deg" in result.stdout
    assert "--enable-depth-control" in result.stdout
    assert "--disable-depth-control" in result.stdout
    assert "--camera-window-name" in result.stdout


def test_main_reports_startup_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_to_start(**_kwargs: object) -> int:
        raise HandTrackerError("MediaPipe is required for hand tracking.")

    monkeypatch.setattr(run_level1_teleop, "run_level1_teleop", fail_to_start)

    result = run_level1_teleop.main(["--camera-id", "0"])
    captured = capsys.readouterr()

    assert result == 2
    assert "ERROR: MediaPipe is required for hand tracking." in captured.err
