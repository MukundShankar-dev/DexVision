from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import check_one_finger_teleop
from dexvision.camera.opencv_camera import CameraFrame
from dexvision.features.hand_features import HandFeatures, no_hand_features
from dexvision.features.smoothing import FeatureSmoother
from dexvision.perception.hand_tracker import HandTracker, HandTrackerError
from dexvision.retargeting.curl_retargeter import (
    CurlRetargeter,
    load_curl_retargeter_config,
)
from dexvision.sim.mujoco_env import MujocoState


ROOT = Path(__file__).resolve().parents[1]
TELEOP_CONFIG_PATH = ROOT / "configs" / "level1_teleop.yaml"
INDEX_TARGETS = ("rh_A_FFJ4", "rh_A_FFJ3", "rh_A_FFJ0")


def _features(
    *,
    thumb: float = 0.0,
    index: float = 0.0,
    middle: float = 0.0,
    ring: float = 0.0,
    pinky: float = 0.0,
    confidence: float = 1.0,
) -> HandFeatures:
    return HandFeatures(
        thumb_curl=thumb,
        index_curl=index,
        middle_curl=middle,
        ring_curl=ring,
        pinky_curl=pinky,
        pinch_thumb_index=0.75,
        palm_roll_proxy=0.25,
        palm_pitch_proxy=-0.25,
        confidence=confidence,
    )


def test_index_only_features_preserves_only_index_curl_and_confidence() -> None:
    source = _features(
        thumb=1.0,
        index=0.6,
        middle=1.0,
        ring=1.0,
        pinky=1.0,
        confidence=0.8,
    )

    one_finger = check_one_finger_teleop.index_only_features(source)

    assert one_finger.thumb_curl == pytest.approx(0.0)
    assert one_finger.index_curl == pytest.approx(0.6)
    assert one_finger.middle_curl == pytest.approx(0.0)
    assert one_finger.ring_curl == pytest.approx(0.0)
    assert one_finger.pinky_curl == pytest.approx(0.0)
    assert one_finger.pinch_thumb_index == pytest.approx(0.0)
    assert one_finger.palm_roll_proxy == pytest.approx(0.0)
    assert one_finger.palm_pitch_proxy == pytest.approx(0.0)
    assert one_finger.confidence == pytest.approx(0.8)


def test_measured_index_normalization_softens_open_pose_and_preserves_curl() -> None:
    open_effective = check_one_finger_teleop.normalize_index_curl(
        0.234,
        open_curl=check_one_finger_teleop.DEFAULT_INDEX_OPEN_CURL,
        closed_curl=check_one_finger_teleop.DEFAULT_INDEX_CLOSED_CURL,
        gamma=check_one_finger_teleop.DEFAULT_INDEX_RESPONSE_GAMMA,
    )
    curled_effective = check_one_finger_teleop.normalize_index_curl(
        0.801,
        open_curl=check_one_finger_teleop.DEFAULT_INDEX_OPEN_CURL,
        closed_curl=check_one_finger_teleop.DEFAULT_INDEX_CLOSED_CURL,
        gamma=check_one_finger_teleop.DEFAULT_INDEX_RESPONSE_GAMMA,
    )

    assert 0.0 < open_effective < 0.05
    assert curled_effective == pytest.approx(1.0)


def test_index_only_targets_move_index_and_keep_other_fingers_neutral() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    neutral_targets = check_one_finger_teleop.build_index_finger_targets(
        retargeter,
        _features(index=0.0, thumb=1.0, middle=1.0, ring=1.0, pinky=1.0),
    )
    curled_targets = check_one_finger_teleop.build_index_finger_targets(
        retargeter,
        _features(index=1.0, thumb=1.0, middle=1.0, ring=1.0, pinky=1.0),
    )

    assert check_one_finger_teleop.index_target_names(retargeter) == INDEX_TARGETS
    assert curled_targets["rh_A_FFJ3"] > neutral_targets["rh_A_FFJ3"]
    assert curled_targets["rh_A_FFJ0"] > neutral_targets["rh_A_FFJ0"]

    for target_name, curled_target in curled_targets.items():
        if target_name not in INDEX_TARGETS:
            assert curled_target == pytest.approx(neutral_targets[target_name])


def test_measured_index_defaults_make_open_pose_targets_near_neutral() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    neutral_targets = check_one_finger_teleop.build_index_finger_targets(
        retargeter,
        no_hand_features(),
        index_open_curl=check_one_finger_teleop.DEFAULT_INDEX_OPEN_CURL,
        index_closed_curl=check_one_finger_teleop.DEFAULT_INDEX_CLOSED_CURL,
        index_response_gamma=check_one_finger_teleop.DEFAULT_INDEX_RESPONSE_GAMMA,
    )
    open_pose_targets = check_one_finger_teleop.build_index_finger_targets(
        retargeter,
        _features(index=0.234),
        index_open_curl=check_one_finger_teleop.DEFAULT_INDEX_OPEN_CURL,
        index_closed_curl=check_one_finger_teleop.DEFAULT_INDEX_CLOSED_CURL,
        index_response_gamma=check_one_finger_teleop.DEFAULT_INDEX_RESPONSE_GAMMA,
    )

    assert open_pose_targets["rh_A_FFJ3"] == pytest.approx(
        neutral_targets["rh_A_FFJ3"],
        abs=0.05,
    )
    assert open_pose_targets["rh_A_FFJ0"] == pytest.approx(
        neutral_targets["rh_A_FFJ0"],
        abs=0.08,
    )


def test_finger_curl_summary_includes_all_fingers() -> None:
    summary = check_one_finger_teleop._format_finger_curl_summary(
        "raw",
        _features(thumb=0.1, index=0.2, middle=0.3, ring=0.4, pinky=0.5),
    )

    assert summary == "raw_curls=T:0.10,I:0.20,M:0.30,R:0.40,P:0.50"


def test_normalized_landmarks_to_pixels_clips_coordinates() -> None:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[5] = [-1.0, -1.0, 0.0]
    landmarks[8] = [2.0, 2.0, 0.0]

    points = check_one_finger_teleop._normalized_landmarks_to_pixels(
        landmarks,
        width=11,
        height=21,
    )

    assert points[5] == (0, 0)
    assert points[8] == (10, 20)


def test_draw_index_landmarks_draws_three_segments_without_crashing() -> None:
    class FakeCv2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16

        def __init__(self) -> None:
            self.lines: list[tuple[tuple[int, int], tuple[int, int]]] = []
            self.circles: list[tuple[int, int]] = []
            self.labels: list[str] = []

        def line(
            self,
            _frame: np.ndarray,
            start: tuple[int, int],
            end: tuple[int, int],
            *_args: object,
        ) -> None:
            self.lines.append((start, end))

        def circle(
            self,
            _frame: np.ndarray,
            point: tuple[int, int],
            *_args: object,
        ) -> None:
            self.circles.append(point)

        def putText(
            self,
            _frame: np.ndarray,
            text: str,
            *_args: object,
        ) -> None:
            self.labels.append(text)

    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[5] = [0.10, 0.20, 0.0]
    landmarks[6] = [0.20, 0.30, 0.0]
    landmarks[7] = [0.30, 0.40, 0.0]
    landmarks[8] = [0.40, 0.50, 0.0]
    tracking_result = check_one_finger_teleop.HandTrackingResult(
        detected=True,
        handedness="Left",
        confidence=1.0,
        image_landmarks=landmarks,
        world_landmarks=None,
        timestamp=1.0,
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    fake_cv2 = FakeCv2()

    check_one_finger_teleop._draw_index_landmarks(
        fake_cv2,
        frame,
        tracking_result,
    )

    assert len(fake_cv2.lines) == 3
    assert len(fake_cv2.circles) == 4
    assert fake_cv2.labels == ["INDEX"]


def test_tracking_loss_default_decays_index_control_toward_neutral() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    smoother = FeatureSmoother(
        alpha=1.0,
        min_confidence=check_one_finger_teleop.DEFAULT_MIN_SMOOTHING_CONFIDENCE,
        low_confidence_behavior=check_one_finger_teleop.DEFAULT_LOW_CONFIDENCE_BEHAVIOR,
        decay_alpha=check_one_finger_teleop.DEFAULT_DECAY_ALPHA,
    )
    stable = smoother.update(_features(index=1.0, confidence=1.0))
    lost = smoother.update(no_hand_features())

    stable_targets = check_one_finger_teleop.build_index_finger_targets(
        retargeter,
        stable,
    )
    lost_targets = check_one_finger_teleop.build_index_finger_targets(retargeter, lost)
    neutral_targets = check_one_finger_teleop.build_index_finger_targets(
        retargeter,
        no_hand_features(),
    )

    assert 0.0 < lost.index_curl < stable.index_curl
    assert (
        neutral_targets["rh_A_FFJ0"]
        < lost_targets["rh_A_FFJ0"]
        < stable_targets["rh_A_FFJ0"]
    )


def test_resolve_mujoco_model_path_defaults_to_teleop_config_model_path() -> None:
    raw_config = load_curl_retargeter_config(TELEOP_CONFIG_PATH)

    model_path = check_one_finger_teleop.resolve_mujoco_model_path(
        raw_config,
        config_path=TELEOP_CONFIG_PATH,
        override=None,
    )

    assert model_path == Path("assets/mujoco/hand_scene.xml")


def test_check_one_finger_teleop_help_runs_without_real_webcam() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.check_one_finger_teleop", "--help"],
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
    assert "--index-open-curl" in result.stdout
    assert "--index-closed-curl" in result.stdout
    assert "--index-response-gamma" in result.stdout


def test_camera_overlay_window_defaults_off_and_can_be_enabled() -> None:
    parser = check_one_finger_teleop.build_parser()

    default_args = parser.parse_args([])
    overlay_args = parser.parse_args(["--show-camera-window"])

    assert default_args.show_camera_window is False
    assert default_args.smoothing_alpha == pytest.approx(
        check_one_finger_teleop.DEFAULT_SMOOTHING_ALPHA
    )
    assert default_args.viewer_sleep == pytest.approx(
        check_one_finger_teleop.DEFAULT_VIEWER_SLEEP
    )
    assert default_args.index_open_curl == pytest.approx(
        check_one_finger_teleop.DEFAULT_INDEX_OPEN_CURL
    )
    assert default_args.index_closed_curl == pytest.approx(
        check_one_finger_teleop.DEFAULT_INDEX_CLOSED_CURL
    )
    assert default_args.index_response_gamma == pytest.approx(
        check_one_finger_teleop.DEFAULT_INDEX_RESPONSE_GAMMA
    )
    assert overlay_args.show_camera_window is True


def test_viewer_closed_detection_uses_optional_viewer_api() -> None:
    class ViewerWithoutRunningStatus:
        pass

    class RunningViewer:
        def is_running(self) -> bool:
            return True

    class ClosedViewer:
        def is_running(self) -> bool:
            return False

    assert (
        check_one_finger_teleop._viewer_was_closed(ViewerWithoutRunningStatus())
        is False
    )
    assert check_one_finger_teleop._viewer_was_closed(RunningViewer()) is False
    assert check_one_finger_teleop._viewer_was_closed(ClosedViewer()) is True


def test_run_loop_default_path_does_not_require_cv2_display_module() -> None:
    class OneFrameCamera:
        def read(self) -> CameraFrame:
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            return CameraFrame(success=True, frame=frame, timestamp=1.0)

    class NoHandTracker:
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

    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    viewer = ClosingViewer()
    env = FakeEnv()

    check_one_finger_teleop._run_loop(
        env=env,
        camera=OneFrameCamera(),
        tracker=NoHandTracker(),
        smoother=FeatureSmoother(),
        retargeter=retargeter,
        index_targets=check_one_finger_teleop.index_target_names(retargeter),
        camera_overlay=None,
        window_name="unused",
        index_open_curl=check_one_finger_teleop.DEFAULT_INDEX_OPEN_CURL,
        index_closed_curl=check_one_finger_teleop.DEFAULT_INDEX_CLOSED_CURL,
        index_response_gamma=check_one_finger_teleop.DEFAULT_INDEX_RESPONSE_GAMMA,
        sim_steps_per_frame=1,
        viewer_handle=viewer,
        viewer_sleep=0.0,
        print_interval=1,
    )

    assert viewer.sync_count == 1
    assert env.targets is not None


def test_run_loop_sends_frames_to_camera_overlay() -> None:
    class OneFrameCamera:
        def read(self) -> CameraFrame:
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            return CameraFrame(success=True, frame=frame, timestamp=1.0)

    class NoHandTracker:
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
            self.payloads: list[check_one_finger_teleop.CameraOverlayFrame] = []

        def send(self, payload: check_one_finger_teleop.CameraOverlayFrame) -> None:
            self.payloads.append(payload)

        def should_stop(self) -> bool:
            return False

        def close(self) -> None:
            pass

    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    viewer = ClosingViewer()
    env = FakeEnv()
    overlay = RecordingOverlay()

    check_one_finger_teleop._run_loop(
        env=env,
        camera=OneFrameCamera(),
        tracker=NoHandTracker(),
        smoother=FeatureSmoother(),
        retargeter=retargeter,
        index_targets=check_one_finger_teleop.index_target_names(retargeter),
        camera_overlay=overlay,
        window_name="unused",
        index_open_curl=check_one_finger_teleop.DEFAULT_INDEX_OPEN_CURL,
        index_closed_curl=check_one_finger_teleop.DEFAULT_INDEX_CLOSED_CURL,
        index_response_gamma=check_one_finger_teleop.DEFAULT_INDEX_RESPONSE_GAMMA,
        sim_steps_per_frame=1,
        viewer_handle=viewer,
        viewer_sleep=0.0,
        print_interval=1,
    )

    assert viewer.sync_count == 1
    assert env.targets is not None
    assert len(overlay.payloads) == 1
    assert overlay.payloads[0].frame.shape == (16, 16, 3)


def test_check_one_finger_teleop_main_reports_startup_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_to_start(**_kwargs: object) -> int:
        raise HandTrackerError("MediaPipe is required for hand tracking.")

    monkeypatch.setattr(check_one_finger_teleop, "run_one_finger_teleop", fail_to_start)

    result = check_one_finger_teleop.main(["--camera-id", "0"])
    captured = capsys.readouterr()

    assert result == 2
    assert "ERROR: MediaPipe is required for hand tracking." in captured.err
