from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from dexvision.apps import check_hand_features
from dexvision.features.hand_features import (
    FingerCalibration,
    FingerState,
    HandFeatureCalibration,
    HandFeatures,
    compute_hand_features,
    extract_hand_features,
    feature_values,
    no_hand_features,
)
from dexvision.perception.hand_tracker import (
    HandTracker,
    HandTrackerError,
    HandTrackingResult,
)


_LONG_FINGERS = ("index", "middle", "ring", "pinky")
_FINGER_INDICES = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_CURL_DIRECTIONS = {
    "index": 1.0,
    "middle": 1.0,
    "ring": -1.0,
    "pinky": -1.0,
}


def _open_hand() -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [0.0, 0.0, 0.0]
    landmarks[1:5] = [
        [-0.25, 0.20, 0.0],
        [-0.38, 0.34, 0.0],
        [-0.51, 0.48, 0.0],
        [-0.64, 0.62, 0.0],
    ]
    landmarks[5:9] = [
        [-0.18, 0.36, 0.0],
        [-0.18, 0.58, 0.0],
        [-0.18, 0.78, 0.0],
        [-0.18, 0.98, 0.0],
    ]
    landmarks[9:13] = [
        [0.00, 0.40, 0.0],
        [0.00, 0.66, 0.0],
        [0.00, 0.90, 0.0],
        [0.00, 1.12, 0.0],
    ]
    landmarks[13:17] = [
        [0.18, 0.36, 0.0],
        [0.18, 0.58, 0.0],
        [0.18, 0.78, 0.0],
        [0.18, 0.96, 0.0],
    ]
    landmarks[17:21] = [
        [0.34, 0.30, 0.0],
        [0.34, 0.50, 0.0],
        [0.34, 0.66, 0.0],
        [0.34, 0.82, 0.0],
    ]
    return landmarks


def _curl_long_finger(landmarks: np.ndarray, finger: str) -> None:
    base, pip, dip, tip = _FINGER_INDICES[finger]
    horizontal_sign = _CURL_DIRECTIONS[finger]
    landmarks[pip] = landmarks[base] + [0.0, 0.20, 0.0]
    landmarks[dip] = landmarks[pip] + [0.16 * horizontal_sign, 0.0, 0.0]
    landmarks[tip] = landmarks[dip] + [0.0, -0.16, 0.0]


def _curl_thumb(landmarks: np.ndarray) -> None:
    landmarks[1:5] = [
        [-0.25, 0.20, 0.0],
        [-0.37, 0.32, 0.0],
        [-0.25, 0.32, 0.0],
        [-0.25, 0.20, 0.0],
    ]


def _fist() -> np.ndarray:
    landmarks = _open_hand()
    _curl_thumb(landmarks)
    for finger in _LONG_FINGERS:
        _curl_long_finger(landmarks, finger)
    return landmarks


def _single_finger_up(finger: str) -> np.ndarray:
    landmarks = _fist()
    open_landmarks = _open_hand()
    for landmark_index in _FINGER_INDICES[finger]:
        landmarks[landmark_index] = open_landmarks[landmark_index]
    return landmarks


def _peace_sign() -> np.ndarray:
    landmarks = _fist()
    open_landmarks = _open_hand()
    for finger in ("index", "middle"):
        for landmark_index in _FINGER_INDICES[finger]:
            landmarks[landmark_index] = open_landmarks[landmark_index]
    return landmarks


def _index_curled_other_fingers_open() -> np.ndarray:
    landmarks = _open_hand()
    _curl_long_finger(landmarks, "index")
    return landmarks


def _states(features: HandFeatures) -> dict[str, FingerState]:
    return {
        "thumb": features.thumb,
        "index": features.index,
        "middle": features.middle,
        "ring": features.ring,
        "pinky": features.pinky,
    }


def test_open_hand_produces_extended_local_finger_states() -> None:
    features = compute_hand_features(_open_hand(), confidence=0.93)

    assert features.palm.valid
    assert features.confidence == pytest.approx(0.93)
    for finger, state in _states(features).items():
        assert state.valid, finger
        assert state.curl <= 0.25, finger
        assert state.extension >= 0.55, finger
        if finger in _LONG_FINGERS:
            assert getattr(features, f"{finger}_bend") <= 0.45, finger
    assert features.index_curl == pytest.approx(features.index.curl)
    assert features.middle_curl == pytest.approx(features.middle.curl)
    assert features.ring_curl == pytest.approx(features.ring.curl)
    assert features.pinky_curl == pytest.approx(features.pinky.curl)


def test_fist_produces_curled_local_finger_states() -> None:
    features = compute_hand_features(_fist())

    for finger, state in _states(features).items():
        assert state.valid, finger
        assert state.curl >= 0.65, finger
        assert not state.is_up, finger
        if finger in _LONG_FINGERS:
            assert getattr(features, f"{finger}_bend") >= 0.65, finger


@pytest.mark.parametrize("finger", _LONG_FINGERS)
def test_single_non_thumb_finger_up_pose_is_local(finger: str) -> None:
    features = compute_hand_features(_single_finger_up(finger))

    for other_finger in _LONG_FINGERS:
        state = getattr(features, other_finger)
        if other_finger == finger:
            assert state.curl <= 0.25
            assert state.extension >= 0.55
            assert getattr(features, f"{other_finger}_bend") <= 0.45
            assert state.is_up
        else:
            assert state.curl >= 0.65
            assert state.extension <= 0.35
            assert getattr(features, f"{other_finger}_bend") >= 0.65
            assert not state.is_up


def test_peace_sign_keeps_only_index_and_middle_up() -> None:
    features = compute_hand_features(_peace_sign())

    assert features.index.is_up
    assert features.middle.is_up
    assert not features.ring.is_up
    assert not features.pinky.is_up
    assert features.index_curl <= 0.25
    assert features.middle_curl <= 0.25
    assert features.ring_curl >= 0.65
    assert features.pinky_curl >= 0.65


def test_moving_index_does_not_move_other_finger_features() -> None:
    open_features = compute_hand_features(_open_hand())
    index_moved_features = compute_hand_features(_index_curled_other_fingers_open())

    assert index_moved_features.index_curl >= 0.65
    for finger in ("middle", "ring", "pinky"):
        before = getattr(open_features, finger)
        after = getattr(index_moved_features, finger)
        assert after.curl == pytest.approx(before.curl, abs=0.02), finger
        assert after.extension == pytest.approx(before.extension, abs=0.02), finger
        assert after.is_up == before.is_up, finger


def test_thumb_uses_separate_logic_and_tracks_pinching() -> None:
    open_features = compute_hand_features(_open_hand())
    fist_features = compute_hand_features(_fist())
    pinch_landmarks = _open_hand()
    pinch_landmarks[4] = pinch_landmarks[8] + [0.02, 0.0, 0.0]

    pinch_features = compute_hand_features(pinch_landmarks)

    assert open_features.thumb.curl <= 0.25
    assert open_features.thumb.extension >= 0.55
    assert fist_features.thumb.curl >= 0.65
    assert pinch_features.pinch_thumb_index < open_features.pinch_thumb_index
    assert pinch_features.pinch_thumb_index < 0.1
    assert pinch_features.thumb.abduction is not None


def test_invalid_landmarks_and_no_hand_frames_stay_finite() -> None:
    invalid_landmarks = np.full((21, 3), np.nan, dtype=np.float32)
    invalid_features = compute_hand_features(invalid_landmarks, confidence=float("nan"))
    missing_features = extract_hand_features(HandTracker.no_hand_result(timestamp=2.5))

    assert invalid_features.confidence == 0.0
    assert all(np.isfinite(feature_values(invalid_features)))
    assert all(not state.valid for state in _states(invalid_features).values())
    assert missing_features == no_hand_features()
    assert missing_features.index_bend == pytest.approx(0.0)
    assert missing_features.middle_bend == pytest.approx(0.0)
    assert missing_features.ring_bend == pytest.approx(0.0)
    assert missing_features.pinky_bend == pytest.approx(0.0)
    assert all(np.isfinite(feature_values(missing_features)))


def test_extract_hand_features_defaults_to_visible_image_landmarks() -> None:
    result = HandTrackingResult(
        detected=True,
        handedness="Right",
        confidence=0.82,
        image_landmarks=_open_hand(),
        world_landmarks=_fist(),
        timestamp=4.0,
    )

    visible_features = extract_hand_features(result)
    world_features = extract_hand_features(result, prefer_world_landmarks=True)

    assert visible_features.confidence == pytest.approx(0.82)
    assert visible_features.index_curl <= 0.25
    assert world_features.index_curl >= 0.65
    assert visible_features.palm_roll == visible_features.palm_roll_proxy
    assert visible_features.palm_pitch == visible_features.palm_pitch_proxy


def test_legacy_constructor_and_calibration_api_remain_available() -> None:
    legacy = HandFeatures(
        thumb_curl=0.1,
        index_curl=0.7,
        middle_curl=0.0,
        ring_curl=0.0,
        pinky_curl=0.0,
        pinch_thumb_index=0.5,
        confidence=1.0,
    )
    calibration = HandFeatureCalibration(
        fingers={
            "index": FingerCalibration(
                curl_min=0.2,
                curl_max=0.8,
                extension_min=0.0,
                extension_max=1.0,
            )
        },
        open_hand_baseline=no_hand_features(),
        fist_baseline=no_hand_features(),
    )
    calibrated = compute_hand_features(_single_finger_up("index"), calibration=calibration)

    assert legacy.index.curl == pytest.approx(0.7)
    assert legacy.index_curl == pytest.approx(0.7)
    assert legacy.index.extension == pytest.approx(0.3)
    assert legacy.index_bend == pytest.approx(0.7)
    assert calibrated.index.curl == pytest.approx(0.0)


def test_hand_features_validate_landmark_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_hand_features(np.zeros((20, 3), dtype=np.float32))


def test_check_hand_features_help_runs_without_real_webcam() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.check_hand_features", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--camera-id" in result.stdout
    assert "--model-path" in result.stdout
    assert "--min-detection-confidence" in result.stdout
    assert "--min-tracking-confidence" in result.stdout
    assert "--assume-mirrored-input" in result.stdout


def test_check_hand_features_main_reports_tracker_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_to_start(
        camera_id: int,
        width: int,
        height: int,
        model_path: object,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        assume_mirrored_input: bool,
    ) -> int:
        del camera_id, width, height, model_path
        del min_detection_confidence, min_tracking_confidence, assume_mirrored_input
        raise HandTrackerError("MediaPipe is required for hand tracking.")

    monkeypatch.setattr(check_hand_features, "run_hand_features", fail_to_start)

    result = check_hand_features.main(["--camera-id", "0"])
    captured = capsys.readouterr()

    assert result == 2
    assert "ERROR: MediaPipe is required for hand tracking." in captured.err
