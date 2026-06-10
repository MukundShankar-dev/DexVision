from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from dexvision.apps import check_hand_features
from dexvision.features.hand_features import (
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

_NON_THUMB_FINGER_INDICES = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_NON_THUMB_CURL_FIELDS = {
    "index": "index_curl",
    "middle": "middle_curl",
    "ring": "ring_curl",
    "pinky": "pinky_curl",
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


def _curl_finger(
    landmarks: np.ndarray,
    indices: tuple[int, int, int, int],
    *,
    horizontal_sign: float,
) -> None:
    base, mid, distal, tip = indices
    landmarks[mid] = landmarks[base] + [0.0, 0.20, 0.0]
    landmarks[distal] = landmarks[mid] + [0.16 * horizontal_sign, 0.0, 0.0]
    landmarks[tip] = landmarks[distal] + [0.0, -0.16, 0.0]


def _fist() -> np.ndarray:
    landmarks = _open_hand()
    landmarks[1:5] = [
        [-0.25, 0.20, 0.0],
        [-0.37, 0.32, 0.0],
        [-0.25, 0.32, 0.0],
        [-0.25, 0.20, 0.0],
    ]
    _curl_finger(landmarks, (5, 6, 7, 8), horizontal_sign=1.0)
    _curl_finger(landmarks, (9, 10, 11, 12), horizontal_sign=1.0)
    _curl_finger(landmarks, (13, 14, 15, 16), horizontal_sign=-1.0)
    _curl_finger(landmarks, (17, 18, 19, 20), horizontal_sign=-1.0)
    return landmarks


def _pointing_hand() -> np.ndarray:
    landmarks = _fist()
    open_landmarks = _open_hand()
    landmarks[5:9] = open_landmarks[5:9]
    return landmarks


def _index_curled_other_fingers_open() -> np.ndarray:
    landmarks = _open_hand()
    _curl_finger(landmarks, (5, 6, 7, 8), horizontal_sign=1.0)
    return landmarks


def _finger_curled_other_fingers_open(finger: str) -> np.ndarray:
    landmarks = _open_hand()
    _curl_finger(
        landmarks,
        _NON_THUMB_FINGER_INDICES[finger],
        horizontal_sign=_CURL_DIRECTIONS[finger],
    )
    return landmarks


def _finger_extended_other_fingers_curled(finger: str) -> np.ndarray:
    landmarks = _fist()
    open_landmarks = _open_hand()
    for landmark_index in _NON_THUMB_FINGER_INDICES[finger]:
        landmarks[landmark_index] = open_landmarks[landmark_index]
    return landmarks


def _extended_index_with_local_angle_noise_and_curled_neighbors() -> np.ndarray:
    landmarks = _pointing_hand()
    landmarks[5] = [-0.18, 0.36, 0.0]
    landmarks[6] = [0.02, 0.56, 0.0]
    landmarks[7] = [0.02, 0.78, 0.0]
    landmarks[8] = [0.02, 1.00, 0.0]
    return landmarks


def _ambiguous_compact_index_image() -> np.ndarray:
    landmarks = _fist()
    landmarks[5] = [0.38, 0.40, 0.0]
    landmarks[6] = [0.20, 0.40, 0.0]
    landmarks[7] = [0.30, 0.40, 0.0]
    landmarks[8] = [0.40, 0.40, 0.0]
    return landmarks


def _realistic_thumb_fist() -> np.ndarray:
    landmarks = _open_hand()
    segment = 0.12
    angle = np.deg2rad(50.0)
    landmarks[1] = [-0.25, 0.20, 0.0]
    landmarks[2] = landmarks[1] + [segment, 0.0, 0.0]
    landmarks[3] = landmarks[2] + [segment * np.cos(angle), segment * np.sin(angle), 0.0]
    landmarks[4] = landmarks[3] + [segment, 0.0, 0.0]
    return landmarks


def test_open_hand_gives_low_curl_values() -> None:
    features = compute_hand_features(_open_hand(), confidence=0.93)

    assert features.thumb_curl < 0.1
    assert features.index_curl < 0.1
    assert features.middle_curl < 0.1
    assert features.ring_curl < 0.1
    assert features.pinky_curl < 0.1
    assert features.confidence == pytest.approx(0.93)


def test_fist_gives_high_curl_values() -> None:
    features = compute_hand_features(_fist())

    assert features.thumb_curl > 0.65
    assert features.index_curl > 0.9
    assert features.middle_curl > 0.9
    assert features.ring_curl > 0.9
    assert features.pinky_curl > 0.9


def test_realistic_thumb_fist_maps_above_half_curl() -> None:
    features = compute_hand_features(_realistic_thumb_fist())

    assert features.thumb_curl > 0.9


def test_pointing_hand_keeps_index_low_and_other_fingers_curled() -> None:
    features = compute_hand_features(_pointing_hand())

    assert features.index_curl <= 0.25
    assert features.middle_curl > 0.9
    assert features.ring_curl > 0.9
    assert features.pinky_curl > 0.9


def test_index_curl_pass_criteria_are_decoupled_from_other_fingers() -> None:
    assert compute_hand_features(_open_hand()).index_curl <= 0.25
    assert compute_hand_features(_pointing_hand()).index_curl <= 0.25
    assert compute_hand_features(_index_curled_other_fingers_open()).index_curl >= 0.65
    assert compute_hand_features(_fist()).index_curl >= 0.65


def test_non_thumb_finger_curls_are_decoupled_from_other_fingers() -> None:
    for finger, field_name in _NON_THUMB_CURL_FIELDS.items():
        extended_features = compute_hand_features(_finger_extended_other_fingers_curled(finger))
        curled_features = compute_hand_features(_finger_curled_other_fingers_open(finger))

        assert getattr(extended_features, field_name) <= 0.25
        assert getattr(curled_features, field_name) >= 0.65


def test_index_curl_stays_low_when_extended_index_chain_is_stable() -> None:
    features = compute_hand_features(_extended_index_with_local_angle_noise_and_curled_neighbors())

    assert features.index_curl <= 0.25


def test_thumb_index_pinch_distance_changes_visibly() -> None:
    open_features = compute_hand_features(_open_hand())
    pinch_landmarks = _open_hand()
    pinch_landmarks[4] = pinch_landmarks[8] + [0.02, 0.0, 0.0]

    pinch_features = compute_hand_features(pinch_landmarks)

    assert pinch_features.pinch_thumb_index < open_features.pinch_thumb_index
    assert pinch_features.pinch_thumb_index < 0.1


def test_missing_tracking_returns_finite_neutral_features() -> None:
    result = HandTracker.no_hand_result(timestamp=2.5)

    features = extract_hand_features(result)

    assert features == no_hand_features()
    assert all(np.isfinite(feature_values(features)))


def test_extract_hand_features_uses_tracking_confidence_and_landmarks() -> None:
    landmarks = _open_hand()
    result = HandTrackingResult(
        detected=True,
        handedness="Right",
        confidence=0.82,
        image_landmarks=landmarks,
        world_landmarks=None,
        timestamp=4.0,
    )

    features = extract_hand_features(result)

    assert features.confidence == pytest.approx(0.82)
    assert features.index_curl < 0.1
    assert features.palm_roll == features.palm_roll_proxy
    assert features.palm_pitch == features.palm_pitch_proxy


def test_extract_hand_features_keeps_index_low_when_only_world_index_is_curled() -> None:
    result = HandTrackingResult(
        detected=True,
        handedness="Right",
        confidence=0.91,
        image_landmarks=_pointing_hand(),
        world_landmarks=_fist(),
        timestamp=5.0,
    )

    features = extract_hand_features(result)

    assert features.index_curl <= 0.25
    assert features.middle_curl > 0.9


def test_extract_hand_features_vetoes_world_curl_for_visible_extended_non_thumb_fingers() -> None:
    for finger, field_name in _NON_THUMB_CURL_FIELDS.items():
        result = HandTrackingResult(
            detected=True,
            handedness="Right",
            confidence=0.91,
            image_landmarks=_finger_extended_other_fingers_curled(finger),
            world_landmarks=_fist(),
            timestamp=5.0,
        )

        features = extract_hand_features(result)

        assert getattr(features, field_name) <= 0.25


def test_extract_hand_features_does_not_veto_curled_world_index_from_ambiguous_image() -> None:
    result = HandTrackingResult(
        detected=True,
        handedness="Right",
        confidence=0.91,
        image_landmarks=_ambiguous_compact_index_image(),
        world_landmarks=_fist(),
        timestamp=5.0,
    )

    features = extract_hand_features(result)

    assert features.index_curl >= 0.65


def test_extract_hand_features_keeps_index_high_when_image_and_world_are_curled() -> None:
    result = HandTrackingResult(
        detected=True,
        handedness="Right",
        confidence=0.91,
        image_landmarks=_fist(),
        world_landmarks=_fist(),
        timestamp=5.0,
    )

    features = extract_hand_features(result)

    assert features.index_curl >= 0.65


def test_hand_features_validate_landmark_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_hand_features(np.zeros((20, 3), dtype=np.float32))


def test_hand_features_sanitize_nonfinite_landmarks_and_confidence() -> None:
    landmarks = _open_hand()
    landmarks[8] = [np.nan, np.inf, -np.inf]

    features = compute_hand_features(landmarks, confidence=float("nan"))

    assert all(np.isfinite(feature_values(features)))
    assert features.confidence == 0.0


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
