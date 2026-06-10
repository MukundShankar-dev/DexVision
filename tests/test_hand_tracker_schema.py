from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dexvision.apps import check_hand_tracking
from dexvision.perception.hand_tracker import HandTracker, HandTrackerError, HandTrackingResult
from dexvision.perception.visualization import draw_hand_tracking


class FakeHands:
    def __init__(self, raw_result: object) -> None:
        self.raw_result = raw_result
        self.closed = False
        self.processed_shape: tuple[int, ...] | None = None

    def process(self, image: np.ndarray) -> object:
        self.processed_shape = image.shape
        return self.raw_result

    def close(self) -> None:
        self.closed = True


def _landmark_list(offset: float = 0.0) -> SimpleNamespace:
    landmarks = [
        SimpleNamespace(
            x=0.1 + offset + (index % 5) * 0.08,
            y=0.2 + (index // 5) * 0.08,
            z=index * 0.001,
        )
        for index in range(21)
    ]
    return SimpleNamespace(landmark=landmarks)


def test_hand_tracking_result_no_hand_schema_is_stable() -> None:
    result = HandTracker.no_hand_result(timestamp=12.5)

    assert result == HandTrackingResult(
        detected=False,
        handedness=None,
        confidence=0.0,
        image_landmarks=None,
        world_landmarks=None,
        timestamp=12.5,
    )


def test_hand_tracker_returns_landmark_arrays_from_synthetic_result() -> None:
    fake_hands = FakeHands(
        SimpleNamespace(
            multi_hand_landmarks=[_landmark_list()],
            multi_hand_world_landmarks=[_landmark_list(offset=0.01)],
            multi_handedness=[
                SimpleNamespace(
                    classification=[SimpleNamespace(label="Right", score=0.91)],
                )
            ],
        )
    )
    tracker = HandTracker(hands_factory=lambda **_kwargs: fake_hands)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    result = tracker.process(frame, timestamp=42.0)
    tracker.close()

    assert result.detected
    assert result.handedness == "Right"
    assert result.confidence == pytest.approx(0.91)
    assert result.image_landmarks is not None
    assert result.image_landmarks.shape == (21, 3)
    assert result.image_landmarks.dtype == np.float32
    assert result.world_landmarks is not None
    assert result.world_landmarks.shape == (21, 3)
    assert result.timestamp == 42.0
    assert fake_hands.processed_shape == frame.shape
    assert fake_hands.closed


def test_hand_tracker_parses_tasks_style_result() -> None:
    fake_hands = FakeHands(
        SimpleNamespace(
            hand_landmarks=[_landmark_list()],
            hand_world_landmarks=[_landmark_list(offset=0.02)],
            handedness=[[SimpleNamespace(category_name="Left", score=0.83)]],
        )
    )
    tracker = HandTracker(hands_factory=lambda **_kwargs: fake_hands)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    result = tracker.process(frame, timestamp=43.0)
    tracker.close()

    assert result.detected
    assert result.handedness == "Left"
    assert result.confidence == pytest.approx(0.83)
    assert result.image_landmarks is not None
    assert result.image_landmarks.shape == (21, 3)
    assert result.world_landmarks is not None
    assert result.world_landmarks.shape == (21, 3)


def test_hand_tracker_handles_no_hand_frames() -> None:
    fake_hands = FakeHands(
        SimpleNamespace(
            multi_hand_landmarks=[],
            multi_hand_world_landmarks=[],
            multi_handedness=[],
        )
    )
    tracker = HandTracker(hands_factory=lambda **_kwargs: fake_hands)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    result = tracker.process(frame, timestamp=3.0)
    tracker.close()

    assert not result.detected
    assert result.handedness is None
    assert result.confidence == 0.0
    assert result.image_landmarks is None
    assert result.world_landmarks is None
    assert result.timestamp == 3.0


def test_hand_tracker_validates_frame_shape_and_dtype() -> None:
    tracker = HandTracker(
        hands_factory=lambda **_kwargs: FakeHands(SimpleNamespace(multi_hand_landmarks=[]))
    )

    with pytest.raises(ValueError, match="shape"):
        tracker.process(np.zeros((16, 16), dtype=np.uint8))

    with pytest.raises(ValueError, match="dtype"):
        tracker.process(np.zeros((16, 16, 3), dtype=np.float32))

    tracker.close()


def test_hand_tracker_reports_unexpected_landmark_shape() -> None:
    fake_hands = FakeHands(
        SimpleNamespace(
            multi_hand_landmarks=[SimpleNamespace(landmark=[SimpleNamespace(x=0.0, y=0.0, z=0.0)])],
        )
    )
    tracker = HandTracker(hands_factory=lambda **_kwargs: fake_hands)

    with pytest.raises(HandTrackerError, match="Expected 21 hand landmarks"):
        tracker.process(np.zeros((16, 16, 3), dtype=np.uint8))

    tracker.close()


def test_draw_hand_tracking_changes_frame_for_detected_hand() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = HandTrackingResult(
        detected=True,
        handedness="Left",
        confidence=0.87,
        image_landmarks=np.asarray(
            [[0.2 + (index % 5) * 0.1, 0.2 + (index // 5) * 0.1, 0.0] for index in range(21)],
            dtype=np.float32,
        ),
        world_landmarks=None,
        timestamp=1.0,
    )

    output = draw_hand_tracking(frame, result)

    assert output is frame
    assert np.count_nonzero(frame) > 0


def test_draw_hand_tracking_handles_no_hand_result() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    draw_hand_tracking(frame, HandTracker.no_hand_result(timestamp=1.0))

    assert np.count_nonzero(frame) > 0


def test_check_hand_tracking_help_runs_without_real_webcam() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.check_hand_tracking", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--camera-id" in result.stdout
    assert "--model-path" in result.stdout
    assert "--min-detection-confidence" in result.stdout
    assert "--min-tracking-confidence" in result.stdout


def test_check_hand_tracking_main_reports_tracker_errors(
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
    ) -> int:
        del model_path
        raise HandTrackerError("MediaPipe is required for hand tracking.")

    monkeypatch.setattr(check_hand_tracking, "run_hand_tracking", fail_to_start)

    result = check_hand_tracking.main(["--camera-id", "0"])
    captured = capsys.readouterr()

    assert result == 2
    assert "ERROR: MediaPipe is required for hand tracking." in captured.err
