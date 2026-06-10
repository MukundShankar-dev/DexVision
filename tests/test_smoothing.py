from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from dexvision.apps import check_smoothing
from dexvision.features.hand_features import HandFeatures, feature_values, no_hand_features
from dexvision.features.smoothing import FeatureSmoother, SmoothingConfig, sanitize_hand_features
from dexvision.perception.hand_tracker import HandTrackerError


def _features(
    value: float,
    *,
    confidence: float = 1.0,
    pinch: float | None = None,
    palm_roll: float = 0.0,
    palm_pitch: float = 0.0,
) -> HandFeatures:
    return HandFeatures(
        thumb_curl=value,
        index_curl=value,
        middle_curl=value,
        ring_curl=value,
        pinky_curl=value,
        pinch_thumb_index=value if pinch is None else pinch,
        palm_roll_proxy=palm_roll,
        palm_pitch_proxy=palm_pitch,
        confidence=confidence,
    )


def test_step_response_moves_partway_toward_new_value() -> None:
    smoother = FeatureSmoother(alpha=0.25)

    first = smoother.update(_features(0.0))
    second = smoother.update(_features(1.0))

    assert first.index_curl == pytest.approx(0.0)
    assert second.index_curl == pytest.approx(0.25)
    assert second.thumb_curl == pytest.approx(0.25)
    assert 0.0 < second.index_curl < 1.0


def test_noisy_input_has_less_variation_after_smoothing() -> None:
    smoother = FeatureSmoother(alpha=0.2)
    raw_values = np.asarray([0.50, 0.80, 0.20, 0.78, 0.22, 0.76, 0.24, 0.74], dtype=np.float32)

    smoothed_values = np.asarray(
        [smoother.update(_features(float(value))).index_curl for value in raw_values],
        dtype=np.float32,
    )

    assert float(np.std(smoothed_values)) < float(np.std(raw_values))
    assert smoothed_values[-1] != pytest.approx(raw_values[-1])


def test_missing_input_holds_last_controls_and_reports_zero_confidence() -> None:
    smoother = FeatureSmoother(alpha=0.5, min_confidence=0.4)
    stable = smoother.update(_features(0.8, confidence=0.9))

    missing = smoother.update(None)

    assert missing.index_curl == pytest.approx(stable.index_curl)
    assert missing.thumb_curl == pytest.approx(stable.thumb_curl)
    assert missing.confidence == 0.0
    assert all(np.isfinite(feature_values(missing)))


def test_low_confidence_input_holds_controls_without_using_noisy_values() -> None:
    smoother = FeatureSmoother(alpha=0.5, min_confidence=0.4)
    smoother.update(_features(0.7, confidence=0.9))

    low_confidence = smoother.update(_features(0.0, confidence=0.1))

    assert low_confidence.index_curl == pytest.approx(0.7)
    assert low_confidence.confidence == pytest.approx(0.1)
    assert all(np.isfinite(feature_values(low_confidence)))


def test_decay_behavior_moves_low_confidence_controls_toward_neutral() -> None:
    smoother = FeatureSmoother(
        alpha=0.5,
        min_confidence=0.4,
        low_confidence_behavior="decay",
        decay_alpha=0.25,
    )
    smoother.update(_features(0.8, confidence=0.9))

    decayed = smoother.update(_features(0.0, confidence=0.1))

    assert decayed.index_curl == pytest.approx(0.6)
    assert decayed.confidence == pytest.approx(0.1)


def test_sanitize_hand_features_clips_nonfinite_values() -> None:
    dirty = HandFeatures(
        thumb_curl=float("nan"),
        index_curl=2.0,
        middle_curl=-1.0,
        ring_curl=float("inf"),
        pinky_curl=0.5,
        pinch_thumb_index=1.5,
        palm_roll_proxy=-2.0,
        palm_pitch_proxy=float("nan"),
        confidence=float("inf"),
    )

    clean = sanitize_hand_features(dirty)

    assert clean == HandFeatures(
        thumb_curl=0.0,
        index_curl=1.0,
        middle_curl=0.0,
        ring_curl=0.0,
        pinky_curl=0.5,
        pinch_thumb_index=1.0,
        palm_roll_proxy=-1.0,
        palm_pitch_proxy=0.0,
        confidence=0.0,
    )


def test_first_low_confidence_update_returns_finite_neutral_features() -> None:
    smoother = FeatureSmoother(min_confidence=0.5)

    smoothed = smoother.update(_features(0.9, confidence=0.0))

    assert smoothed == no_hand_features()
    assert all(np.isfinite(feature_values(smoothed)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"min_confidence": 1.5}, "min_confidence"),
        ({"low_confidence_behavior": "freeze"}, "low_confidence_behavior"),
        ({"decay_alpha": -0.1}, "decay_alpha"),
    ],
)
def test_smoothing_config_validates_ranges(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SmoothingConfig(**kwargs)  # type: ignore[arg-type]


def test_check_smoothing_help_runs_without_real_webcam() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.check_smoothing", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--camera-id" in result.stdout
    assert "--smoothing-alpha" in result.stdout
    assert "--min-smoothing-confidence" in result.stdout
    assert "--low-confidence-behavior" in result.stdout
    assert "--assume-mirrored-input" in result.stdout


def test_check_smoothing_main_reports_tracker_errors(
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
        smoothing_alpha: float,
        min_smoothing_confidence: float,
        low_confidence_behavior: object,
        decay_alpha: float,
    ) -> int:
        del camera_id, width, height, model_path
        del min_detection_confidence, min_tracking_confidence
        del assume_mirrored_input, smoothing_alpha, min_smoothing_confidence
        del low_confidence_behavior, decay_alpha
        raise HandTrackerError("MediaPipe is required for hand tracking.")

    monkeypatch.setattr(check_smoothing, "run_smoothing", fail_to_start)

    result = check_smoothing.main(["--camera-id", "0"])
    captured = capsys.readouterr()

    assert result == 2
    assert "ERROR: MediaPipe is required for hand tracking." in captured.err
