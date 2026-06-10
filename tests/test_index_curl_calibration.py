from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dexvision.apps import calibrate_index_curl
from dexvision.perception.hand_tracker import HandTrackerError


ROOT = Path(__file__).resolve().parents[1]


def test_summarize_index_curl_samples_returns_expected_stats() -> None:
    stats = calibrate_index_curl.summarize_index_curl_samples([0.2, 0.4, 0.3])

    assert stats.count == 3
    assert stats.minimum == pytest.approx(0.2)
    assert stats.median == pytest.approx(0.3)
    assert stats.maximum == pytest.approx(0.4)
    assert stats.mean == pytest.approx(0.3)
    assert stats.std == pytest.approx(0.081649658)


def test_summarize_index_curl_samples_rejects_empty_or_nonfinite_samples() -> None:
    with pytest.raises(
        calibrate_index_curl.IndexCurlCalibrationError,
        match="No finite",
    ):
        calibrate_index_curl.summarize_index_curl_samples([])

    with pytest.raises(
        calibrate_index_curl.IndexCurlCalibrationError,
        match="No finite",
    ):
        calibrate_index_curl.summarize_index_curl_samples([float("nan"), float("inf")])


def test_format_stats_is_human_readable() -> None:
    stats = calibrate_index_curl.summarize_index_curl_samples([0.1, 0.2, 0.3])

    text = calibrate_index_curl._format_stats("Raw index_curl", stats)

    assert "Raw index_curl" in text
    assert "median=0.200" in text
    assert "std=" in text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"seconds": 0.0, "warmup_seconds": 1.0, "min_sample_confidence": 0.5},
            "seconds",
        ),
        (
            {"seconds": 1.0, "warmup_seconds": -0.1, "min_sample_confidence": 0.5},
            "warmup_seconds",
        ),
        (
            {"seconds": 1.0, "warmup_seconds": 0.0, "min_sample_confidence": 1.5},
            "min_sample_confidence",
        ),
    ],
)
def test_validate_run_parameters_rejects_invalid_values(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calibrate_index_curl._validate_run_parameters(**kwargs)


def test_calibrate_index_curl_help_runs_without_real_webcam() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.calibrate_index_curl", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0
    assert "--pose" in result.stdout
    assert "open" in result.stdout
    assert "curled" in result.stdout
    assert "--seconds" in result.stdout
    assert "--json" in result.stdout


def test_calibrate_index_curl_main_reports_tracker_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_to_start(**_kwargs: object) -> int:
        raise HandTrackerError("MediaPipe is required for hand tracking.")

    monkeypatch.setattr(calibrate_index_curl, "run_calibration", fail_to_start)

    result = calibrate_index_curl.main(["--camera-id", "0", "--pose", "open"])
    captured = capsys.readouterr()

    assert result == 2
    assert "ERROR: MediaPipe is required for hand tracking." in captured.err
