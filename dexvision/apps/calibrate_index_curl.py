"""Measure observed index-curl ranges for Level 1 one-finger teleop."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dexvision.camera.opencv_camera import CameraOpenError, OpenCVCamera
from dexvision.features.hand_features import extract_hand_features
from dexvision.features.smoothing import FeatureSmoother, LowConfidenceBehavior
from dexvision.perception.hand_tracker import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    HandTracker,
    HandTrackerError,
)


PoseName = Literal["open", "curled"]
DEFAULT_SECONDS = 3.0
DEFAULT_WARMUP_SECONDS = 1.0
DEFAULT_MIN_SAMPLE_CONFIDENCE = 0.5
DEFAULT_SMOOTHING_ALPHA = 0.75
DEFAULT_MIN_SMOOTHING_CONFIDENCE = 0.2
DEFAULT_LOW_CONFIDENCE_BEHAVIOR: LowConfidenceBehavior = "decay"
DEFAULT_DECAY_ALPHA = 0.15


class IndexCurlCalibrationError(RuntimeError):
    """Raised when index-curl calibration cannot produce usable stats."""


@dataclass(frozen=True)
class IndexCurlStats:
    """Summary statistics for one stream of index-curl samples."""

    count: int
    minimum: float
    median: float
    maximum: float
    mean: float
    std: float


@dataclass(frozen=True)
class IndexCurlCalibrationResult:
    """Calibration summary for one requested pose."""

    pose: PoseName
    seconds: float
    warmup_seconds: float
    total_frames: int
    detected_frames: int
    recorded_frames: int
    raw_index_curl: IndexCurlStats
    smoothed_index_curl: IndexCurlStats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect index_curl stats for an open or curled index pose."
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested capture height.",
    )
    parser.add_argument(
        "--pose",
        choices=("open", "curled"),
        required=True,
        help="Pose being held during this calibration capture.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help="Seconds of samples to record after warmup.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=DEFAULT_WARMUP_SECONDS,
        help="Seconds to warm up tracking and smoothing before recording samples.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "MediaPipe Tasks hand-landmarker model path. Defaults to "
            f"{DEFAULT_HAND_LANDMARKER_MODEL} when legacy MediaPipe Hands is "
            "unavailable."
        ),
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe detection confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum MediaPipe tracking confidence.",
    )
    parser.add_argument(
        "--assume-mirrored-input",
        action="store_true",
        help="Keep MediaPipe handedness labels for selfie-mirrored camera images.",
    )
    parser.add_argument(
        "--min-sample-confidence",
        type=float,
        default=DEFAULT_MIN_SAMPLE_CONFIDENCE,
        help="Only record detected samples with at least this feature confidence.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=DEFAULT_SMOOTHING_ALPHA,
        help="EMA smoothing alpha in (0.0, 1.0]; match teleop defaults by default.",
    )
    parser.add_argument(
        "--min-smoothing-confidence",
        type=float,
        default=DEFAULT_MIN_SMOOTHING_CONFIDENCE,
        help="Decay smoothed controls below this feature confidence.",
    )
    parser.add_argument(
        "--low-confidence-behavior",
        choices=("hold", "decay"),
        default=DEFAULT_LOW_CONFIDENCE_BEHAVIOR,
        help="How smoothed controls behave when tracking confidence is low.",
    )
    parser.add_argument(
        "--decay-alpha",
        type=float,
        default=DEFAULT_DECAY_ALPHA,
        help="EMA alpha used only when --low-confidence-behavior=decay.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON after the human-readable summary.",
    )
    return parser


def summarize_index_curl_samples(samples: Sequence[float]) -> IndexCurlStats:
    """Return finite summary stats for one list of index-curl samples."""

    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise IndexCurlCalibrationError("No finite index_curl samples were recorded.")

    return IndexCurlStats(
        count=int(values.size),
        minimum=float(np.min(values)),
        median=float(np.median(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
    )


def run_calibration(
    *,
    camera_id: int,
    width: int,
    height: int,
    pose: PoseName,
    seconds: float,
    warmup_seconds: float,
    model_path: Path | None,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    assume_mirrored_input: bool,
    min_sample_confidence: float,
    smoothing_alpha: float,
    min_smoothing_confidence: float,
    low_confidence_behavior: LowConfidenceBehavior,
    decay_alpha: float,
    print_json: bool,
) -> int:
    """Collect and print index-curl calibration stats."""

    _validate_run_parameters(
        seconds=seconds,
        warmup_seconds=warmup_seconds,
        min_sample_confidence=min_sample_confidence,
    )
    smoother = FeatureSmoother(
        alpha=smoothing_alpha,
        min_confidence=min_smoothing_confidence,
        low_confidence_behavior=low_confidence_behavior,
        decay_alpha=decay_alpha,
    )

    print("DexVision index curl calibration")
    print(f"Pose: {pose}")
    print(f"Camera: id={camera_id}, width={width}, height={height}")
    print(f"Hand tracker model: {model_path or DEFAULT_HAND_LANDMARKER_MODEL}")
    print(
        "Timing: "
        f"warmup={warmup_seconds:.2f}s, record={seconds:.2f}s, "
        f"min_sample_confidence={min_sample_confidence:.2f}"
    )
    print("Hold the requested pose steady until the summary prints.")

    raw_samples: list[float] = []
    smoothed_samples: list[float] = []
    total_frames = 0
    detected_frames = 0
    start_time = time.monotonic()
    record_start_time = start_time + warmup_seconds
    end_time = record_start_time + seconds

    with (
        OpenCVCamera(camera_id=camera_id, width=width, height=height) as camera,
        HandTracker(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_path=model_path,
            assume_mirrored_input=assume_mirrored_input,
        ) as tracker,
    ):
        while time.monotonic() < end_time:
            camera_result = camera.read()
            if not camera_result.success or camera_result.frame is None:
                continue

            total_frames += 1
            tracking_result = tracker.process(
                camera_result.frame,
                timestamp=camera_result.timestamp,
            )
            raw_features = extract_hand_features(tracking_result)
            smoothed_features = smoother.update(raw_features)
            if tracking_result.detected:
                detected_frames += 1

            if time.monotonic() < record_start_time:
                continue
            if (
                not tracking_result.detected
                or raw_features.confidence < min_sample_confidence
            ):
                continue

            raw_samples.append(raw_features.index_curl)
            smoothed_samples.append(smoothed_features.index_curl)

    result = IndexCurlCalibrationResult(
        pose=pose,
        seconds=seconds,
        warmup_seconds=warmup_seconds,
        total_frames=total_frames,
        detected_frames=detected_frames,
        recorded_frames=len(raw_samples),
        raw_index_curl=summarize_index_curl_samples(raw_samples),
        smoothed_index_curl=summarize_index_curl_samples(smoothed_samples),
    )
    _print_result(result)
    if print_json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


def _print_result(result: IndexCurlCalibrationResult) -> None:
    print("")
    print("Calibration summary")
    print(f"Pose: {result.pose}")
    print(
        "Frames: "
        f"total={result.total_frames}, detected={result.detected_frames}, "
        f"recorded={result.recorded_frames}"
    )
    print(_format_stats("Raw index_curl", result.raw_index_curl))
    print(_format_stats("Smoothed index_curl", result.smoothed_index_curl))
    print("")
    print("Compare open and curled medians:")
    print("- wide separation means retargeting response/limits need tuning")
    print("- narrow separation means the Level 1.3 index feature needs improvement")


def _format_stats(label: str, stats: IndexCurlStats) -> str:
    return (
        f"{label}: "
        f"count={stats.count}, min={stats.minimum:.3f}, "
        f"median={stats.median:.3f}, max={stats.maximum:.3f}, "
        f"mean={stats.mean:.3f}, std={stats.std:.3f}"
    )


def _validate_run_parameters(
    *,
    seconds: float,
    warmup_seconds: float,
    min_sample_confidence: float,
) -> None:
    if seconds <= 0.0:
        raise ValueError("seconds must be positive.")
    if warmup_seconds < 0.0:
        raise ValueError("warmup_seconds must be non-negative.")
    if not 0.0 <= min_sample_confidence <= 1.0:
        raise ValueError("min_sample_confidence must be in [0.0, 1.0].")


def _is_cv2_error(exc: Exception) -> bool:
    return exc.__class__.__module__.split(".", maxsplit=1)[0] == "cv2"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_calibration(
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            pose=args.pose,
            seconds=args.seconds,
            warmup_seconds=args.warmup_seconds,
            model_path=args.model_path,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            assume_mirrored_input=args.assume_mirrored_input,
            min_sample_confidence=args.min_sample_confidence,
            smoothing_alpha=args.smoothing_alpha,
            min_smoothing_confidence=args.min_smoothing_confidence,
            low_confidence_behavior=args.low_confidence_behavior,
            decay_alpha=args.decay_alpha,
            print_json=args.json,
        )
    except (
        CameraOpenError,
        HandTrackerError,
        IndexCurlCalibrationError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Index curl calibration closed cleanly.")
        return 130
    except Exception as exc:
        if _is_cv2_error(exc):
            print(
                "ERROR: OpenCV failed during camera calibration. Ensure this is "
                f"running in a desktop session with camera access. Details: {exc}",
                file=sys.stderr,
            )
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
