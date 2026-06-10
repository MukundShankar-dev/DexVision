"""Open a camera feed and compare raw versus smoothed hand features."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import suppress
from pathlib import Path

from dexvision.camera.opencv_camera import CameraOpenError, OpenCVCamera
from dexvision.features.hand_features import HandFeatures, extract_hand_features
from dexvision.features.smoothing import FeatureSmoother, LowConfidenceBehavior
from dexvision.perception.hand_tracker import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    HandTracker,
    HandTrackerError,
)
from dexvision.perception.visualization import draw_hand_tracking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display raw and smoothed DexVision hand features."
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "MediaPipe Tasks hand-landmarker model path. Defaults to "
            f"{DEFAULT_HAND_LANDMARKER_MODEL} when legacy MediaPipe Hands is unavailable."
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
        "--smoothing-alpha",
        type=float,
        default=0.35,
        help="EMA smoothing alpha in (0.0, 1.0]; higher values respond faster.",
    )
    parser.add_argument(
        "--min-smoothing-confidence",
        type=float,
        default=0.2,
        help="Freeze or decay smoothed controls below this feature confidence.",
    )
    parser.add_argument(
        "--low-confidence-behavior",
        choices=("hold", "decay"),
        default="hold",
        help="How smoothed controls behave when tracking confidence is low.",
    )
    parser.add_argument(
        "--decay-alpha",
        type=float,
        default=0.05,
        help="EMA alpha used only when --low-confidence-behavior=decay.",
    )
    return parser


def _load_cv2_for_display():
    try:
        import cv2
    except ImportError as exc:
        raise CameraOpenError(
            "OpenCV is required for smoothing display. Install the package providing "
            f"'cv2' ({exc})."
        ) from exc
    return cv2


def _draw_fps(cv2_module, frame, fps: float) -> None:
    cv2_module.putText(
        frame,
        f"FPS: {fps:5.1f}",
        (16, 64),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2_module.LINE_AA,
    )


def _draw_feature_comparison(
    cv2_module,
    frame,
    raw_features: HandFeatures,
    smoothed_features: HandFeatures,
) -> None:
    rows = [
        ("Thumb", raw_features.thumb_curl, smoothed_features.thumb_curl),
        ("Index", raw_features.index_curl, smoothed_features.index_curl),
        ("Middle", raw_features.middle_curl, smoothed_features.middle_curl),
        ("Ring", raw_features.ring_curl, smoothed_features.ring_curl),
        ("Pinky", raw_features.pinky_curl, smoothed_features.pinky_curl),
        ("Pinch", raw_features.pinch_thumb_index, smoothed_features.pinch_thumb_index),
        ("Conf", raw_features.confidence, smoothed_features.confidence),
    ]
    x = 16
    y = 96
    label_width = 72
    bar_width = 118
    bar_height = 12
    value_width = 44
    column_gap = 16
    row_gap = 24
    raw_x = x + label_width
    smooth_x = raw_x + bar_width + value_width + column_gap

    cv2_module.putText(
        frame,
        "Raw",
        (raw_x, y - 12),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame,
        "Smoothed",
        (smooth_x, y - 12),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )

    for index, (label, raw_value, smoothed_value) in enumerate(rows):
        row_y = y + index * row_gap
        color = (255, 190, 40) if label == "Pinch" else (0, 220, 120)
        if label == "Conf":
            color = (255, 220, 110)

        cv2_module.putText(
            frame,
            label,
            (x, row_y + bar_height),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            cv2_module.LINE_AA,
        )
        _draw_bar(
            cv2_module,
            frame,
            raw_x,
            row_y,
            raw_value,
            bar_width,
            bar_height,
            (160, 160, 160),
        )
        _draw_bar(cv2_module, frame, smooth_x, row_y, smoothed_value, bar_width, bar_height, color)
        _draw_value(cv2_module, frame, raw_x + bar_width + 6, row_y, raw_value)
        _draw_value(cv2_module, frame, smooth_x + bar_width + 6, row_y, smoothed_value)

    palm_y = y + len(rows) * row_gap + 8
    cv2_module.putText(
        frame,
        "Palm raw/smoothed roll: "
        f"{raw_features.palm_roll_proxy:+.2f}/{smoothed_features.palm_roll_proxy:+.2f} "
        "pitch: "
        f"{raw_features.palm_pitch_proxy:+.2f}/{smoothed_features.palm_pitch_proxy:+.2f}",
        (x, palm_y),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )


def _draw_bar(
    cv2_module,
    frame,
    x: int,
    y: int,
    value: float,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    safe_value = max(0.0, min(float(value), 1.0))
    filled_width = int(round(width * safe_value))
    cv2_module.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), 1)
    if filled_width > 0:
        cv2_module.rectangle(frame, (x, y), (x + filled_width, y + height), color, -1)


def _draw_value(cv2_module, frame, x: int, y: int, value: float) -> None:
    safe_value = max(0.0, min(float(value), 1.0))
    cv2_module.putText(
        frame,
        f"{safe_value:.2f}",
        (x, y + 12),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )


def run_smoothing(
    camera_id: int,
    width: int,
    height: int,
    model_path: Path | None,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    assume_mirrored_input: bool,
    smoothing_alpha: float,
    min_smoothing_confidence: float,
    low_confidence_behavior: LowConfidenceBehavior,
    decay_alpha: float,
) -> int:
    cv2_module = _load_cv2_for_display()
    window_name = "DexVision Feature Smoothing Check"
    last_frame_time = time.monotonic()
    fps = 0.0
    smoother = FeatureSmoother(
        alpha=smoothing_alpha,
        min_confidence=min_smoothing_confidence,
        low_confidence_behavior=low_confidence_behavior,
        decay_alpha=decay_alpha,
    )

    print("DexVision feature smoothing check")
    print(
        "Opening "
        f"camera_id={camera_id}, width={width}, height={height}, "
        f"model_path={model_path or DEFAULT_HAND_LANDMARKER_MODEL}, "
        f"min_detection_confidence={min_detection_confidence}, "
        f"min_tracking_confidence={min_tracking_confidence}, "
        f"assume_mirrored_input={assume_mirrored_input}"
    )
    print(
        "Smoothing "
        f"alpha={smoothing_alpha}, min_confidence={min_smoothing_confidence}, "
        f"low_confidence_behavior={low_confidence_behavior}, decay_alpha={decay_alpha}"
    )
    print("Raw bars are gray; smoothed bars are colored. Press q to quit.")

    try:
        with (
            OpenCVCamera(camera_id=camera_id, width=width, height=height) as camera,
            HandTracker(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_path=model_path,
                assume_mirrored_input=assume_mirrored_input,
            ) as tracker,
        ):
            while True:
                camera_result = camera.read()
                if not camera_result.success or camera_result.frame is None:
                    print("WARNING: Camera read failed; waiting for the next frame.")
                    if cv2_module.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                tracking_result = tracker.process(
                    camera_result.frame,
                    timestamp=camera_result.timestamp,
                )
                raw_features = extract_hand_features(tracking_result)
                smoothed_features = smoother.update(raw_features)

                now = time.monotonic()
                elapsed = max(now - last_frame_time, 1e-9)
                last_frame_time = now
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0.0 else (0.9 * fps) + (0.1 * instant_fps)

                draw_hand_tracking(camera_result.frame, tracking_result)
                _draw_fps(cv2_module, camera_result.frame, fps)
                _draw_feature_comparison(
                    cv2_module,
                    camera_result.frame,
                    raw_features,
                    smoothed_features,
                )
                cv2_module.imshow(window_name, camera_result.frame)

                if cv2_module.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        with suppress(Exception):
            cv2_module.destroyWindow(window_name)

    print("Feature smoothing check closed cleanly.")
    return 0


def _is_cv2_error(exc: Exception) -> bool:
    return exc.__class__.__module__.split(".", maxsplit=1)[0] == "cv2"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_smoothing(
            args.camera_id,
            args.width,
            args.height,
            args.model_path,
            args.min_detection_confidence,
            args.min_tracking_confidence,
            args.assume_mirrored_input,
            args.smoothing_alpha,
            args.min_smoothing_confidence,
            args.low_confidence_behavior,
            args.decay_alpha,
        )
    except (CameraOpenError, HandTrackerError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Feature smoothing check closed cleanly.")
        return 130
    except Exception as exc:
        if _is_cv2_error(exc):
            print(
                "ERROR: OpenCV display failed. Ensure this is running in a desktop "
                f"session with GUI support. Details: {exc}",
                file=sys.stderr,
            )
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
