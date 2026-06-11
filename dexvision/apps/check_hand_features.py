"""Open a camera feed, track hand landmarks, and visualize hand features."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import suppress
from pathlib import Path

from dexvision.camera.opencv_camera import CameraOpenError, OpenCVCamera
from dexvision.features.hand_features import (
    FINGER_NAMES,
    HandFeatures,
    extract_hand_features,
)
from dexvision.perception.hand_tracker import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    HandTracker,
    HandTrackerError,
)
from dexvision.perception.visualization import draw_hand_tracking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Display live DexVision hand feature values.")
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
    return parser


def _load_cv2_for_display():
    try:
        import cv2
    except ImportError as exc:
        raise CameraOpenError(
            "OpenCV is required for hand-feature display. Install the package providing "
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


def _draw_feature_bars(cv2_module, frame, features: HandFeatures) -> None:
    rows = [
        ("Thumb", features.thumb),
        ("Index", features.index),
        ("Middle", features.middle),
        ("Ring", features.ring),
        ("Pinky", features.pinky),
    ]
    x = 16
    y = 96
    label_width = 62
    bar_width = 72
    bar_height = 12
    value_width = 42
    column_gap = 10
    row_gap = 28
    curl_x = x + label_width
    extension_x = curl_x + bar_width + value_width + column_gap
    bend_x = extension_x + bar_width + value_width + column_gap
    up_x = bend_x + bar_width + value_width + 8

    cv2_module.putText(
        frame,
        "Curl",
        (curl_x, y - 12),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame,
        "Ext",
        (extension_x, y - 12),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame,
        "Bend",
        (bend_x, y - 12),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )

    for index, (label, state) in enumerate(rows):
        row_y = y + index * row_gap
        finger_name = label.lower()
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
            x=curl_x,
            y=row_y,
            width=bar_width,
            height=bar_height,
            value=state.curl,
            color=(0, 220, 120),
        )
        _draw_bar(
            cv2_module,
            frame,
            x=extension_x,
            y=row_y,
            width=bar_width,
            height=bar_height,
            value=state.extension,
            color=(255, 190, 40),
        )
        bend = getattr(features, f"{finger_name}_bend", None)
        if bend is not None:
            _draw_bar(
                cv2_module,
                frame,
                x=bend_x,
                y=row_y,
                width=bar_width,
                height=bar_height,
                value=bend,
                color=(80, 210, 255),
            )
        value_color = (0, 255, 80) if state.is_up else (180, 180, 180)
        cv2_module.putText(
            frame,
            "UP" if state.is_up else "--",
            (up_x, row_y + bar_height),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.45,
            value_color,
            1,
            cv2_module.LINE_AA,
        )

    palm_y = y + len(rows) * row_gap + 8
    cv2_module.putText(
        frame,
        f"Pinch {features.pinch_thumb_index:.2f}   Conf {features.confidence:.2f}",
        (x, palm_y),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.5,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame,
        f"Palm roll/pitch: {features.palm_roll_proxy:+.2f} / {features.palm_pitch_proxy:+.2f}",
        (x, palm_y + 24),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.5,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )


def _draw_bar(
    cv2_module,
    frame,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    color: tuple[int, int, int],
) -> None:
    safe_value = max(0.0, min(float(value), 1.0))
    filled_width = int(round(width * safe_value))
    cv2_module.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), 1)
    if filled_width > 0:
        cv2_module.rectangle(frame, (x, y), (x + filled_width, y + height), color, -1)
    cv2_module.putText(
        frame,
        f"{safe_value:.2f}",
        (x + width + 6, y + height),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )


def run_hand_features(
    camera_id: int,
    width: int,
    height: int,
    model_path: Path | None,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    assume_mirrored_input: bool = False,
) -> int:
    cv2_module = _load_cv2_for_display()
    window_name = "DexVision Hand Feature Check"
    last_frame_time = time.monotonic()
    fps = 0.0

    print("DexVision hand feature check")
    print(
        "Opening "
        f"camera_id={camera_id}, width={width}, height={height}, "
        f"model_path={model_path or DEFAULT_HAND_LANDMARKER_MODEL}, "
        f"min_detection_confidence={min_detection_confidence}, "
        f"min_tracking_confidence={min_tracking_confidence}, "
        f"assume_mirrored_input={assume_mirrored_input}"
    )
    print(
        "Feature rows: "
        f"{', '.join(FINGER_NAMES)} with curl, extension, bend, and is_up; "
        "plus pinch_thumb_index and confidence"
    )
    print("Press q to quit.")

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
                features = extract_hand_features(tracking_result)

                now = time.monotonic()
                elapsed = max(now - last_frame_time, 1e-9)
                last_frame_time = now
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0.0 else (0.9 * fps) + (0.1 * instant_fps)

                draw_hand_tracking(camera_result.frame, tracking_result)
                _draw_fps(cv2_module, camera_result.frame, fps)
                _draw_feature_bars(cv2_module, camera_result.frame, features)
                cv2_module.imshow(window_name, camera_result.frame)

                if cv2_module.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        with suppress(Exception):
            cv2_module.destroyWindow(window_name)

    print("Hand feature check closed cleanly.")
    return 0


def _is_cv2_error(exc: Exception) -> bool:
    return exc.__class__.__module__.split(".", maxsplit=1)[0] == "cv2"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_hand_features(
            args.camera_id,
            args.width,
            args.height,
            args.model_path,
            args.min_detection_confidence,
            args.min_tracking_confidence,
            args.assume_mirrored_input,
        )
    except (CameraOpenError, HandTrackerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Hand feature check closed cleanly.")
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
