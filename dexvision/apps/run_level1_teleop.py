"""Run full-hand Level 1 camera-to-MuJoCo teleoperation."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import shlex
import sys
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dexvision.camera.opencv_camera import CameraOpenError, OpenCVCamera
from dexvision.features.hand_features import HandFeatures, extract_hand_features, no_hand_features
from dexvision.features.smoothing import FeatureSmoother, LowConfidenceBehavior
from dexvision.perception.hand_tracker import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    HandTrackingResult,
    HandTracker,
    HandTrackerError,
)
from dexvision.perception.visualization import draw_hand_tracking
from dexvision.retargeting.curl_retargeter import (
    CurlRetargeter,
    CurlRetargeterError,
    load_curl_retargeter_config,
)
from dexvision.sim.mujoco_env import MujocoEnv, MujocoError, MujocoState


DEFAULT_CONFIG = Path("configs/level1_teleop.yaml")
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_SMOOTHING_ALPHA = 0.75
DEFAULT_MIN_SMOOTHING_CONFIDENCE = 0.2
DEFAULT_LOW_CONFIDENCE_BEHAVIOR: LowConfidenceBehavior = "decay"
DEFAULT_DECAY_ALPHA = 0.15
DEFAULT_SIM_STEPS_PER_FRAME = 1
DEFAULT_PRINT_INTERVAL = 30
DEFAULT_VIEWER_SLEEP = 0.0
DEFAULT_CAMERA_WINDOW_NAME = "DexVision Level 1 Demo"
MAX_ABS_QPOS = 3.5
MAX_ABS_QVEL = 45.0


class ViewerHandle(Protocol):
    """Small protocol for MuJoCo passive viewer handles."""

    def sync(self) -> None:
        """Synchronize the viewer with the current MuJoCo state."""


class CameraOverlaySink(Protocol):
    """Receives camera overlay frames from the live teleop loop."""

    def send(self, payload: "CameraOverlayFrame") -> None:
        """Send one best-effort camera overlay frame."""

    def should_stop(self) -> bool:
        """Return whether the overlay requested the teleop loop to stop."""

    def close(self) -> None:
        """Release overlay resources."""


@dataclass(frozen=True)
class CameraOverlayFrame:
    """One camera frame plus the status needed to draw the Level 1 demo overlay."""

    frame: np.ndarray
    tracking_result: HandTrackingResult
    raw_features: HandFeatures
    smoothed_features: HandFeatures
    targets: dict[str, float]
    target_names: tuple[str, ...]
    fps: float
    simulation_time: float
    status_message: str


class CameraOverlayProcess:
    """Best-effort OpenCV camera overlay running outside the MuJoCo process."""

    def __init__(self, *, window_name: str) -> None:
        self._ctx = mp.get_context("spawn")
        self._queue = self._ctx.Queue(maxsize=1)
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_camera_overlay_worker,
            args=(self._queue, self._stop_event, window_name),
            daemon=True,
        )
        self._warned_stopped = False

    def start(self) -> "CameraOverlayProcess":
        """Start the overlay process."""

        self._process.start()
        return self

    def send(self, payload: CameraOverlayFrame) -> None:
        """Send the newest frame without blocking the teleop loop."""

        if not self._process.is_alive():
            self._warn_once_stopped()
            return

        _drop_stale_queue_items(self._queue)
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    def should_stop(self) -> bool:
        """Return whether the user closed the camera overlay."""

        if self._stop_event.is_set():
            return True
        if not self._process.is_alive():
            self._warn_once_stopped()
        return False

    def close(self) -> None:
        """Stop the overlay process."""

        self._stop_event.set()
        with suppress(Exception):
            self._queue.put_nowait(None)
        self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        with suppress(Exception):
            self._queue.close()

    def _warn_once_stopped(self) -> None:
        if self._warned_stopped:
            return
        self._warned_stopped = True
        print(
            "WARNING: Camera overlay process is not running; "
            "teleop will continue in the MuJoCo viewer."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DexVision Level 1 full-hand camera-to-MuJoCo teleoperation."
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_CAMERA_WIDTH,
        help="Requested capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_CAMERA_HEIGHT,
        help="Requested capture height.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Level 1 teleop YAML config path. Defaults to {DEFAULT_CONFIG}.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override MuJoCo model XML path. Defaults to the config model_path.",
    )
    parser.add_argument(
        "--hand-landmarker-model",
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
        default=DEFAULT_SMOOTHING_ALPHA,
        help="EMA smoothing alpha in (0.0, 1.0]; higher values respond faster.",
    )
    parser.add_argument(
        "--min-smoothing-confidence",
        type=float,
        default=DEFAULT_MIN_SMOOTHING_CONFIDENCE,
        help="Decay controls below this feature confidence.",
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
        "--sim-steps-per-frame",
        type=int,
        default=DEFAULT_SIM_STEPS_PER_FRAME,
        help="MuJoCo integration steps to run per camera frame.",
    )
    parser.add_argument(
        "--viewer-sleep",
        type=float,
        default=DEFAULT_VIEWER_SLEEP,
        help="Seconds to sleep after each MuJoCo viewer sync.",
    )
    parser.add_argument(
        "--print-interval",
        type=int,
        default=DEFAULT_PRINT_INTERVAL,
        help="Print teleop status every N processed camera frames.",
    )
    parser.add_argument(
        "--show-camera-window",
        action="store_true",
        help="Show a best-effort OpenCV camera overlay in a separate process.",
    )
    parser.add_argument(
        "--camera-window-name",
        default=DEFAULT_CAMERA_WINDOW_NAME,
        help=f"OpenCV camera overlay window title. Defaults to {DEFAULT_CAMERA_WINDOW_NAME!r}.",
    )
    return parser


def resolve_mujoco_model_path(
    raw_config: object,
    *,
    config_path: Path,
    override: Path | None,
) -> Path:
    """Resolve the MuJoCo XML path from CLI override or teleop config."""

    if override is not None:
        return override
    if not isinstance(raw_config, Mapping):
        raise CurlRetargeterError(f"Teleop config must be a mapping: {config_path}")

    raw_model_path = raw_config.get("model_path")
    if not isinstance(raw_model_path, str) or not raw_model_path:
        raise CurlRetargeterError(
            f"Teleop config must contain a non-empty model_path: {config_path}"
        )
    return Path(raw_model_path)


def configured_control_fields(retargeter: CurlRetargeter) -> tuple[str, ...]:
    """Return the hand feature fields configured for robot control."""

    return tuple(f"{finger.name}:{finger.feature}" for finger in retargeter.config.fingers)


def robot_target_names(retargeter: CurlRetargeter) -> tuple[str, ...]:
    """Return all robot target names produced by the retargeter."""

    names = [target.name for target in retargeter.config.static_targets]
    for finger in retargeter.config.fingers:
        names.extend(target.name for target in finger.targets)
    if not names:
        raise CurlRetargeterError("Teleop config must define at least one robot target.")
    return tuple(names)


def build_full_hand_targets(
    retargeter: CurlRetargeter,
    features: HandFeatures,
) -> dict[str, float]:
    """Map full-hand smoothed features to robot targets."""

    return retargeter.map(features)


def run_level1_teleop(
    *,
    camera_id: int,
    width: int,
    height: int,
    config_path: Path,
    mujoco_model_path: Path | None,
    hand_landmarker_model_path: Path | None,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    assume_mirrored_input: bool,
    smoothing_alpha: float,
    min_smoothing_confidence: float,
    low_confidence_behavior: LowConfidenceBehavior,
    decay_alpha: float,
    sim_steps_per_frame: int,
    viewer_sleep: float,
    print_interval: int,
    show_camera_window: bool,
    camera_window_name: str,
) -> int:
    """Run live full-hand camera-to-MuJoCo teleoperation."""

    _validate_run_parameters(
        sim_steps_per_frame=sim_steps_per_frame,
        viewer_sleep=viewer_sleep,
        print_interval=print_interval,
    )
    raw_config = load_curl_retargeter_config(config_path)
    model_path = resolve_mujoco_model_path(
        raw_config,
        config_path=config_path,
        override=mujoco_model_path,
    )
    retargeter = CurlRetargeter.from_mapping(raw_config)
    target_names = robot_target_names(retargeter)
    control_fields = configured_control_fields(retargeter)
    neutral_targets = build_full_hand_targets(retargeter, no_hand_features())
    smoother = FeatureSmoother(
        alpha=smoothing_alpha,
        min_confidence=min_smoothing_confidence,
        low_confidence_behavior=low_confidence_behavior,
        decay_alpha=decay_alpha,
    )

    print("DexVision Level 1 full-hand teleop")
    print(f"Camera: id={camera_id}, width={width}, height={height}")
    hand_tracker_model = hand_landmarker_model_path or DEFAULT_HAND_LANDMARKER_MODEL
    print(f"Hand tracker model: {hand_tracker_model}")
    print(f"Teleop config: {config_path}")
    print(f"MuJoCo model: {model_path}")
    print(f"Control fields: {', '.join(control_fields)}")
    print(f"Robot targets: {', '.join(target_names)}")
    print(
        "Long fingers use Level 1.3B bend controls "
        "(bend = 1.0 - smoothed extension); thumb uses configured thumb control."
    )
    print(
        "Tracking loss behavior: "
        f"{low_confidence_behavior} below confidence {min_smoothing_confidence:.2f}."
    )
    print(f"Camera overlay window: {'on' if show_camera_window else 'off'}")
    print("Close the MuJoCo viewer or press Ctrl-C in the terminal to quit.")
    if show_camera_window:
        print("Camera overlay shows landmarks, feature bars, FPS, and tracking status.")
        print("Press q in the camera overlay to quit.")
    _ensure_viewer_can_launch(
        camera_id=camera_id,
        width=width,
        height=height,
        config_path=config_path,
        model_path=mujoco_model_path,
        hand_landmarker_model_path=hand_landmarker_model_path,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        assume_mirrored_input=assume_mirrored_input,
        smoothing_alpha=smoothing_alpha,
        min_smoothing_confidence=min_smoothing_confidence,
        low_confidence_behavior=low_confidence_behavior,
        decay_alpha=decay_alpha,
        sim_steps_per_frame=sim_steps_per_frame,
        viewer_sleep=viewer_sleep,
        print_interval=print_interval,
        show_camera_window=show_camera_window,
        camera_window_name=camera_window_name,
    )

    camera_overlay = (
        CameraOverlayProcess(window_name=camera_window_name).start()
        if show_camera_window
        else None
    )
    try:
        with (
            OpenCVCamera(camera_id=camera_id, width=width, height=height) as camera,
            HandTracker(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_path=hand_landmarker_model_path,
                assume_mirrored_input=assume_mirrored_input,
            ) as tracker,
            MujocoEnv(model_path) as env,
        ):
            env.reset()
            env.set_joint_targets(neutral_targets)
            env.step(n_steps=max(1, sim_steps_per_frame))
            _run_with_viewer(
                env=env,
                camera=camera,
                tracker=tracker,
                smoother=smoother,
                retargeter=retargeter,
                target_names=target_names,
                camera_overlay=camera_overlay,
                sim_steps_per_frame=sim_steps_per_frame,
                viewer_sleep=viewer_sleep,
                print_interval=print_interval,
            )
    finally:
        if camera_overlay is not None:
            camera_overlay.close()

    print("Level 1 full-hand teleop closed cleanly.")
    return 0


def _run_with_viewer(
    *,
    env: MujocoEnv,
    camera: OpenCVCamera,
    tracker: HandTracker,
    smoother: FeatureSmoother,
    retargeter: CurlRetargeter,
    target_names: tuple[str, ...],
    camera_overlay: CameraOverlaySink | None,
    sim_steps_per_frame: int,
    viewer_sleep: float,
    print_interval: int,
) -> None:
    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - MuJoCo import tested elsewhere.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    try:
        with viewer.launch_passive(env.model, env.data) as viewer_handle:
            _run_loop(
                env=env,
                camera=camera,
                tracker=tracker,
                smoother=smoother,
                retargeter=retargeter,
                target_names=target_names,
                camera_overlay=camera_overlay,
                sim_steps_per_frame=sim_steps_per_frame,
                viewer_handle=viewer_handle,
                viewer_sleep=viewer_sleep,
                print_interval=print_interval,
            )
    except Exception as exc:  # pragma: no cover - requires desktop GUI to exercise.
        raise MujocoError(f"MuJoCo viewer failed to open or run: {exc}") from exc


def _run_loop(
    *,
    env: MujocoEnv,
    camera: OpenCVCamera,
    tracker: HandTracker,
    smoother: FeatureSmoother,
    retargeter: CurlRetargeter,
    target_names: tuple[str, ...],
    camera_overlay: CameraOverlaySink | None,
    sim_steps_per_frame: int,
    viewer_handle: ViewerHandle,
    viewer_sleep: float,
    print_interval: int,
) -> None:
    last_frame_time = time.monotonic()
    fps = 0.0
    frame_index = 0

    while True:
        camera_result = camera.read()
        if not camera_result.success or camera_result.frame is None:
            print("WARNING: Camera read failed; waiting for the next frame.")
            if _viewer_was_closed(viewer_handle):
                break
            if camera_overlay is not None and camera_overlay.should_stop():
                break
            continue

        tracking_result = tracker.process(
            camera_result.frame,
            timestamp=camera_result.timestamp,
        )
        raw_features = extract_hand_features(tracking_result)
        smoothed_features = smoother.update(raw_features)
        targets = build_full_hand_targets(retargeter, smoothed_features)
        env.set_joint_targets(targets)
        state = env.step(n_steps=sim_steps_per_frame)
        _raise_if_unstable(state)

        frame_index += 1
        now = time.monotonic()
        elapsed = max(now - last_frame_time, 1e-9)
        last_frame_time = now
        instant_fps = 1.0 / elapsed
        fps = instant_fps if fps == 0.0 else (0.9 * fps) + (0.1 * instant_fps)
        status_message = _format_tracking_status(
            detected=tracking_result.detected,
            confidence=raw_features.confidence,
            min_confidence=smoother.config.min_confidence,
            low_confidence_behavior=smoother.config.low_confidence_behavior,
        )

        if camera_overlay is not None:
            camera_overlay.send(
                CameraOverlayFrame(
                    frame=camera_result.frame.copy(),
                    tracking_result=tracking_result,
                    raw_features=raw_features,
                    smoothed_features=smoothed_features,
                    targets=dict(targets),
                    target_names=target_names,
                    fps=fps,
                    simulation_time=state.time,
                    status_message=status_message,
                )
            )

        viewer_handle.sync()
        if viewer_sleep > 0.0:
            time.sleep(viewer_sleep)

        if frame_index == 1 or frame_index % print_interval == 0:
            print(
                f"frame={frame_index:05d} "
                f"detected={tracking_result.detected} "
                f"confidence={raw_features.confidence:.2f} "
                f"status={status_message!r} "
                f"fps={fps:.1f} "
                f"{_format_control_summary(smoothed_features)} "
                f"targets={_format_target_summary(targets, target_names)} "
                f"t={state.time:.3f}s"
            )

        if _viewer_was_closed(viewer_handle):
            break
        if camera_overlay is not None and camera_overlay.should_stop():
            break


def _camera_overlay_worker(frame_queue: Any, stop_event: Any, window_name: str) -> None:
    cv2_module = None
    draw_overlay = True
    warned_overlay_failure = False
    try:
        cv2_module = _load_cv2_for_display()
        while not stop_event.is_set():
            try:
                payload = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload is None:
                break

            draw_overlay, warned_overlay_failure = _show_camera_overlay_frame_safely(
                cv2_module,
                window_name,
                payload,
                draw_overlay=draw_overlay,
                warned_overlay_failure=warned_overlay_failure,
            )
            if cv2_module.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
    except Exception as exc:  # pragma: no cover - depends on desktop GUI backend.
        print(
            "WARNING: OpenCV camera overlay process stopped; "
            f"teleop can continue without the overlay. Details: {exc}",
            file=sys.stderr,
        )
    finally:
        if cv2_module is not None:
            with suppress(Exception):
                cv2_module.destroyWindow(window_name)


def _show_camera_overlay_frame_safely(
    cv2_module: Any,
    window_name: str,
    payload: CameraOverlayFrame,
    *,
    draw_overlay: bool,
    warned_overlay_failure: bool,
) -> tuple[bool, bool]:
    """Draw one camera overlay frame, falling back to raw frames if drawing fails."""

    try:
        _show_camera_overlay_frame(
            cv2_module,
            window_name,
            payload,
            draw_overlay=draw_overlay,
        )
        return draw_overlay, warned_overlay_failure
    except Exception as exc:  # pragma: no cover - exact OpenCV errors vary by backend.
        if not warned_overlay_failure:
            print(
                "WARNING: Camera overlay annotations failed and will be disabled; "
                f"raw camera frames will continue. Details: {exc}",
                file=sys.stderr,
            )
        _show_camera_overlay_frame(
            cv2_module,
            window_name,
            payload,
            draw_overlay=False,
        )
        return False, True


def _show_camera_overlay_frame(
    cv2_module: Any,
    window_name: str,
    payload: CameraOverlayFrame,
    *,
    draw_overlay: bool = True,
) -> None:
    frame = payload.frame
    if draw_overlay:
        draw_hand_tracking(frame, payload.tracking_result)
        _draw_level1_demo_overlay(cv2_module, frame, payload)
    cv2_module.imshow(window_name, frame)


def _drop_stale_queue_items(frame_queue: Any) -> None:
    while True:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            return


def _load_cv2_for_display():
    try:
        import cv2
    except ImportError as exc:
        raise CameraOpenError(
            "OpenCV is required for the Level 1 demo display. Install the package "
            f"providing 'cv2' ({exc})."
        ) from exc
    return cv2


def _draw_level1_demo_overlay(
    cv2_module: object,
    frame: np.ndarray,
    payload: CameraOverlayFrame,
) -> None:
    x = 16
    y = 62
    bar_width = 170
    bar_height = 13

    cv2_module.rectangle(frame, (8, 44), (428, 268), (24, 24, 24), -1)
    cv2_module.rectangle(frame, (8, 44), (428, 268), (80, 80, 80), 1)
    cv2_module.putText(
        frame,
        f"FPS {payload.fps:5.1f}   sim {payload.simulation_time:6.3f}s",
        (x, y),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame,
        f"Confidence {payload.raw_features.confidence:.2f}",
        (x, y + 24),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 235, 235),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame,
        payload.status_message,
        (x, y + 48),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.5,
        _status_color(payload.status_message),
        2,
        cv2_module.LINE_AA,
    )

    controls = (
        ("Thumb curl", payload.smoothed_features.thumb_curl, payload.raw_features.thumb_curl),
        ("Index bend", payload.smoothed_features.index_bend, payload.raw_features.index_bend),
        ("Middle bend", payload.smoothed_features.middle_bend, payload.raw_features.middle_bend),
        ("Ring bend", payload.smoothed_features.ring_bend, payload.raw_features.ring_bend),
        ("Pinky bend", payload.smoothed_features.pinky_bend, payload.raw_features.pinky_bend),
    )
    for index, (label, smoothed_value, raw_value) in enumerate(controls):
        _draw_labeled_bar(
            cv2_module,
            frame,
            label=label,
            value=smoothed_value,
            raw_marker=raw_value,
            x=x,
            y=y + 74 + (index * 24),
            width=bar_width,
            height=bar_height,
            color=(0, 210, 120),
        )

    cv2_module.putText(
        frame,
        "raw marker | smoothed fill",
        (x, y + 204),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 230, 230),
        1,
        cv2_module.LINE_AA,
    )


def _draw_labeled_bar(
    cv2_module: object,
    frame: np.ndarray,
    *,
    label: str,
    value: float,
    raw_marker: float,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    safe_value = _clip_overlay_value(value)
    safe_marker = _clip_overlay_value(raw_marker)
    label_width = 112
    bar_x = x + label_width
    filled_width = int(round(width * safe_value))
    marker_x = bar_x + int(round(width * safe_marker))

    cv2_module.putText(
        frame,
        label,
        (x, y + height),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.43,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.rectangle(
        frame,
        (bar_x, y),
        (bar_x + width, y + height),
        (72, 72, 72),
        1,
    )
    if filled_width > 0:
        cv2_module.rectangle(
            frame,
            (bar_x, y),
            (bar_x + filled_width, y + height),
            color,
            -1,
        )
    cv2_module.line(
        frame,
        (marker_x, y - 2),
        (marker_x, y + height + 2),
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame,
        f"{safe_value:.2f}",
        (bar_x + width + 10, y + height),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )


def _clip_overlay_value(value: float) -> float:
    return float(np.clip(value if np.isfinite(value) else 0.0, 0.0, 1.0))


def _format_tracking_status(
    *,
    detected: bool,
    confidence: float,
    min_confidence: float,
    low_confidence_behavior: LowConfidenceBehavior,
) -> str:
    safe_action = (
        "controls decaying to open"
        if low_confidence_behavior == "decay"
        else "controls holding last pose"
    )
    if not detected:
        return f"TRACKING LOST - {safe_action}"
    if confidence < min_confidence:
        return f"LOW CONFIDENCE {confidence:.2f} - {safe_action}"
    return f"TRACKING OK {confidence:.2f}"


def _status_color(status_message: str) -> tuple[int, int, int]:
    if status_message.startswith("TRACKING OK"):
        return (0, 220, 120)
    if status_message.startswith("LOW CONFIDENCE"):
        return (0, 190, 255)
    return (0, 120, 255)


def _format_control_summary(features: HandFeatures) -> str:
    return (
        "controls="
        f"Tcurl:{features.thumb_curl:.2f},"
        f"Ibend:{features.index_bend:.2f},"
        f"Mbend:{features.middle_bend:.2f},"
        f"Rbend:{features.ring_bend:.2f},"
        f"Pbend:{features.pinky_bend:.2f}"
    )


def _format_target_summary(
    targets: Mapping[str, float],
    target_names: tuple[str, ...],
    *,
    max_items: int = 10,
) -> str:
    names = tuple(name for name in target_names if name in targets)
    shown = names[:max_items]
    summary = ", ".join(f"{name}={targets[name]:.2f}" for name in shown)
    remaining = len(names) - len(shown)
    if remaining > 0:
        return f"{summary}, ...({remaining} more)"
    return summary


def _viewer_was_closed(viewer_handle: ViewerHandle) -> bool:
    is_running = getattr(viewer_handle, "is_running", None)
    if not callable(is_running):
        return False
    return not bool(is_running())


def _ensure_viewer_can_launch(
    *,
    camera_id: int,
    width: int,
    height: int,
    config_path: Path,
    model_path: Path | None,
    hand_landmarker_model_path: Path | None,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    assume_mirrored_input: bool,
    smoothing_alpha: float,
    min_smoothing_confidence: float,
    low_confidence_behavior: LowConfidenceBehavior,
    decay_alpha: float,
    sim_steps_per_frame: int,
    viewer_sleep: float,
    print_interval: int,
    show_camera_window: bool,
    camera_window_name: str,
) -> None:
    if sys.platform != "darwin":
        return

    try:
        from mujoco import viewer
    except ImportError as exc:  # pragma: no cover - MuJoCo import tested elsewhere.
        raise MujocoError(f"MuJoCo viewer support is unavailable: {exc}") from exc

    mjpython_base = getattr(viewer, "_MjPythonBase", None)
    mjpython_dispatcher = getattr(viewer, "_MJPYTHON", None)
    if mjpython_base is not None and isinstance(mjpython_dispatcher, mjpython_base):
        return

    command = _format_mjpython_command(
        camera_id=camera_id,
        width=width,
        height=height,
        config_path=config_path,
        model_path=model_path,
        hand_landmarker_model_path=hand_landmarker_model_path,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        assume_mirrored_input=assume_mirrored_input,
        smoothing_alpha=smoothing_alpha,
        min_smoothing_confidence=min_smoothing_confidence,
        low_confidence_behavior=low_confidence_behavior,
        decay_alpha=decay_alpha,
        sim_steps_per_frame=sim_steps_per_frame,
        viewer_sleep=viewer_sleep,
        print_interval=print_interval,
        show_camera_window=show_camera_window,
        camera_window_name=camera_window_name,
    )
    raise MujocoError(
        "MuJoCo viewer on macOS requires the mjpython launcher.\n"
        "Run this from a regular macOS Terminal or iTerm session:\n"
        f"  {command}"
    )


def _format_mjpython_command(
    *,
    camera_id: int,
    width: int,
    height: int,
    config_path: Path,
    model_path: Path | None,
    hand_landmarker_model_path: Path | None,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    assume_mirrored_input: bool,
    smoothing_alpha: float,
    min_smoothing_confidence: float,
    low_confidence_behavior: LowConfidenceBehavior,
    decay_alpha: float,
    sim_steps_per_frame: int,
    viewer_sleep: float,
    print_interval: int,
    show_camera_window: bool,
    camera_window_name: str,
) -> str:
    command = [
        "mjpython",
        "-m",
        "dexvision.apps.run_level1_teleop",
        "--camera-id",
        str(camera_id),
    ]
    if width != DEFAULT_CAMERA_WIDTH:
        command.extend(["--width", str(width)])
    if height != DEFAULT_CAMERA_HEIGHT:
        command.extend(["--height", str(height)])
    if config_path != DEFAULT_CONFIG:
        command.extend(["--config", str(config_path)])
    if model_path is not None:
        command.extend(["--model", str(model_path)])
    if hand_landmarker_model_path is not None:
        command.extend(["--hand-landmarker-model", str(hand_landmarker_model_path)])
    if min_detection_confidence != 0.5:
        command.extend(["--min-detection-confidence", str(min_detection_confidence)])
    if min_tracking_confidence != 0.5:
        command.extend(["--min-tracking-confidence", str(min_tracking_confidence)])
    if assume_mirrored_input:
        command.append("--assume-mirrored-input")
    if smoothing_alpha != DEFAULT_SMOOTHING_ALPHA:
        command.extend(["--smoothing-alpha", str(smoothing_alpha)])
    if min_smoothing_confidence != DEFAULT_MIN_SMOOTHING_CONFIDENCE:
        command.extend(["--min-smoothing-confidence", str(min_smoothing_confidence)])
    if low_confidence_behavior != DEFAULT_LOW_CONFIDENCE_BEHAVIOR:
        command.extend(["--low-confidence-behavior", low_confidence_behavior])
    if decay_alpha != DEFAULT_DECAY_ALPHA:
        command.extend(["--decay-alpha", str(decay_alpha)])
    if sim_steps_per_frame != DEFAULT_SIM_STEPS_PER_FRAME:
        command.extend(["--sim-steps-per-frame", str(sim_steps_per_frame)])
    if viewer_sleep != DEFAULT_VIEWER_SLEEP:
        command.extend(["--viewer-sleep", str(viewer_sleep)])
    if print_interval != DEFAULT_PRINT_INTERVAL:
        command.extend(["--print-interval", str(print_interval)])
    if show_camera_window:
        command.append("--show-camera-window")
    if camera_window_name != DEFAULT_CAMERA_WINDOW_NAME:
        command.extend(["--camera-window-name", camera_window_name])
    return " ".join(shlex.quote(part) for part in command)


def _raise_if_unstable(state: MujocoState) -> None:
    if not np.all(np.isfinite(state.qpos)) or not np.all(np.isfinite(state.qvel)):
        raise MujocoError("Simulation became unstable: non-finite qpos or qvel.")
    max_abs_qpos = _max_abs(state.qpos)
    max_abs_qvel = _max_abs(state.qvel)
    if max_abs_qpos > MAX_ABS_QPOS or max_abs_qvel > MAX_ABS_QVEL:
        raise MujocoError(
            "Simulation became unstable: "
            f"max_abs_qpos={max_abs_qpos:.6f}, max_abs_qvel={max_abs_qvel:.6f}."
        )


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _validate_run_parameters(
    *,
    sim_steps_per_frame: int,
    viewer_sleep: float,
    print_interval: int,
) -> None:
    if sim_steps_per_frame <= 0:
        raise ValueError("sim_steps_per_frame must be a positive integer.")
    if viewer_sleep < 0.0:
        raise ValueError("viewer_sleep must be non-negative.")
    if print_interval <= 0:
        raise ValueError("print_interval must be a positive integer.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_level1_teleop(
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            config_path=args.config,
            mujoco_model_path=args.model,
            hand_landmarker_model_path=args.hand_landmarker_model,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            assume_mirrored_input=args.assume_mirrored_input,
            smoothing_alpha=args.smoothing_alpha,
            min_smoothing_confidence=args.min_smoothing_confidence,
            low_confidence_behavior=args.low_confidence_behavior,
            decay_alpha=args.decay_alpha,
            sim_steps_per_frame=args.sim_steps_per_frame,
            viewer_sleep=args.viewer_sleep,
            print_interval=args.print_interval,
            show_camera_window=args.show_camera_window,
            camera_window_name=args.camera_window_name,
        )
    except (
        CameraOpenError,
        CurlRetargeterError,
        HandTrackerError,
        MujocoError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Level 1 full-hand teleop closed cleanly.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
