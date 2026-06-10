"""Use one real index finger to control one MuJoCo robot index finger."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import shlex
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dexvision.camera.opencv_camera import CameraOpenError, OpenCVCamera
from dexvision.features.hand_features import (
    HandFeatures,
    extract_hand_features,
    no_hand_features,
)
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
DEFAULT_INDEX_OPEN_CURL = 0.20
DEFAULT_INDEX_CLOSED_CURL = 0.80
DEFAULT_INDEX_RESPONSE_GAMMA = 1.30
INDEX_CURL_FIELD = "index_curl"
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
    """One camera frame plus the status needed to draw the teleop overlay."""

    frame: np.ndarray
    tracking_result: HandTrackingResult
    raw_features: HandFeatures
    smoothed_features: HandFeatures
    targets: dict[str, float]
    index_targets: tuple[str, ...]
    fps: float
    simulation_time: float


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
        description=(
            "Control only the MuJoCo robot index finger from a real index finger."
        )
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
        "--smoothing-alpha",
        type=float,
        default=DEFAULT_SMOOTHING_ALPHA,
        help="EMA smoothing alpha in (0.0, 1.0]; higher values respond faster.",
    )
    parser.add_argument(
        "--min-smoothing-confidence",
        type=float,
        default=DEFAULT_MIN_SMOOTHING_CONFIDENCE,
        help="Decay index control below this feature confidence.",
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
        "--index-open-curl",
        type=float,
        default=DEFAULT_INDEX_OPEN_CURL,
        help=(
            "Measured smoothed index_curl for an open index finger. Values at "
            "or below this map to the robot index open target."
        ),
    )
    parser.add_argument(
        "--index-closed-curl",
        type=float,
        default=DEFAULT_INDEX_CLOSED_CURL,
        help=(
            "Measured smoothed index_curl for a fully curled index finger. "
            "Values at or above this map to the robot index closed target."
        ),
    )
    parser.add_argument(
        "--index-response-gamma",
        type=float,
        default=DEFAULT_INDEX_RESPONSE_GAMMA,
        help=(
            "Curve applied after index curl normalization. Values above 1.0 "
            "soften small open-hand curl without changing the endpoints."
        ),
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
        help=(
            "Show a best-effort OpenCV camera overlay in a separate process."
        ),
    )
    return parser


def normalize_index_curl(
    index_curl: float,
    *,
    open_curl: float = 0.0,
    closed_curl: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Normalize measured index curl into a robot control curl in [0, 1]."""

    if not np.isfinite(index_curl):
        return 0.0
    if closed_curl <= open_curl:
        raise ValueError("closed_curl must be greater than open_curl.")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")

    normalized = (float(index_curl) - open_curl) / (closed_curl - open_curl)
    clipped = float(np.clip(normalized, 0.0, 1.0))
    return float(clipped**gamma)


def index_only_features(
    features: HandFeatures,
    *,
    index_open_curl: float = 0.0,
    index_closed_curl: float = 1.0,
    index_response_gamma: float = 1.0,
) -> HandFeatures:
    """Return features with only the index curl connected to robot control."""

    return HandFeatures(
        thumb_curl=0.0,
        index_curl=normalize_index_curl(
            features.index_curl,
            open_curl=index_open_curl,
            closed_curl=index_closed_curl,
            gamma=index_response_gamma,
        ),
        middle_curl=0.0,
        ring_curl=0.0,
        pinky_curl=0.0,
        pinch_thumb_index=0.0,
        palm_roll_proxy=0.0,
        palm_pitch_proxy=0.0,
        confidence=features.confidence,
    )


def build_index_finger_targets(
    retargeter: CurlRetargeter,
    features: HandFeatures,
    *,
    index_open_curl: float = 0.0,
    index_closed_curl: float = 1.0,
    index_response_gamma: float = 1.0,
) -> dict[str, float]:
    """Map only human ``index_curl`` to robot targets.

    All non-index finger curls are forced open before retargeting so their
    configured robot actuators stay at neutral open-hand targets.
    """

    return retargeter.map(
        index_only_features(
            features,
            index_open_curl=index_open_curl,
            index_closed_curl=index_closed_curl,
            index_response_gamma=index_response_gamma,
        )
    )


def index_target_names(retargeter: CurlRetargeter) -> tuple[str, ...]:
    """Return robot target names driven by the configured index curl mapping."""

    names: list[str] = []
    for finger in retargeter.config.fingers:
        if finger.feature == INDEX_CURL_FIELD:
            names.extend(target.name for target in finger.targets)

    if not names:
        raise CurlRetargeterError(
            f"Teleop config must include a finger mapping for {INDEX_CURL_FIELD!r}."
        )
    return tuple(names)


def resolve_mujoco_model_path(
    raw_config: object,
    *,
    config_path: Path,
    override: Path | None,
) -> Path:
    """Resolve the MuJoCo XML path from CLI override or teleop config."""

    if override is not None:
        return override
    if not isinstance(raw_config, dict):
        raise CurlRetargeterError(f"Teleop config must be a mapping: {config_path}")

    raw_model_path = raw_config.get("model_path")
    if not isinstance(raw_model_path, str) or not raw_model_path:
        raise CurlRetargeterError(
            f"Teleop config must contain a non-empty model_path: {config_path}"
        )
    return Path(raw_model_path)


def run_one_finger_teleop(
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
    index_open_curl: float,
    index_closed_curl: float,
    index_response_gamma: float,
    sim_steps_per_frame: int,
    viewer_sleep: float,
    print_interval: int,
    show_camera_window: bool,
) -> int:
    """Run live one-finger camera-to-MuJoCo teleoperation."""

    _validate_run_parameters(
        sim_steps_per_frame=sim_steps_per_frame,
        viewer_sleep=viewer_sleep,
        print_interval=print_interval,
        index_open_curl=index_open_curl,
        index_closed_curl=index_closed_curl,
        index_response_gamma=index_response_gamma,
    )
    raw_config = load_curl_retargeter_config(config_path)
    model_path = resolve_mujoco_model_path(
        raw_config,
        config_path=config_path,
        override=mujoco_model_path,
    )
    retargeter = CurlRetargeter.from_mapping(raw_config)
    index_targets = index_target_names(retargeter)
    neutral_targets = build_index_finger_targets(
        retargeter,
        no_hand_features(),
        index_open_curl=index_open_curl,
        index_closed_curl=index_closed_curl,
        index_response_gamma=index_response_gamma,
    )
    smoother = FeatureSmoother(
        alpha=smoothing_alpha,
        min_confidence=min_smoothing_confidence,
        low_confidence_behavior=low_confidence_behavior,
        decay_alpha=decay_alpha,
    )

    print("DexVision one-finger teleop check")
    print(f"Camera: id={camera_id}, width={width}, height={height}")
    hand_tracker_model = hand_landmarker_model_path or DEFAULT_HAND_LANDMARKER_MODEL
    print(f"Hand tracker model: {hand_tracker_model}")
    print(f"Teleop config: {config_path}")
    print(f"MuJoCo model: {model_path}")
    print(f"Index robot targets: {', '.join(index_targets)}")
    print("Only human index_curl is connected; other robot fingers stay neutral/open.")
    print(
        "Index response: "
        f"open={index_open_curl:.2f}, closed={index_closed_curl:.2f}, "
        f"gamma={index_response_gamma:.2f}, smoothing_alpha={smoothing_alpha:.2f}."
    )
    print(f"Camera overlay window: {'on' if show_camera_window else 'off'}")
    print(
        "Tracking loss behavior: "
        f"{low_confidence_behavior} below confidence {min_smoothing_confidence:.2f}."
    )
    print("Close the MuJoCo viewer or press Ctrl-C in the terminal to quit.")
    if show_camera_window:
        print("Camera overlay runs in a separate window; press q there to quit.")
    _ensure_viewer_can_launch()

    window_name = "DexVision One-Finger Teleop"
    camera_overlay = (
        CameraOverlayProcess(window_name=window_name).start()
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
                index_targets=index_targets,
                camera_overlay=camera_overlay,
                window_name=window_name,
                index_open_curl=index_open_curl,
                index_closed_curl=index_closed_curl,
                index_response_gamma=index_response_gamma,
                sim_steps_per_frame=sim_steps_per_frame,
                viewer_sleep=viewer_sleep,
                print_interval=print_interval,
            )
    finally:
        if camera_overlay is not None:
            camera_overlay.close()

    print("One-finger teleop check closed cleanly.")
    return 0


def _run_with_viewer(
    *,
    env: MujocoEnv,
    camera: OpenCVCamera,
    tracker: HandTracker,
    smoother: FeatureSmoother,
    retargeter: CurlRetargeter,
    index_targets: tuple[str, ...],
    camera_overlay: CameraOverlaySink | None,
    window_name: str,
    index_open_curl: float,
    index_closed_curl: float,
    index_response_gamma: float,
    sim_steps_per_frame: int,
    viewer_sleep: float,
    print_interval: int,
) -> None:
    _ensure_viewer_can_launch()
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
                index_targets=index_targets,
                camera_overlay=camera_overlay,
                window_name=window_name,
                index_open_curl=index_open_curl,
                index_closed_curl=index_closed_curl,
                index_response_gamma=index_response_gamma,
                sim_steps_per_frame=sim_steps_per_frame,
                viewer_handle=viewer_handle,
                viewer_sleep=viewer_sleep,
                print_interval=print_interval,
            )
    except Exception as exc:  # pragma: no cover - requires desktop GUI to exercise.
        if _is_cv2_error(exc):
            raise
        raise MujocoError(f"MuJoCo viewer failed to open or run: {exc}") from exc


def _run_loop(
    *,
    env: MujocoEnv,
    camera: OpenCVCamera,
    tracker: HandTracker,
    smoother: FeatureSmoother,
    retargeter: CurlRetargeter,
    index_targets: tuple[str, ...],
    camera_overlay: CameraOverlaySink | None,
    window_name: str,
    index_open_curl: float,
    index_closed_curl: float,
    index_response_gamma: float,
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
        effective_features = index_only_features(
            smoothed_features,
            index_open_curl=index_open_curl,
            index_closed_curl=index_closed_curl,
            index_response_gamma=index_response_gamma,
        )
        targets = retargeter.map(effective_features)
        env.set_joint_targets(targets)
        state = env.step(n_steps=sim_steps_per_frame)
        _raise_if_unstable(state)

        frame_index += 1
        now = time.monotonic()
        elapsed = max(now - last_frame_time, 1e-9)
        last_frame_time = now
        instant_fps = 1.0 / elapsed
        fps = instant_fps if fps == 0.0 else (0.9 * fps) + (0.1 * instant_fps)

        if camera_overlay is not None:
            camera_overlay.send(
                CameraOverlayFrame(
                    frame=camera_result.frame.copy(),
                    tracking_result=tracking_result,
                    raw_features=raw_features,
                    smoothed_features=smoothed_features,
                    targets=dict(targets),
                    index_targets=index_targets,
                    fps=fps,
                    simulation_time=state.time,
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
                f"{_format_finger_curl_summary('raw', raw_features)} "
                f"{_format_finger_curl_summary('smooth', smoothed_features)} "
                f"index_effective={effective_features.index_curl:.2f} "
                f"targets={_format_target_summary(targets, index_targets)} "
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
        _draw_index_landmarks(cv2_module, frame, payload.tracking_result)
        _draw_status_overlay(
            cv2_module,
            frame,
            fps=payload.fps,
            raw_features=payload.raw_features,
            smoothed_features=payload.smoothed_features,
            targets=payload.targets,
            index_targets=payload.index_targets,
            simulation_time=payload.simulation_time,
        )
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
            "OpenCV is required for one-finger teleop display. Install the package "
            f"providing 'cv2' ({exc})."
        ) from exc
    return cv2


def _draw_status_overlay(
    cv2_module: object,
    frame: np.ndarray,
    *,
    fps: float,
    raw_features: HandFeatures,
    smoothed_features: HandFeatures,
    targets: dict[str, float],
    index_targets: tuple[str, ...],
    simulation_time: float,
) -> None:
    x = 16
    y = 64
    bar_width = 180
    bar_height = 14

    cv2_module.putText(
        frame,
        f"FPS: {fps:5.1f}  sim: {simulation_time:6.3f}s",
        (x, y),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2_module.LINE_AA,
    )
    _draw_labeled_bar(
        cv2_module,
        frame,
        label="Index raw",
        value=raw_features.index_curl,
        x=x,
        y=y + 28,
        width=bar_width,
        height=bar_height,
        color=(160, 160, 160),
    )
    _draw_labeled_bar(
        cv2_module,
        frame,
        label="Index smooth",
        value=smoothed_features.index_curl,
        x=x,
        y=y + 54,
        width=bar_width,
        height=bar_height,
        color=(0, 220, 120),
    )
    cv2_module.putText(
        frame,
        _format_target_summary(targets, index_targets),
        (x, y + 96),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame,
        _format_finger_curl_summary("Raw", raw_features),
        (x, y + 122),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.42,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame,
        _format_finger_curl_summary("Smooth", smoothed_features),
        (x, y + 146),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.42,
        (120, 255, 80),
        1,
        cv2_module.LINE_AA,
    )


def _draw_index_landmarks(
    cv2_module: object,
    frame: np.ndarray,
    tracking_result: HandTrackingResult,
) -> None:
    if not tracking_result.detected or tracking_result.image_landmarks is None:
        return

    points = _normalized_landmarks_to_pixels(
        tracking_result.image_landmarks,
        width=frame.shape[1],
        height=frame.shape[0],
    )
    index_points = [points[index] for index in (5, 6, 7, 8)]
    for start, end in zip(index_points, index_points[1:]):
        cv2_module.line(frame, start, end, (0, 255, 255), 4, cv2_module.LINE_AA)
    for point in index_points:
        cv2_module.circle(frame, point, 7, (0, 255, 255), -1, cv2_module.LINE_AA)

    tip_x, tip_y = index_points[-1]
    cv2_module.putText(
        frame,
        "INDEX",
        (tip_x + 8, max(20, tip_y - 8)),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
        cv2_module.LINE_AA,
    )


def _normalized_landmarks_to_pixels(
    image_landmarks: np.ndarray,
    *,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    if image_landmarks.shape != (21, 3):
        raise ValueError(
            f"image_landmarks must have shape [21, 3], got {image_landmarks.shape}."
        )

    points: list[tuple[int, int]] = []
    for x, y, _z in image_landmarks:
        safe_x = 0.0 if not np.isfinite(x) else float(np.clip(x, 0.0, 1.0))
        safe_y = 0.0 if not np.isfinite(y) else float(np.clip(y, 0.0, 1.0))
        pixel_x = int(round(safe_x * (width - 1)))
        pixel_y = int(round(safe_y * (height - 1)))
        points.append((pixel_x, pixel_y))
    return points


def _draw_labeled_bar(
    cv2_module: object,
    frame: np.ndarray,
    *,
    label: str,
    value: float,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    safe_value = float(np.clip(value if np.isfinite(value) else 0.0, 0.0, 1.0))
    label_width = 112
    filled_width = int(round(width * safe_value))
    cv2_module.putText(
        frame,
        label,
        (x, y + height),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )
    bar_x = x + label_width
    cv2_module.rectangle(
        frame,
        (bar_x, y),
        (bar_x + width, y + height),
        (60, 60, 60),
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
    cv2_module.putText(
        frame,
        f"{safe_value:.2f}",
        (bar_x + width + 10, y + height),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2_module.LINE_AA,
    )


def _format_target_summary(
    targets: dict[str, float],
    target_names: tuple[str, ...],
) -> str:
    return ", ".join(f"{name}={targets[name]:.2f}" for name in target_names)


def _format_finger_curl_summary(label: str, features: HandFeatures) -> str:
    return (
        f"{label}_curls="
        f"T:{features.thumb_curl:.2f},"
        f"I:{features.index_curl:.2f},"
        f"M:{features.middle_curl:.2f},"
        f"R:{features.ring_curl:.2f},"
        f"P:{features.pinky_curl:.2f}"
    )


def _viewer_was_closed(viewer_handle: ViewerHandle) -> bool:
    is_running = getattr(viewer_handle, "is_running", None)
    if not callable(is_running):
        return False
    return not bool(is_running())


def _ensure_viewer_can_launch() -> None:
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

    raise MujocoError(
        "MuJoCo viewer on macOS requires the mjpython launcher.\n"
        "Run this from a regular macOS Terminal or iTerm session:\n"
        f"  {_format_mjpython_command()}"
    )


def _format_mjpython_command() -> str:
    command = [
        "mjpython",
        "-m",
        "dexvision.apps.check_one_finger_teleop",
        "--camera-id",
        "0",
    ]
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
    index_open_curl: float,
    index_closed_curl: float,
    index_response_gamma: float,
) -> None:
    if sim_steps_per_frame <= 0:
        raise ValueError("sim_steps_per_frame must be a positive integer.")
    if viewer_sleep < 0.0:
        raise ValueError("viewer_sleep must be non-negative.")
    if print_interval <= 0:
        raise ValueError("print_interval must be a positive integer.")
    if not 0.0 <= index_open_curl < index_closed_curl <= 1.0:
        raise ValueError(
            "index_open_curl and index_closed_curl must satisfy "
            "0.0 <= open < closed <= 1.0."
        )
    if index_response_gamma <= 0.0:
        raise ValueError("index_response_gamma must be positive.")


def _is_cv2_error(exc: Exception) -> bool:
    return exc.__class__.__module__.split(".", maxsplit=1)[0] == "cv2"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return run_one_finger_teleop(
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
            index_open_curl=args.index_open_curl,
            index_closed_curl=args.index_closed_curl,
            index_response_gamma=args.index_response_gamma,
            sim_steps_per_frame=args.sim_steps_per_frame,
            viewer_sleep=args.viewer_sleep,
            print_interval=args.print_interval,
            show_camera_window=args.show_camera_window,
        )
    except (
        CameraOpenError,
        HandTrackerError,
        MujocoError,
        CurlRetargeterError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. One-finger teleop check closed cleanly.")
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
