"""MediaPipe hand landmark tracking for DexVision."""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np


DEFAULT_HAND_LANDMARKER_MODEL = (
    Path(__file__).resolve().parents[2] / "assets" / "models" / "hand_landmarker.task"
)


class HandTrackerError(RuntimeError):
    """Raised when hand tracking cannot be initialized or used."""


@dataclass(frozen=True)
class HandTrackingResult:
    """One hand-tracking result.

    Attributes:
        detected: Whether a hand was detected in the frame.
        handedness: MediaPipe handedness label, usually "Left" or "Right".
        confidence: Handedness confidence score in [0.0, 1.0] when available.
        image_landmarks: Normalized image landmarks with shape [21, 3].
        world_landmarks: Metric world landmarks with shape [21, 3] when available.
        timestamp: Monotonic timestamp in seconds.
    """

    detected: bool
    handedness: str | None
    confidence: float
    image_landmarks: np.ndarray | None
    world_landmarks: np.ndarray | None
    timestamp: float


class LegacyHandsLike(Protocol):
    """Minimal legacy MediaPipe Hands protocol used by this module."""

    def process(self, image: np.ndarray) -> Any: ...

    def close(self) -> None: ...


class HandBackend(Protocol):
    """Internal hand-tracking backend protocol."""

    def process(self, image: np.ndarray, timestamp_ms: int) -> Any: ...

    def close(self) -> None: ...


HandsFactory = Callable[..., LegacyHandsLike]


class HandTracker:
    """Track a single hand in BGR camera frames using MediaPipe Hands."""

    def __init__(
        self,
        *,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | Path | None = None,
        hands_factory: HandsFactory | None = None,
        assume_mirrored_input: bool = False,
    ) -> None:
        if max_num_hands <= 0:
            raise ValueError("max_num_hands must be a positive integer.")
        self._assume_mirrored_input = assume_mirrored_input
        self._last_timestamp_ms: int | None = None
        self._backend = self._make_backend(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_path=model_path,
            hands_factory=hands_factory,
        )

    def process(self, frame: np.ndarray, *, timestamp: float | None = None) -> HandTrackingResult:
        """Detect hand landmarks in one BGR frame.

        Args:
            frame: BGR image with shape [H, W, 3] and dtype uint8.
            timestamp: Optional timestamp to carry through from camera capture.

        Returns:
            A stable ``HandTrackingResult``. No-hand frames return ``detected=False``
            with landmark arrays set to ``None``.
        """

        result_timestamp = time.monotonic() if timestamp is None else timestamp
        self._validate_frame(frame)

        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        rgb_frame.flags.writeable = False
        raw_result = self._backend.process(rgb_frame, self._timestamp_ms(result_timestamp))

        image_landmark_sets = _first_available_attr(
            raw_result,
            ("multi_hand_landmarks", "hand_landmarks"),
        )
        if not image_landmark_sets:
            return self.no_hand_result(result_timestamp)

        image_landmarks = _landmark_list_to_array(image_landmark_sets[0])
        world_landmarks = _first_world_landmark_array(raw_result)
        handedness, confidence = _first_handedness(raw_result)
        handedness = _correct_handedness(
            handedness,
            assume_mirrored_input=self._assume_mirrored_input,
        )

        return HandTrackingResult(
            detected=True,
            handedness=handedness,
            confidence=confidence,
            image_landmarks=image_landmarks,
            world_landmarks=world_landmarks,
            timestamp=result_timestamp,
        )

    def close(self) -> None:
        """Release the MediaPipe graph resources."""

        self._backend.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def no_hand_result(timestamp: float | None = None) -> HandTrackingResult:
        """Create a schema-stable no-hand result."""

        return HandTrackingResult(
            detected=False,
            handedness=None,
            confidence=0.0,
            image_landmarks=None,
            world_landmarks=None,
            timestamp=time.monotonic() if timestamp is None else timestamp,
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape [H, W, 3].")
        if frame.dtype != np.uint8:
            raise ValueError("frame must have dtype uint8.")

    def _timestamp_ms(self, timestamp: float) -> int:
        timestamp_ms = int(round(timestamp * 1000.0))
        if self._last_timestamp_ms is not None:
            timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    @staticmethod
    def _make_backend(
        *,
        static_image_mode: bool,
        max_num_hands: int,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_path: str | Path | None,
        hands_factory: HandsFactory | None,
    ) -> HandBackend:
        if hands_factory is None:
            try:
                import mediapipe as mp
            except ImportError as exc:  # pragma: no cover - depends on environment.
                raise HandTrackerError(
                    "MediaPipe is required for hand tracking. Install the package "
                    f"providing 'mediapipe' ({exc})."
                ) from exc

            legacy_hands = getattr(getattr(mp, "solutions", None), "hands", None)
            if legacy_hands is not None:
                hands_factory = legacy_hands.Hands
            else:
                return _make_tasks_backend(
                    mp_module=mp,
                    static_image_mode=static_image_mode,
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    model_path=model_path,
                )

        try:
            return _LegacyHandsBackend(
                hands_factory(
                    static_image_mode=static_image_mode,
                    max_num_hands=max_num_hands,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive dependency wrapper.
            raise HandTrackerError(f"Could not initialize MediaPipe Hands: {exc}") from exc


class _LegacyHandsBackend:
    def __init__(self, hands: LegacyHandsLike) -> None:
        self._hands = hands

    def process(self, image: np.ndarray, timestamp_ms: int) -> Any:
        del timestamp_ms
        return self._hands.process(image)

    def close(self) -> None:
        self._hands.close()


class _TasksHandLandmarkerBackend:
    def __init__(self, *, mp_module: Any, landmarker: Any, image_mode: bool) -> None:
        self._mp = mp_module
        self._landmarker = landmarker
        self._image_mode = image_mode

    def process(self, image: np.ndarray, timestamp_ms: int) -> Any:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image)
        if self._image_mode:
            return self._landmarker.detect(mp_image)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()


def _make_tasks_backend(
    *,
    mp_module: Any,
    static_image_mode: bool,
    max_num_hands: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    model_path: str | Path | None,
) -> HandBackend:
    tasks = getattr(mp_module, "tasks", None)
    vision = getattr(tasks, "vision", None)
    if tasks is None or vision is None or not hasattr(vision, "HandLandmarker"):
        raise HandTrackerError(
            "This MediaPipe installation exposes neither legacy Hands nor Tasks "
            "HandLandmarker. Reinstall the 'mediapipe' package."
        )

    resolved_model_path = _resolve_model_path(model_path)
    running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
    options = vision.HandLandmarkerOptions(
        base_options=tasks.BaseOptions(
            model_asset_path=str(resolved_model_path),
            delegate=tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=running_mode,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    try:
        landmarker = vision.HandLandmarker.create_from_options(options)
    except Exception as exc:  # pragma: no cover - defensive dependency wrapper.
        raise HandTrackerError(
            "Could not initialize MediaPipe Tasks HandLandmarker with model "
            f"{resolved_model_path}: {exc}"
        ) from exc
    return _TasksHandLandmarkerBackend(
        mp_module=mp_module,
        landmarker=landmarker,
        image_mode=static_image_mode,
    )


def _resolve_model_path(model_path: str | Path | None) -> Path:
    resolved_model_path = (
        Path(model_path) if model_path is not None else DEFAULT_HAND_LANDMARKER_MODEL
    )
    if not resolved_model_path.exists():
        raise HandTrackerError(
            "MediaPipe Tasks HandLandmarker requires a local model bundle. "
            f"Expected {resolved_model_path}. Download the HandLandmarker model "
            "or pass --model-path /path/to/hand_landmarker.task."
        )
    if not resolved_model_path.is_file():
        raise HandTrackerError(f"Model path must be a file: {resolved_model_path}")
    return resolved_model_path


def _landmark_list_to_array(landmark_list: Any) -> np.ndarray:
    landmarks = getattr(landmark_list, "landmark", landmark_list)
    if landmarks is None:
        raise HandTrackerError("MediaPipe result is missing landmark data.")

    points = np.asarray(
        [[float(point.x), float(point.y), float(point.z)] for point in landmarks],
        dtype=np.float32,
    )
    if points.shape != (21, 3):
        raise HandTrackerError(
            "Expected 21 hand landmarks with x/y/z coordinates, "
            f"got shape {points.shape}."
        )
    return points


def _first_world_landmark_array(raw_result: Any) -> np.ndarray | None:
    world_landmark_sets = _first_available_attr(
        raw_result,
        ("multi_hand_world_landmarks", "hand_world_landmarks"),
    )
    if not world_landmark_sets:
        return None
    return _landmark_list_to_array(world_landmark_sets[0])


def _first_handedness(raw_result: Any) -> tuple[str | None, float]:
    handedness_sets = _first_available_attr(raw_result, ("multi_handedness", "handedness"))
    if not handedness_sets:
        return None, 0.0

    classifications = getattr(handedness_sets[0], "classification", handedness_sets[0])
    if not classifications:
        return None, 0.0

    classification = classifications[0]
    label = (
        getattr(classification, "label", None)
        or getattr(classification, "category_name", None)
        or getattr(classification, "display_name", None)
    )
    score = float(getattr(classification, "score", 0.0))
    return label, score


def _correct_handedness(label: str | None, *, assume_mirrored_input: bool) -> str | None:
    if assume_mirrored_input:
        return label
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def _first_available_attr(raw_result: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = getattr(raw_result, name, None)
        if value is not None:
            return value
    return None
