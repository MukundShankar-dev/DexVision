"""Convert MediaPipe hand landmarks into DexVision hand-control features."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Final

import numpy as np

from dexvision.perception.hand_tracker import HandTrackingResult


FINGER_CURL_FIELDS: Final[tuple[str, ...]] = (
    "thumb_curl",
    "index_curl",
    "middle_curl",
    "ring_curl",
    "pinky_curl",
)

_FINGER_JOINTS: Final[dict[str, tuple[tuple[int, int, int], ...]]] = {
    "thumb": ((1, 2, 3), (2, 3, 4)),
    "index": ((5, 6, 7), (6, 7, 8)),
    "middle": ((9, 10, 11), (10, 11, 12)),
    "ring": ((13, 14, 15), (14, 15, 16)),
    "pinky": ((17, 18, 19), (18, 19, 20)),
}
_NON_THUMB_FINGER_LANDMARKS: Final[dict[str, tuple[int, int, int, int]]] = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_NON_THUMB_FINGERS: Final[tuple[str, ...]] = tuple(_NON_THUMB_FINGER_LANDMARKS)
_PALM_WIDTH_EPSILON: Final[float] = 1e-6
_OPEN_ANGLE_RAD: Final[float] = 170.0 * pi / 180.0
_CLOSED_ANGLE_RAD: Final[float] = 90.0 * pi / 180.0
_THUMB_CLOSED_ANGLE_RAD: Final[float] = 130.0 * pi / 180.0
_NON_THUMB_OPEN_EXTENSION_RATIO: Final[float] = 0.96
_NON_THUMB_CLOSED_EXTENSION_RATIO: Final[float] = 0.72


@dataclass(frozen=True)
class HandFeatures:
    """Normalized hand features derived from one set of 21 hand landmarks.

    Curl values use ``0.0`` for extended/open fingers and ``1.0`` for curled
    fingers. ``pinch_thumb_index`` is the thumb-tip to index-tip distance
    normalized by palm width, so smaller values mean a tighter pinch.
    """

    thumb_curl: float
    index_curl: float
    middle_curl: float
    ring_curl: float
    pinky_curl: float
    pinch_thumb_index: float
    palm_roll_proxy: float
    palm_pitch_proxy: float
    confidence: float

    @property
    def palm_roll(self) -> float:
        """Compatibility alias for the current module contract."""

        return self.palm_roll_proxy

    @property
    def palm_pitch(self) -> float:
        """Compatibility alias for the current module contract."""

        return self.palm_pitch_proxy


def no_hand_features() -> HandFeatures:
    """Return a finite neutral feature vector for frames with no tracked hand."""

    return HandFeatures(
        thumb_curl=0.0,
        index_curl=0.0,
        middle_curl=0.0,
        ring_curl=0.0,
        pinky_curl=0.0,
        pinch_thumb_index=0.0,
        palm_roll_proxy=0.0,
        palm_pitch_proxy=0.0,
        confidence=0.0,
    )


def extract_hand_features(
    result: HandTrackingResult,
    *,
    prefer_world_landmarks: bool = True,
) -> HandFeatures:
    """Convert a hand-tracking result into normalized hand features.

    Args:
        result: Stable hand-tracking output from ``dexvision.perception``.
        prefer_world_landmarks: Use metric world landmarks when available,
            otherwise fall back to normalized image landmarks.

    Returns:
        Finite ``HandFeatures``. Missing hands return ``no_hand_features()``.
    """

    if not result.detected:
        return no_hand_features()

    landmarks = (
        result.world_landmarks
        if prefer_world_landmarks and result.world_landmarks is not None
        else result.image_landmarks
    )
    if landmarks is None:
        return no_hand_features()

    features = compute_hand_features(landmarks, confidence=result.confidence)
    if result.image_landmarks is None or landmarks is result.image_landmarks:
        return features

    image_points = _as_landmark_array(result.image_landmarks)
    curls = _image_vetoed_non_thumb_curls(features, image_points)

    return HandFeatures(
        thumb_curl=features.thumb_curl,
        index_curl=curls["index"],
        middle_curl=curls["middle"],
        ring_curl=curls["ring"],
        pinky_curl=curls["pinky"],
        pinch_thumb_index=features.pinch_thumb_index,
        palm_roll_proxy=features.palm_roll_proxy,
        palm_pitch_proxy=features.palm_pitch_proxy,
        confidence=features.confidence,
    )


def compute_hand_features(landmarks: np.ndarray, *, confidence: float = 1.0) -> HandFeatures:
    """Compute hand features from landmarks with shape ``[21, 3]``.

    Args:
        landmarks: Hand landmarks as ``x, y, z`` coordinates with shape
            ``[21, 3]``. Coordinates may be normalized image coordinates or
            world coordinates; features are scale-invariant where possible.
        confidence: Upstream hand-tracking confidence.

    Returns:
        A finite ``HandFeatures`` instance.
    """

    points = _as_landmark_array(landmarks)
    palm_width = _palm_width(points)

    thumb_curl = _finger_curl(points, "thumb")
    index_curl = _finger_curl(points, "index")
    middle_curl = _finger_curl(points, "middle")
    ring_curl = _finger_curl(points, "ring")
    pinky_curl = _finger_curl(points, "pinky")
    pinch_thumb_index = _clip01(_distance(points[4], points[8]) / palm_width)
    palm_roll_proxy, palm_pitch_proxy = _palm_orientation_proxies(points)

    return HandFeatures(
        thumb_curl=thumb_curl,
        index_curl=index_curl,
        middle_curl=middle_curl,
        ring_curl=ring_curl,
        pinky_curl=pinky_curl,
        pinch_thumb_index=pinch_thumb_index,
        palm_roll_proxy=palm_roll_proxy,
        palm_pitch_proxy=palm_pitch_proxy,
        confidence=_clip01(confidence),
    )


def feature_values(features: HandFeatures) -> tuple[float, ...]:
    """Return feature values in a stable order for tests and simple consumers."""

    return (
        features.thumb_curl,
        features.index_curl,
        features.middle_curl,
        features.ring_curl,
        features.pinky_curl,
        features.pinch_thumb_index,
        features.palm_roll_proxy,
        features.palm_pitch_proxy,
        features.confidence,
    )


def _as_landmark_array(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"landmarks must have shape [21, 3], got {points.shape}.")
    return np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)


def _finger_curl(points: np.ndarray, finger: str) -> float:
    if finger == "thumb":
        return _finger_angle_curl(points, finger, closed_angle=_THUMB_CLOSED_ANGLE_RAD)
    return _non_thumb_finger_curl(points, finger)


def _finger_angle_curl(points: np.ndarray, finger: str, *, closed_angle: float) -> float:
    curls = [
        _angle_to_curl(_joint_angle(points[a], points[b], points[c]), closed_angle=closed_angle)
        for a, b, c in _FINGER_JOINTS[finger]
    ]
    return _clip01(float(np.mean(curls)))


def _non_thumb_finger_curl(points: np.ndarray, finger: str) -> float:
    angle_curl = _finger_angle_curl(points, finger, closed_angle=_CLOSED_ANGLE_RAD)
    extension_curl = _non_thumb_local_extension_curl(points, finger)
    return min(angle_curl, extension_curl)


def _image_vetoed_non_thumb_curls(
    features: HandFeatures,
    image_points: np.ndarray,
) -> dict[str, float]:
    curls = {
        "index": features.index_curl,
        "middle": features.middle_curl,
        "ring": features.ring_curl,
        "pinky": features.pinky_curl,
    }
    for finger in _NON_THUMB_FINGERS:
        if _non_thumb_finger_is_confidently_extended(image_points, finger):
            curls[finger] = min(curls[finger], _non_thumb_finger_curl(image_points, finger))
    return curls


def _non_thumb_finger_is_confidently_extended(points: np.ndarray, finger: str) -> bool:
    return (
        _non_thumb_full_extension_ratio(points, finger) >= 0.80
        and _non_thumb_finger_curl(points, finger) <= 0.25
    )


def _non_thumb_full_extension_ratio(points: np.ndarray, finger: str) -> float:
    base, pip, dip, tip = _NON_THUMB_FINGER_LANDMARKS[finger]
    chain_length = _distance(points[base], points[pip])
    chain_length += _distance(points[pip], points[dip])
    chain_length += _distance(points[dip], points[tip])
    if chain_length <= _PALM_WIDTH_EPSILON:
        return 0.0
    return _distance(points[base], points[tip]) / chain_length


def _non_thumb_local_extension_curl(points: np.ndarray, finger: str) -> float:
    _base, pip, dip, tip = _NON_THUMB_FINGER_LANDMARKS[finger]
    chain_length = _distance(points[pip], points[dip]) + _distance(points[dip], points[tip])
    if chain_length <= _PALM_WIDTH_EPSILON:
        return 0.0

    extension_ratio = _distance(points[pip], points[tip]) / chain_length
    normalized = (_NON_THUMB_OPEN_EXTENSION_RATIO - extension_ratio) / (
        _NON_THUMB_OPEN_EXTENSION_RATIO - _NON_THUMB_CLOSED_EXTENSION_RATIO
    )
    return _clip01(normalized)


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = a - b
    second = c - b
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= _PALM_WIDTH_EPSILON or second_norm <= _PALM_WIDTH_EPSILON:
        return pi

    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _angle_to_curl(angle_rad: float, *, closed_angle: float) -> float:
    normalized = (_OPEN_ANGLE_RAD - angle_rad) / (_OPEN_ANGLE_RAD - closed_angle)
    return _clip01(normalized)


def _palm_orientation_proxies(points: np.ndarray) -> tuple[float, float]:
    across_palm = points[17] - points[5]
    toward_fingers = points[9] - points[0]
    normal = np.cross(across_palm, toward_fingers)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= _PALM_WIDTH_EPSILON:
        return 0.0, 0.0

    normal = normal / normal_norm
    return _clip_signed(float(normal[0])), _clip_signed(float(normal[1]))


def _palm_width(points: np.ndarray) -> float:
    palm_width = _distance(points[5], points[17])
    if palm_width > _PALM_WIDTH_EPSILON:
        return palm_width

    fallback = _distance(points[0], points[9])
    return max(fallback, 1.0)


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _clip_signed(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))
