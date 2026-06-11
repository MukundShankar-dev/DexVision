"""Convert MediaPipe hand landmarks into local DexVision hand features."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np

from dexvision.perception.hand_tracker import HandTrackingResult


FINGER_CURL_FIELDS: Final[tuple[str, ...]] = (
    "thumb_curl",
    "index_curl",
    "middle_curl",
    "ring_curl",
    "pinky_curl",
)
FINGER_BEND_FIELDS: Final[tuple[str, ...]] = (
    "index_bend",
    "middle_bend",
    "ring_bend",
    "pinky_bend",
)
FINGER_CONTROL_FIELDS: Final[tuple[str, ...]] = FINGER_CURL_FIELDS + FINGER_BEND_FIELDS
FINGER_NAMES: Final[tuple[str, ...]] = ("thumb", "index", "middle", "ring", "pinky")
NON_THUMB_FINGER_NAMES: Final[tuple[str, ...]] = ("index", "middle", "ring", "pinky")

_FINGER_LANDMARKS: Final[Mapping[str, tuple[int, int, int, int]]] = MappingProxyType(
    {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
)
_FINGER_JOINTS: Final[Mapping[str, tuple[tuple[int, int, int], ...]]] = MappingProxyType(
    {
        "thumb": ((1, 2, 3), (2, 3, 4)),
        "index": ((5, 6, 7), (6, 7, 8)),
        "middle": ((9, 10, 11), (10, 11, 12)),
        "ring": ((13, 14, 15), (14, 15, 16)),
        "pinky": ((17, 18, 19), (18, 19, 20)),
    }
)

_EPSILON: Final[float] = 1e-6
_OPEN_ANGLE_RAD: Final[float] = 170.0 * pi / 180.0
_CLOSED_ANGLE_RAD: Final[float] = 90.0 * pi / 180.0
_THUMB_CLOSED_ANGLE_RAD: Final[float] = 130.0 * pi / 180.0
_UP_EXTENSION_THRESHOLD: Final[float] = 0.55
_UP_CURL_THRESHOLD: Final[float] = 0.35


@dataclass(frozen=True)
class FingerState:
    """State for one finger derived from local landmarks."""

    curl: float
    extension: float
    abduction: float | None
    is_up: bool
    valid: bool


@dataclass(frozen=True, eq=False)
class PalmState:
    """Palm-local coordinate frame.

    Axes are unit vectors in the source landmark coordinate system. ``x_axis``
    runs across the palm from pinky MCP toward index MCP, ``y_axis`` points from
    wrist toward the fingers, and ``z_axis`` is the palm normal.
    """

    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    valid: bool

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PalmState):
            return NotImplemented
        return (
            self.valid == other.valid
            and np.allclose(self.origin, other.origin)
            and np.allclose(self.x_axis, other.x_axis)
            and np.allclose(self.y_axis, other.y_axis)
            and np.allclose(self.z_axis, other.z_axis)
        )


@dataclass(frozen=True)
class FingerCalibration:
    """Optional per-finger normalization bounds.

    TODO: Populate these from a live calibration flow that records open-hand and
    fist baselines for the current camera/user setup.
    """

    curl_min: float = 0.0
    curl_max: float = 1.0
    extension_min: float = 0.0
    extension_max: float = 1.0


@dataclass(frozen=True)
class HandFeatureCalibration:
    """Optional calibration container for local finger features."""

    fingers: Mapping[str, FingerCalibration] = field(default_factory=dict)
    open_hand_baseline: "HandFeatures | None" = None
    fist_baseline: "HandFeatures | None" = None


@dataclass(frozen=True, init=False)
class HandFeatures:
    """Structured hand features with legacy scalar compatibility fields."""

    thumb: FingerState
    index: FingerState
    middle: FingerState
    ring: FingerState
    pinky: FingerState
    palm: PalmState
    pinch_thumb_index: float
    confidence: float
    palm_roll_proxy_value: float
    palm_pitch_proxy_value: float

    def __init__(
        self,
        *,
        thumb: FingerState | None = None,
        index: FingerState | None = None,
        middle: FingerState | None = None,
        ring: FingerState | None = None,
        pinky: FingerState | None = None,
        palm: PalmState | None = None,
        pinch_thumb_index: float = 0.0,
        confidence: float = 1.0,
        palm_roll_proxy: float = 0.0,
        palm_pitch_proxy: float = 0.0,
        palm_roll_proxy_value: float | None = None,
        palm_pitch_proxy_value: float | None = None,
        thumb_curl: float | None = None,
        index_curl: float | None = None,
        middle_curl: float | None = None,
        ring_curl: float | None = None,
        pinky_curl: float | None = None,
    ) -> None:
        object.__setattr__(self, "thumb", _coerce_finger_state(thumb, thumb_curl))
        object.__setattr__(self, "index", _coerce_finger_state(index, index_curl))
        object.__setattr__(self, "middle", _coerce_finger_state(middle, middle_curl))
        object.__setattr__(self, "ring", _coerce_finger_state(ring, ring_curl))
        object.__setattr__(self, "pinky", _coerce_finger_state(pinky, pinky_curl))
        object.__setattr__(self, "palm", palm or _invalid_palm_state())
        object.__setattr__(self, "pinch_thumb_index", _clip01(pinch_thumb_index))
        object.__setattr__(self, "confidence", _clip01(confidence))
        object.__setattr__(
            self,
            "palm_roll_proxy_value",
            _clip_signed(
                palm_roll_proxy if palm_roll_proxy_value is None else palm_roll_proxy_value
            ),
        )
        object.__setattr__(
            self,
            "palm_pitch_proxy_value",
            _clip_signed(
                palm_pitch_proxy if palm_pitch_proxy_value is None else palm_pitch_proxy_value
            ),
        )

    @property
    def thumb_curl(self) -> float:
        return self.thumb.curl

    @property
    def index_curl(self) -> float:
        return self.index.curl

    @property
    def middle_curl(self) -> float:
        return self.middle.curl

    @property
    def ring_curl(self) -> float:
        return self.ring.curl

    @property
    def pinky_curl(self) -> float:
        return self.pinky.curl

    @property
    def index_bend(self) -> float:
        return _finger_bend(self.index)

    @property
    def middle_bend(self) -> float:
        return _finger_bend(self.middle)

    @property
    def ring_bend(self) -> float:
        return _finger_bend(self.ring)

    @property
    def pinky_bend(self) -> float:
        return _finger_bend(self.pinky)

    @property
    def palm_roll_proxy(self) -> float:
        return self.palm_roll_proxy_value

    @property
    def palm_pitch_proxy(self) -> float:
        return self.palm_pitch_proxy_value

    @property
    def palm_roll(self) -> float:
        return self.palm_roll_proxy

    @property
    def palm_pitch(self) -> float:
        return self.palm_pitch_proxy

    @property
    def palm_yaw(self) -> float:
        return 0.0


def no_hand_features() -> HandFeatures:
    """Return a finite neutral feature vector for frames with no tracked hand."""

    invalid = _invalid_finger_state()
    return HandFeatures(
        thumb=invalid,
        index=invalid,
        middle=invalid,
        ring=invalid,
        pinky=invalid,
        palm=_invalid_palm_state(),
        pinch_thumb_index=0.0,
        palm_roll_proxy=0.0,
        palm_pitch_proxy=0.0,
        confidence=0.0,
    )


def extract_hand_features(
    result: HandTrackingResult,
    *,
    prefer_world_landmarks: bool = False,
    calibration: HandFeatureCalibration | None = None,
) -> HandFeatures:
    """Convert a hand-tracking result into structured local hand features."""

    if not result.detected:
        return no_hand_features()

    landmarks = _select_landmarks(
        image_landmarks=result.image_landmarks,
        world_landmarks=result.world_landmarks,
        prefer_world_landmarks=prefer_world_landmarks,
    )
    if landmarks is None:
        return no_hand_features()

    return compute_hand_features(
        landmarks,
        confidence=result.confidence,
        calibration=calibration,
    )


def compute_hand_features(
    landmarks: np.ndarray,
    *,
    confidence: float = 1.0,
    calibration: HandFeatureCalibration | None = None,
) -> HandFeatures:
    """Compute local per-finger features from landmarks with shape ``[21, 3]``."""

    points = _as_landmark_array(landmarks)
    palm = _palm_state(points)
    thumb = _apply_finger_calibration(_thumb_state(points, palm), "thumb", calibration)
    index = _apply_finger_calibration(
        _long_finger_state(points, palm, "index"),
        "index",
        calibration,
    )
    middle = _apply_finger_calibration(
        _long_finger_state(points, palm, "middle"),
        "middle",
        calibration,
    )
    ring = _apply_finger_calibration(_long_finger_state(points, palm, "ring"), "ring", calibration)
    pinky = _apply_finger_calibration(
        _long_finger_state(points, palm, "pinky"),
        "pinky",
        calibration,
    )
    pinch_thumb_index = _pinch_thumb_index(points, palm)
    palm_roll_proxy, palm_pitch_proxy = _palm_orientation_proxies(palm)

    return HandFeatures(
        thumb=thumb,
        index=index,
        middle=middle,
        ring=ring,
        pinky=pinky,
        palm=palm,
        pinch_thumb_index=pinch_thumb_index,
        palm_roll_proxy=palm_roll_proxy,
        palm_pitch_proxy=palm_pitch_proxy,
        confidence=confidence,
    )


def feature_values(features: HandFeatures) -> tuple[float, ...]:
    """Return scalar compatibility values in a stable order."""

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


def _coerce_finger_state(state: FingerState | None, legacy_curl: float | None) -> FingerState:
    if state is None:
        return _finger_from_legacy_curl(legacy_curl)
    return FingerState(
        curl=_clip01(state.curl),
        extension=_clip01(state.extension),
        abduction=None if state.abduction is None else _clip_signed(state.abduction),
        is_up=bool(state.is_up),
        valid=bool(state.valid),
    )


def _finger_from_legacy_curl(value: float | None) -> FingerState:
    curl = _clip01(0.0 if value is None else value)
    extension = _clip01(1.0 - curl)
    return FingerState(
        curl=curl,
        extension=extension,
        abduction=None,
        is_up=extension >= _UP_EXTENSION_THRESHOLD and curl <= _UP_CURL_THRESHOLD,
        valid=True,
    )


def _finger_bend(state: FingerState) -> float:
    return _clip01(1.0 - state.extension)


def _select_landmarks(
    *,
    image_landmarks: np.ndarray | None,
    world_landmarks: np.ndarray | None,
    prefer_world_landmarks: bool,
) -> np.ndarray | None:
    if prefer_world_landmarks and world_landmarks is not None:
        return world_landmarks
    if image_landmarks is not None:
        return image_landmarks
    return world_landmarks


def _as_landmark_array(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"landmarks must have shape [21, 3], got {points.shape}.")
    return np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)


def _palm_state(points: np.ndarray) -> PalmState:
    origin = points[0].astype(np.float32)
    across = points[5] - points[17]
    toward_fingers = points[9] - points[0]

    x_axis = _unit(across)
    z_axis = _unit(np.cross(x_axis, toward_fingers))
    y_axis = _unit(np.cross(z_axis, x_axis))
    valid = _is_valid_axis(x_axis) and _is_valid_axis(y_axis) and _is_valid_axis(z_axis)
    if not valid:
        return _invalid_palm_state(origin=origin)

    return PalmState(
        origin=origin,
        x_axis=x_axis.astype(np.float32),
        y_axis=y_axis.astype(np.float32),
        z_axis=z_axis.astype(np.float32),
        valid=True,
    )


def _long_finger_state(points: np.ndarray, palm: PalmState, finger: str) -> FingerState:
    base, pip, dip, tip = _FINGER_LANDMARKS[finger]
    chain_length = _finger_chain_length(points, (base, pip, dip, tip))
    valid = palm.valid and chain_length > _EPSILON
    if not valid:
        return _invalid_finger_state()

    curl = _finger_angle_curl(points, finger, closed_angle=_CLOSED_ANGLE_RAD)
    extension = _long_finger_extension(points, palm, base=base, tip=tip, chain_length=chain_length)
    abduction = _finger_abduction(points, palm, base=base, tip=tip, chain_length=chain_length)
    return FingerState(
        curl=curl,
        extension=extension,
        abduction=abduction,
        is_up=extension >= _UP_EXTENSION_THRESHOLD and curl <= _UP_CURL_THRESHOLD,
        valid=True,
    )


def _thumb_state(points: np.ndarray, palm: PalmState) -> FingerState:
    cmc, mcp, ip, tip = _FINGER_LANDMARKS["thumb"]
    chain_length = _finger_chain_length(points, (cmc, mcp, ip, tip))
    valid = palm.valid and chain_length > _EPSILON
    if not valid:
        return _invalid_finger_state()

    curl = _finger_angle_curl(points, "thumb", closed_angle=_THUMB_CLOSED_ANGLE_RAD)
    extension = _clip01(_distance(points[cmc], points[tip]) / chain_length)
    opposition = _thumb_opposition(points, palm)
    return FingerState(
        curl=curl,
        extension=extension,
        abduction=opposition,
        is_up=extension >= 0.45 and curl <= 0.45,
        valid=True,
    )


def _finger_angle_curl(points: np.ndarray, finger: str, *, closed_angle: float) -> float:
    curls = [
        _angle_to_curl(_joint_angle(points[a], points[b], points[c]), closed_angle=closed_angle)
        for a, b, c in _FINGER_JOINTS[finger]
    ]
    return _clip01(float(np.mean(curls)))


def _long_finger_extension(
    points: np.ndarray,
    palm: PalmState,
    *,
    base: int,
    tip: int,
    chain_length: float,
) -> float:
    local_tip = _to_palm_local(points[tip], palm)
    local_base = _to_palm_local(points[base], palm)
    along_fingers = float(local_tip[1] - local_base[1])
    return _clip01(along_fingers / chain_length)


def _finger_abduction(
    points: np.ndarray,
    palm: PalmState,
    *,
    base: int,
    tip: int,
    chain_length: float,
) -> float:
    local_tip = _to_palm_local(points[tip], palm)
    local_base = _to_palm_local(points[base], palm)
    across_palm = float(local_tip[0] - local_base[0])
    return _clip_signed(across_palm / chain_length)


def _thumb_opposition(points: np.ndarray, palm: PalmState) -> float | None:
    if not palm.valid:
        return None
    palm_width = _palm_width(points)
    if palm_width <= _EPSILON:
        return None
    thumb_tip = points[4]
    index_mcp = points[5]
    return _clip01(1.0 - (_distance(thumb_tip, index_mcp) / (1.6 * palm_width)))


def _pinch_thumb_index(points: np.ndarray, palm: PalmState) -> float:
    palm_width = _palm_width(points)
    if not palm.valid or palm_width <= _EPSILON:
        return 0.0
    return _clip01(_distance(points[4], points[8]) / palm_width)


def _apply_finger_calibration(
    state: FingerState,
    finger: str,
    calibration: HandFeatureCalibration | None,
) -> FingerState:
    if calibration is None or finger not in calibration.fingers:
        return state

    bounds = calibration.fingers[finger]
    curl = _normalize_range(state.curl, bounds.curl_min, bounds.curl_max)
    extension = _normalize_range(state.extension, bounds.extension_min, bounds.extension_max)
    return FingerState(
        curl=curl,
        extension=extension,
        abduction=state.abduction,
        is_up=extension >= _UP_EXTENSION_THRESHOLD and curl <= _UP_CURL_THRESHOLD,
        valid=state.valid,
    )


def _normalize_range(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return _clip01(value)
    return _clip01((value - minimum) / (maximum - minimum))


def _to_palm_local(point: np.ndarray, palm: PalmState) -> np.ndarray:
    relative = point - palm.origin
    return np.asarray(
        [
            float(np.dot(relative, palm.x_axis)),
            float(np.dot(relative, palm.y_axis)),
            float(np.dot(relative, palm.z_axis)),
        ],
        dtype=np.float32,
    )


def _finger_chain_length(points: np.ndarray, indices: tuple[int, int, int, int]) -> float:
    first, second, third, fourth = indices
    return (
        _distance(points[first], points[second])
        + _distance(points[second], points[third])
        + _distance(points[third], points[fourth])
    )


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = a - b
    second = c - b
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= _EPSILON or second_norm <= _EPSILON:
        return pi

    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _angle_to_curl(angle_rad: float, *, closed_angle: float) -> float:
    normalized = (_OPEN_ANGLE_RAD - angle_rad) / (_OPEN_ANGLE_RAD - closed_angle)
    return _clip01(normalized)


def _palm_orientation_proxies(palm: PalmState) -> tuple[float, float]:
    if not palm.valid:
        return 0.0, 0.0
    return _clip_signed(float(palm.z_axis[0])), _clip_signed(float(palm.z_axis[1]))


def _palm_width(points: np.ndarray) -> float:
    palm_width = _distance(points[5], points[17])
    if palm_width > _EPSILON:
        return palm_width
    fallback = _distance(points[0], points[9])
    return max(fallback, 1.0)


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(vector / norm, dtype=np.float32)


def _is_valid_axis(axis: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(axis)) and float(np.linalg.norm(axis)) > 0.5)


def _invalid_palm_state(origin: np.ndarray | None = None) -> PalmState:
    return PalmState(
        origin=(
            np.zeros(3, dtype=np.float32)
            if origin is None
            else np.asarray(origin, dtype=np.float32)
        ),
        x_axis=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        y_axis=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        z_axis=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        valid=False,
    )


def _invalid_finger_state() -> FingerState:
    return FingerState(curl=0.0, extension=1.0, abduction=None, is_up=False, valid=False)


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _clip_signed(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))
