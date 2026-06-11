"""Estimate and smooth a hand-base target from tracked palm landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from dexvision.perception.hand_tracker import HandTrackingResult


_EPSILON: Final[float] = 1e-6
_PALM_LANDMARKS: Final[tuple[int, int, int, int]] = (0, 5, 9, 17)
PositionSource = Literal["wrist", "palm_center"]
HandScaleSource = Literal["palm_width", "robust_palm_scale"]
DEFAULT_POSITION_SOURCE: Final[PositionSource] = "wrist"
DEFAULT_HAND_SCALE_SOURCE: Final[HandScaleSource] = "robust_palm_scale"
_LANDMARK_TO_CONTROL_AXES: Final[np.ndarray] = np.asarray(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, eq=False)
class HandBaseTarget:
    """Palm/wrist pose target for simulated hand-base control.

    Attributes:
        position: Control-space position with shape ``[3]``.
        orientation_quat: Control-space orientation quaternion ``[w, x, y, z]``.
        confidence: Landmark confidence in ``[0.0, 1.0]``.
        valid: Whether the pose was computed from a usable palm frame.
    """

    position: np.ndarray
    orientation_quat: np.ndarray
    confidence: float
    valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _coerce_vector(self.position, 3, "position"))
        object.__setattr__(
            self,
            "orientation_quat",
            normalize_quaternion(_coerce_vector(self.orientation_quat, 4, "orientation_quat")),
        )
        object.__setattr__(self, "confidence", _clip01(self.confidence))
        object.__setattr__(self, "valid", bool(self.valid))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandBaseTarget):
            return NotImplemented
        return (
            self.valid == other.valid
            and self.confidence == other.confidence
            and np.allclose(self.position, other.position)
            and np.allclose(self.orientation_quat, other.orientation_quat)
        )

    @property
    def orientation_matrix(self) -> np.ndarray:
        """Return the equivalent rotation matrix with shape ``[3, 3]``."""

        return quaternion_to_rotation_matrix(self.orientation_quat)


@dataclass(frozen=True, eq=False)
class ImagePalmCenterTarget:
    """Normalized image-space palm-center/scale target for calibrated base control.

    Attributes:
        palm_center: Normalized image coordinates ``[x, y]`` with shape ``[2]``.
        hand_scale: Monocular image-space hand scale, usually palm width.
        confidence: Landmark confidence in ``[0.0, 1.0]``.
        valid: Whether the center was computed from finite palm landmarks.
    """

    palm_center: np.ndarray
    confidence: float
    valid: bool
    hand_scale: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "palm_center", _coerce_vector(self.palm_center, 2, "palm_center"))
        object.__setattr__(self, "hand_scale", _coerce_nonnegative_scalar(self.hand_scale))
        object.__setattr__(self, "confidence", _clip01(self.confidence))
        object.__setattr__(self, "valid", bool(self.valid))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImagePalmCenterTarget):
            return NotImplemented
        return (
            self.valid == other.valid
            and self.confidence == other.confidence
            and np.allclose(self.palm_center, other.palm_center)
            and np.isclose(self.hand_scale, other.hand_scale)
        )


class HandBaseTargetSmoother:
    """Exponential smoother for hand-base position and orientation targets."""

    def __init__(self, *, alpha: float = 0.35, min_confidence: float = 0.2) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0.0, 1.0].")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in the range [0.0, 1.0].")
        self.alpha = float(alpha)
        self.min_confidence = float(min_confidence)
        self._state: HandBaseTarget | None = None

    @property
    def state(self) -> HandBaseTarget | None:
        """Return the last smoothed target, if initialized."""

        return self._state

    def reset(self) -> None:
        """Clear smoothing history."""

        self._state = None

    def update(self, target: HandBaseTarget | None) -> HandBaseTarget:
        """Update the smoother and return a finite target.

        Invalid or low-confidence input freezes the last valid pose while
        carrying through the current confidence. If no valid pose has been seen,
        a neutral invalid target is returned.
        """

        current = sanitize_hand_base_target(target)
        if not current.valid or current.confidence < self.min_confidence:
            if self._state is None:
                self._state = no_hand_base_target(confidence=current.confidence)
                return self._state
            self._state = HandBaseTarget(
                position=self._state.position,
                orientation_quat=self._state.orientation_quat,
                confidence=current.confidence,
                valid=self._state.valid,
            )
            return self._state

        if self._state is None or not self._state.valid:
            self._state = current
            return self._state

        self._state = HandBaseTarget(
            position=_ema_vector(self._state.position, current.position, self.alpha),
            orientation_quat=quaternion_nlerp(
                self._state.orientation_quat,
                current.orientation_quat,
                self.alpha,
            ),
            confidence=_ema_scalar(self._state.confidence, current.confidence, self.alpha),
            valid=True,
        )
        return self._state


def no_hand_base_target(*, confidence: float = 0.0) -> HandBaseTarget:
    """Return a neutral invalid base target for missing hand tracking."""

    return HandBaseTarget(
        position=np.zeros(3, dtype=np.float64),
        orientation_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        confidence=confidence,
        valid=False,
    )


def no_image_palm_center_target(*, confidence: float = 0.0) -> ImagePalmCenterTarget:
    """Return a neutral invalid image-space palm-center target."""

    return ImagePalmCenterTarget(
        palm_center=np.zeros(2, dtype=np.float64),
        confidence=confidence,
        valid=False,
        hand_scale=0.0,
    )


def extract_image_palm_center_target(
    result: HandTrackingResult,
    *,
    depth_source: HandScaleSource = DEFAULT_HAND_SCALE_SOURCE,
) -> ImagePalmCenterTarget:
    """Convert a hand-tracking result into a normalized image-space palm center."""

    if not result.detected or result.image_landmarks is None:
        return no_image_palm_center_target(confidence=result.confidence)
    return estimate_image_palm_center_target(
        result.image_landmarks,
        confidence=result.confidence,
        depth_source=depth_source,
    )


def estimate_image_palm_center_target(
    landmarks: np.ndarray,
    *,
    confidence: float = 1.0,
    depth_source: HandScaleSource = DEFAULT_HAND_SCALE_SOURCE,
) -> ImagePalmCenterTarget:
    """Estimate palm center and monocular scale from image landmarks ``[21, 3]``."""

    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"landmarks must have shape [21, 3], got {points.shape}.")
    palm_points = points[list(_PALM_LANDMARKS), :2]
    if not np.all(np.isfinite(palm_points)):
        return no_image_palm_center_target(confidence=confidence)
    return ImagePalmCenterTarget(
        palm_center=np.mean(palm_points, axis=0),
        confidence=confidence,
        valid=True,
        hand_scale=estimate_hand_scale(points, depth_source=depth_source),
    )


def estimate_hand_scale(
    landmarks: np.ndarray,
    *,
    depth_source: HandScaleSource = DEFAULT_HAND_SCALE_SOURCE,
) -> float:
    """Estimate normalized image-space hand scale from landmarks with shape ``[21, 3]``.

    ``palm_width`` is the distance between index MCP and pinky MCP.
    ``robust_palm_scale`` averages palm width with wrist-to-middle-MCP length
    and MCP-chain spread to reduce single-landmark jitter.
    """

    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"landmarks must have shape [21, 3], got {points.shape}.")
    source = _coerce_hand_scale_source(depth_source)
    palm_width = _finite_distance_2d(points[5], points[17])
    if source == "palm_width":
        return palm_width

    components = [
        palm_width,
        _finite_distance_2d(points[0], points[9]),
        _finite_distance_2d(points[5], points[9]) + _finite_distance_2d(points[9], points[17]),
    ]
    valid_components = [value for value in components if value > _EPSILON]
    if not valid_components:
        return 0.0
    return float(np.mean(valid_components))


def extract_hand_base_target(
    result: HandTrackingResult,
    *,
    prefer_world_landmarks: bool = False,
    position_source: PositionSource = DEFAULT_POSITION_SOURCE,
) -> HandBaseTarget:
    """Convert a hand-tracking result into a palm/wrist base target."""

    if not result.detected:
        return no_hand_base_target(confidence=result.confidence)
    landmarks = _select_landmarks(
        image_landmarks=result.image_landmarks,
        world_landmarks=result.world_landmarks,
        prefer_world_landmarks=prefer_world_landmarks,
    )
    if landmarks is None:
        return no_hand_base_target(confidence=result.confidence)
    return estimate_hand_base_target(
        landmarks,
        confidence=result.confidence,
        handedness=result.handedness,
        position_source=position_source,
    )


def estimate_hand_base_target(
    landmarks: np.ndarray,
    *,
    confidence: float = 1.0,
    handedness: str | None = None,
    position_source: PositionSource = DEFAULT_POSITION_SOURCE,
) -> HandBaseTarget:
    """Estimate a hand-base pose from landmarks with shape ``[21, 3]``.

    The source landmarks use MediaPipe image/world axes. The returned target is
    expressed in a right-handed control frame where ``x`` approximates camera
    depth/approach, ``y`` approximates camera left-right, and ``z`` approximates
    camera up-down.
    """

    points = _as_landmark_array(landmarks)
    position_anchor = _select_position_anchor(points, position_source)
    position = _landmark_point_to_control_position(position_anchor)

    x_axis_source = _unit(_canonicalize_orientation_vector(points[5] - points[17], handedness))
    y_hint_source = _unit(_canonicalize_orientation_vector(points[9] - points[0], handedness))
    if not _is_valid_axis(x_axis_source) or not _is_valid_axis(y_hint_source):
        return no_hand_base_target(confidence=confidence)

    x_axis = _unit(_LANDMARK_TO_CONTROL_AXES @ x_axis_source)
    y_hint = _unit(_LANDMARK_TO_CONTROL_AXES @ y_hint_source)
    z_axis = _unit(np.cross(x_axis, y_hint))
    y_axis = _unit(np.cross(z_axis, x_axis))
    if not (_is_valid_axis(x_axis) and _is_valid_axis(y_axis) and _is_valid_axis(z_axis)):
        return no_hand_base_target(confidence=confidence)

    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    if float(np.linalg.det(rotation_matrix)) < 0.0:
        z_axis = -z_axis
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    return HandBaseTarget(
        position=position,
        orientation_quat=rotation_matrix_to_quaternion(rotation_matrix),
        confidence=confidence,
        valid=True,
    )


def sanitize_hand_base_target(target: HandBaseTarget | None) -> HandBaseTarget:
    """Return a finite hand-base target."""

    if target is None:
        return no_hand_base_target()
    if not target.valid:
        return no_hand_base_target(confidence=target.confidence)
    if not np.all(np.isfinite(target.position)) or not np.all(np.isfinite(target.orientation_quat)):
        return no_hand_base_target(confidence=target.confidence)
    return HandBaseTarget(
        position=target.position,
        orientation_quat=target.orientation_quat,
        confidence=target.confidence,
        valid=True,
    )


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a MuJoCo-compatible ``[w, x, y, z]`` quaternion."""

    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape [3, 3], got {matrix.shape}.")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = float(np.sqrt(trace + 1.0) * 2.0)
        quat = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = float(np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0)
            quat = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif diagonal_index == 1:
            scale = float(np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0)
            quat = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = float(np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0)
            quat = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    return normalize_quaternion(quat)


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a ``[w, x, y, z]`` quaternion to a rotation matrix."""

    w, x, y, z = normalize_quaternion(quaternion)
    return np.asarray(
        [
            [1.0 - (2.0 * (y * y + z * z)), 2.0 * ((x * y) - (z * w)), 2.0 * ((x * z) + (y * w))],
            [2.0 * ((x * y) + (z * w)), 1.0 - (2.0 * (x * x + z * z)), 2.0 * ((y * z) - (x * w))],
            [2.0 * ((x * z) - (y * w)), 2.0 * ((y * z) + (x * w)), 1.0 - (2.0 * (x * x + y * y))],
        ],
        dtype=np.float64,
    )


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Return a unit quaternion, falling back to identity for invalid input."""

    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= _EPSILON:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    normalized = quat / norm
    return normalized if normalized[0] >= 0.0 else -normalized


def quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Hamilton product for ``[w, x, y, z]`` quaternions."""

    w1, x1, y1, z1 = normalize_quaternion(first)
    w2, x2, y2, z2 = normalize_quaternion(second)
    return normalize_quaternion(
        np.asarray(
            [
                (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2),
                (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2),
                (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2),
                (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2),
            ],
            dtype=np.float64,
        )
    )


def quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    """Return the inverse of a unit ``[w, x, y, z]`` quaternion."""

    w, x, y, z = normalize_quaternion(quaternion)
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def quaternion_angle(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest angular distance between two quaternions in radians."""

    start = normalize_quaternion(first)
    end = normalize_quaternion(second)
    dot = abs(float(np.dot(start, end)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def quaternion_nlerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Normalized quaternion interpolation with shortest-path sign handling."""

    safe_alpha = float(np.clip(alpha, 0.0, 1.0))
    start = normalize_quaternion(first)
    end = normalize_quaternion(second)
    if float(np.dot(start, end)) < 0.0:
        end = -end
    return normalize_quaternion(((1.0 - safe_alpha) * start) + (safe_alpha * end))


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
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"landmarks must have shape [21, 3], got {points.shape}.")
    return np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)


def _landmark_point_to_control_position(point: np.ndarray) -> np.ndarray:
    centered = np.asarray([point[0] - 0.5, point[1] - 0.5, point[2]], dtype=np.float64)
    return np.asarray(_LANDMARK_TO_CONTROL_AXES @ centered, dtype=np.float64)


def _select_position_anchor(points: np.ndarray, position_source: PositionSource) -> np.ndarray:
    if position_source == "wrist":
        return points[0]
    if position_source == "palm_center":
        return np.mean(points[list(_PALM_LANDMARKS)], axis=0)
    raise ValueError("position_source must be 'wrist' or 'palm_center'.")


def _canonicalize_orientation_vector(
    vector: np.ndarray,
    handedness: str | None,
) -> np.ndarray:
    """Mirror left-hand orientation vectors into the right-hand robot convention."""

    if handedness is not None and handedness.casefold() == "left":
        return np.asarray([-vector[0], vector[1], vector[2]], dtype=np.float64)
    return np.asarray(vector, dtype=np.float64)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON or not np.all(np.isfinite(vector)):
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vector / norm, dtype=np.float64)


def _is_valid_axis(axis: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(axis)) and abs(float(np.linalg.norm(axis)) - 1.0) < 1e-4)


def _ema_vector(previous: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray((alpha * current) + ((1.0 - alpha) * previous), dtype=np.float64)


def _ema_scalar(previous: float, current: float, alpha: float) -> float:
    return float((alpha * current) + ((1.0 - alpha) * previous))


def _coerce_vector(value: np.ndarray, length: int, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    expected_shape = (length,)
    if vector.shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {expected_shape}, got {vector.shape}.")
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _coerce_hand_scale_source(value: str) -> HandScaleSource:
    if value not in ("palm_width", "robust_palm_scale"):
        raise ValueError("depth_source must be 'palm_width' or 'robust_palm_scale'.")
    return value  # type: ignore[return-value]


def _finite_distance_2d(first: np.ndarray, second: np.ndarray) -> float:
    first_xy = np.asarray(first[:2], dtype=np.float64)
    second_xy = np.asarray(second[:2], dtype=np.float64)
    if not np.all(np.isfinite(first_xy)) or not np.all(np.isfinite(second_xy)):
        return 0.0
    return float(np.linalg.norm(first_xy - second_xy))


def _coerce_nonnegative_scalar(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(0.0, value))
