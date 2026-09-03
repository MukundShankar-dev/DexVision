"""Approximate fingertip-target retargeting in a normalized palm frame."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Final

import numpy as np

from dexvision.retargeting.curl_retargeter import (
    CurlRetargeter,
    CurlRetargeterConfig,
    load_curl_retargeter_config,
)


FINGERTIP_NAMES: Final[tuple[str, ...]] = (
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
)
_FINGER_CHAINS: Final[dict[str, tuple[int, int, int, int]]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_EPSILON: Final[float] = 1e-8


class FingertipRetargeterError(ValueError):
    """Raised when fingertip targets or retargeter configuration are invalid."""


@dataclass(frozen=True)
class _NormalizedHandGeometry:
    """Palm-local geometry used by the approximation solve."""

    fingertips: np.ndarray  # [5, 3], ordered by FINGERTIP_NAMES
    finger_bases: np.ndarray  # [5, 3], ordered by FINGERTIP_NAMES
    chain_lengths: np.ndarray  # [5], normalized by palm width


def compute_normalized_fingertip_targets(landmarks: np.ndarray) -> np.ndarray:
    """Return palm-local, palm-width-normalized fingertip targets.

    Args:
        landmarks: MediaPipe-compatible hand landmarks with shape ``[21, 3]``.

    Returns:
        A finite ``float64`` array with shape ``[5, 3]`` in
        ``(thumb, index, middle, ring, pinky)`` order. The axes are palm-local:
        x crosses the palm from pinky to index, y points from wrist toward the
        fingers, and z is the palm normal.
    """

    return _normalized_hand_geometry(landmarks).fingertips.copy()


class FingertipIKRetargeter:
    """Map normalized human fingertip targets to bounded robot controls.

    This intentionally uses an inexpensive geometric approximation rather than
    a numerical optimizer. Long-finger bend comes from fingertip travel along
    the palm y-axis relative to that finger's chain length. Thumb bend comes
    from the CMC-to-tip distance relative to thumb chain length.

    If target extraction or solving fails, :meth:`map` returns the last valid
    robot targets. Before the first successful solve it returns a safe open-hand
    target set. The fallback output uses the same configured target limits.
    """

    def __init__(self, config: CurlRetargeterConfig) -> None:
        self.config = config
        unsupported = sorted(
            finger.name for finger in config.fingers if finger.name not in FINGERTIP_NAMES
        )
        if unsupported:
            names = ", ".join(unsupported)
            raise FingertipRetargeterError(
                f"Unsupported fingertip mapping names: {names}."
            )

        self._safe_open_targets = self._map_controls(
            {finger: 0.0 for finger in FINGERTIP_NAMES}
        )
        self._last_valid_targets: dict[str, float] | None = None
        self.last_used_fallback = False
        self.last_fingertip_targets: np.ndarray | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "FingertipIKRetargeter":
        """Load target ranges from a YAML teleoperation config."""

        return cls.from_mapping(load_curl_retargeter_config(config_path))

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> "FingertipIKRetargeter":
        """Build from curl-compatible target mappings.

        The existing Level 1 configuration remains the source of robot target
        names, open/closed poses, and limits. A ``fingertip`` or
        ``fingertip_ik`` type marker is accepted without mutating the caller's
        mapping.
        """

        normalized_config = _as_curl_compatible_config(raw_config)
        return cls(CurlRetargeter.from_mapping(normalized_config).config)

    def compute_fingertip_targets(self, landmarks: np.ndarray) -> np.ndarray:
        """Compute normalized fingertip targets with shape ``[5, 3]``."""

        return compute_normalized_fingertip_targets(landmarks)

    def solve(self, landmarks: np.ndarray) -> dict[str, float]:
        """Solve one valid landmark frame without applying fallback behavior."""

        geometry = _normalized_hand_geometry(landmarks)
        controls = _bend_controls(geometry)
        targets = self._map_controls(controls)
        if not targets or not all(math.isfinite(value) for value in targets.values()):
            raise FingertipRetargeterError("Fingertip solve produced invalid robot targets.")
        self.last_fingertip_targets = geometry.fingertips.copy()
        return targets

    def map(
        self,
        features_or_landmarks: object | None,
        robot_state: object | None = None,
    ) -> dict[str, float]:
        """Map landmarks to clipped robot targets with a safe fallback.

        ``features_or_landmarks`` may be a ``[21, 3]`` landmark array or a
        hand-tracking result exposing image/world landmarks and confidence.
        ``robot_state`` is reserved by the shared retargeter contract.
        """

        del robot_state
        landmarks, confidence = _landmarks_and_confidence(features_or_landmarks)
        if landmarks is None or confidence < self.config.min_confidence:
            self.last_used_fallback = True
            return dict(self._safe_open_targets)

        try:
            targets = self.solve(landmarks)
        except Exception:
            self.last_used_fallback = True
            fallback = self._last_valid_targets or self._safe_open_targets
            return dict(fallback)

        self.last_used_fallback = False
        self._last_valid_targets = dict(targets)
        return dict(targets)

    def fallback_targets(self) -> dict[str, float]:
        """Return the current last-valid or safe-open fallback targets."""

        return dict(self._last_valid_targets or self._safe_open_targets)

    def _map_controls(self, controls: Mapping[str, float]) -> dict[str, float]:
        targets = {
            target.name: target.clipped_value()
            for target in self.config.static_targets
        }
        for finger in self.config.fingers:
            control = _clip01(float(controls[finger.name]))
            for target in finger.targets:
                targets[target.name] = target.map_control(control)
        return targets


# A concise alias is useful to callers that do not distinguish this baseline
# from future numerical IK implementations.
FingertipRetargeter = FingertipIKRetargeter


def _normalized_hand_geometry(landmarks: np.ndarray) -> _NormalizedHandGeometry:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise FingertipRetargeterError(
            f"landmarks must have shape [21, 3], got {points.shape}."
        )
    if not np.all(np.isfinite(points)):
        raise FingertipRetargeterError("landmarks must contain only finite values.")

    origin = points[0]
    across_palm = points[5] - points[17]
    toward_fingers = points[9] - origin
    palm_width = float(np.linalg.norm(across_palm))
    if palm_width <= _EPSILON:
        raise FingertipRetargeterError("Cannot normalize a hand with zero palm width.")

    x_axis = _unit(across_palm, name="palm x-axis")
    z_axis = _unit(np.cross(x_axis, toward_fingers), name="palm normal")
    y_axis = _unit(np.cross(z_axis, x_axis), name="palm y-axis")
    axes = np.stack((x_axis, y_axis, z_axis), axis=1)
    palm_local = ((points - origin) @ axes) / palm_width

    fingertips = np.empty((len(FINGERTIP_NAMES), 3), dtype=np.float64)
    finger_bases = np.empty_like(fingertips)
    chain_lengths = np.empty(len(FINGERTIP_NAMES), dtype=np.float64)
    for finger_index, finger_name in enumerate(FINGERTIP_NAMES):
        chain = _FINGER_CHAINS[finger_name]
        fingertips[finger_index] = palm_local[chain[-1]]
        finger_bases[finger_index] = palm_local[chain[0]]
        chain_length = sum(
            float(np.linalg.norm(points[end] - points[start]))
            for start, end in zip(chain[:-1], chain[1:], strict=True)
        )
        chain_lengths[finger_index] = chain_length / palm_width

    if np.any(chain_lengths <= _EPSILON):
        raise FingertipRetargeterError("Each finger must have a non-zero chain length.")
    if not (
        np.all(np.isfinite(fingertips))
        and np.all(np.isfinite(finger_bases))
        and np.all(np.isfinite(chain_lengths))
    ):
        raise FingertipRetargeterError("Normalized fingertip geometry must be finite.")

    return _NormalizedHandGeometry(
        fingertips=fingertips,
        finger_bases=finger_bases,
        chain_lengths=chain_lengths,
    )


def _bend_controls(geometry: _NormalizedHandGeometry) -> dict[str, float]:
    controls: dict[str, float] = {}
    for finger_index, finger_name in enumerate(FINGERTIP_NAMES):
        tip = geometry.fingertips[finger_index]
        base = geometry.finger_bases[finger_index]
        chain_length = float(geometry.chain_lengths[finger_index])
        if finger_name == "thumb":
            extension = float(np.linalg.norm(tip - base)) / chain_length
        else:
            extension = float(tip[1] - base[1]) / chain_length
        controls[finger_name] = _clip01(1.0 - extension)
    return controls


def _landmarks_and_confidence(value: object | None) -> tuple[np.ndarray | None, float]:
    if value is None:
        return None, 0.0
    if isinstance(value, np.ndarray):
        return value, 1.0

    detected = getattr(value, "detected", True)
    if not bool(detected):
        return None, 0.0
    image_landmarks = getattr(value, "image_landmarks", None)
    world_landmarks = getattr(value, "world_landmarks", None)
    landmarks = image_landmarks if image_landmarks is not None else world_landmarks
    confidence_value = getattr(value, "confidence", 1.0)
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    return landmarks, _clip01(confidence)


def _as_curl_compatible_config(raw_config: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_retargeting = raw_config.get("retargeting", raw_config)
    if not isinstance(raw_retargeting, Mapping):
        raise FingertipRetargeterError(
            "Fingertip retargeter config must contain a 'retargeting' mapping."
        )
    retargeter_type = raw_retargeting.get("type", "curl")
    if retargeter_type not in {"curl", "fingertip", "fingertip_ik"}:
        raise FingertipRetargeterError(
            f"Unsupported fingertip retargeter type: {retargeter_type!r}."
        )

    compatible_retargeting = dict(raw_retargeting)
    compatible_retargeting["type"] = "curl"
    if "retargeting" in raw_config:
        compatible_config = dict(raw_config)
        compatible_config["retargeting"] = compatible_retargeting
        return compatible_config
    return compatible_retargeting


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= _EPSILON:
        raise FingertipRetargeterError(f"Cannot compute {name} from degenerate landmarks.")
    return np.asarray(vector / norm, dtype=np.float64)


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))
