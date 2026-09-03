"""Optimization-based fingertip retargeting with bounded safe fallbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Final

import numpy as np

from dexvision.retargeting.curl_retargeter import (
    CurlRetargeter,
    CurlRetargeterConfig,
    load_curl_retargeter_config,
)
from dexvision.retargeting.fingertip_ik_retargeter import FINGERTIP_NAMES


LOGGER = logging.getLogger(__name__)
_EPSILON: Final[float] = 1e-8
_FINGER_CHAINS: Final[dict[str, tuple[int, int, int, int]]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


class OptimizationRetargeterError(ValueError):
    """Raised when optimization inputs or configuration are invalid."""


@dataclass(frozen=True)
class OptimizationWeights:
    """Non-negative weights for the Level 2.9 objective."""

    fingertip: float = 1.0
    joint_limit: float = 100.0
    smoothness: float = 0.05


@dataclass(frozen=True)
class OptimizationSolveStats:
    """Diagnostics from the most recent optimization attempt."""

    solve_time_seconds: float
    objective_value: float | None
    success: bool
    backend: str


@dataclass(frozen=True)
class _TargetGeometry:
    fingertips: np.ndarray  # [5, 3], palm-local and palm-width-normalized
    finger_bases: np.ndarray  # [5, 3], same frame as fingertips
    chain_lengths: np.ndarray  # [5], normalized by palm width


@dataclass(frozen=True)
class _OptimizerResult:
    controls: np.ndarray
    objective_value: float
    success: bool
    backend: str


class OptimizationRetargeter:
    """Optimize bounded robot finger controls against human fingertip targets.

    The decision vector contains one normalized bend control for each finger.
    A small palm-local kinematic surrogate predicts fingertip positions from
    those controls. The objective combines 3D fingertip error, penalties for
    configured actuator-limit violations, and distance from the previous
    solution. SciPy is preferred when installed; a deterministic projected
    gradient solver keeps the retargeter usable without that optional package.

    Failed solves return the last valid clipped targets, or the configured safe
    open pose before the first successful solve.
    """

    def __init__(
        self,
        config: CurlRetargeterConfig,
        *,
        weights: OptimizationWeights | None = None,
        max_iterations: int = 100,
    ) -> None:
        self.config = config
        self.weights = weights or OptimizationWeights()
        _validate_weights(self.weights)
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations <= 0
        ):
            raise OptimizationRetargeterError("max_iterations must be a positive integer.")
        self.max_iterations = int(max_iterations)

        unsupported = sorted(
            finger.name for finger in config.fingers if finger.name not in FINGERTIP_NAMES
        )
        if unsupported:
            raise OptimizationRetargeterError(
                "Unsupported optimization finger names: " + ", ".join(unsupported) + "."
            )

        self._previous_controls = np.zeros(len(FINGERTIP_NAMES), dtype=np.float64)
        self._safe_open_targets = self._map_controls(self._previous_controls)
        self._last_valid_targets: dict[str, float] | None = None
        self.last_used_fallback = False
        self.last_fingertip_targets: np.ndarray | None = None
        self.last_stats = OptimizationSolveStats(0.0, None, False, "not-run")

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "OptimizationRetargeter":
        """Load target mappings and optional optimizer settings from YAML."""

        return cls.from_mapping(load_curl_retargeter_config(config_path))

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> "OptimizationRetargeter":
        """Build from curl-compatible mappings plus an optional optimizer block."""

        raw_retargeting = raw_config.get("retargeting", raw_config)
        if not isinstance(raw_retargeting, Mapping):
            raise OptimizationRetargeterError(
                "Optimization retargeter config must contain a 'retargeting' mapping."
            )
        retargeter_type = raw_retargeting.get("type", "curl")
        if retargeter_type not in {"curl", "optimization", "optimization_ik"}:
            raise OptimizationRetargeterError(
                f"Unsupported optimization retargeter type: {retargeter_type!r}."
            )

        compatible_retargeting = dict(raw_retargeting)
        compatible_retargeting["type"] = "curl"
        if "retargeting" in raw_config:
            compatible_config = dict(raw_config)
            compatible_config["retargeting"] = compatible_retargeting
        else:
            compatible_config = compatible_retargeting
        config = CurlRetargeter.from_mapping(compatible_config).config

        raw_optimizer = raw_retargeting.get("optimization", {})
        if not isinstance(raw_optimizer, Mapping):
            raise OptimizationRetargeterError("retargeting.optimization must be a mapping.")
        weights = OptimizationWeights(
            fingertip=_finite_float(
                raw_optimizer.get("fingertip_weight", 1.0),
                name="fingertip_weight",
            ),
            joint_limit=_finite_float(
                raw_optimizer.get("joint_limit_weight", 100.0),
                name="joint_limit_weight",
            ),
            smoothness=_finite_float(
                raw_optimizer.get("smoothness_weight", 0.05),
                name="smoothness_weight",
            ),
        )
        max_iterations = raw_optimizer.get("max_iterations", 100)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise OptimizationRetargeterError("max_iterations must be a positive integer.")
        return cls(config, weights=weights, max_iterations=max_iterations)

    @property
    def last_solve_time_seconds(self) -> float:
        """Return the elapsed time recorded for the latest solve attempt."""

        return self.last_stats.solve_time_seconds

    def solve(self, landmarks: np.ndarray) -> dict[str, float]:
        """Optimize one valid ``[21, 3]`` landmark frame without fallback."""

        geometry = _target_geometry(landmarks)
        previous = self._previous_controls.copy()
        objective = self._objective(geometry, previous)
        initial = _geometric_controls(geometry)
        result = self._run_optimizer(objective, initial)
        controls = np.asarray(result.controls, dtype=np.float64)
        result_objective = (
            float(result.objective_value)
            if math.isfinite(result.objective_value)
            else None
        )
        self.last_stats = OptimizationSolveStats(
            solve_time_seconds=self.last_stats.solve_time_seconds,
            objective_value=result_objective,
            success=False,
            backend=result.backend,
        )
        if (
            not result.success
            or controls.shape != (len(FINGERTIP_NAMES),)
            or not np.all(np.isfinite(controls))
            or result_objective is None
        ):
            raise OptimizationRetargeterError("Optimization did not return a valid solution.")

        controls = np.clip(controls, 0.0, 1.0)
        targets = self._map_controls(controls)
        if not targets or not all(math.isfinite(value) for value in targets.values()):
            raise OptimizationRetargeterError("Optimization produced invalid robot targets.")
        self._previous_controls = controls
        self.last_fingertip_targets = geometry.fingertips.copy()
        self.last_stats = OptimizationSolveStats(
            solve_time_seconds=self.last_stats.solve_time_seconds,
            objective_value=result_objective,
            success=True,
            backend=result.backend,
        )
        return targets

    def map(
        self,
        features_or_landmarks: object | None,
        robot_state: object | None = None,
    ) -> dict[str, float]:
        """Map landmarks to bounded targets, timing and logging every solve."""

        del robot_state
        landmarks, confidence = _landmarks_and_confidence(features_or_landmarks)
        if landmarks is None or confidence < self.config.min_confidence:
            self.last_used_fallback = True
            return dict(self._safe_open_targets)

        started = perf_counter()
        try:
            targets = self.solve(landmarks)
        except Exception as exc:
            elapsed = perf_counter() - started
            backend = self.last_stats.backend
            objective_value = self.last_stats.objective_value
            self.last_stats = OptimizationSolveStats(
                elapsed,
                objective_value,
                False,
                backend,
            )
            self.last_used_fallback = True
            LOGGER.warning(
                "Optimization retargeting failed after %.6f s; using fallback: %s",
                elapsed,
                exc,
            )
            return dict(self._last_valid_targets or self._safe_open_targets)

        elapsed = perf_counter() - started
        self.last_stats = OptimizationSolveStats(
            elapsed,
            self.last_stats.objective_value,
            True,
            self.last_stats.backend,
        )
        self.last_used_fallback = False
        self._last_valid_targets = dict(targets)
        LOGGER.info(
            "Optimization retargeting solve time: %.6f s (backend=%s, objective=%.6g)",
            elapsed,
            self.last_stats.backend,
            self.last_stats.objective_value,
        )
        return dict(targets)

    def fallback_targets(self) -> dict[str, float]:
        """Return the last valid targets, or the safe open pose if unavailable."""

        return dict(self._last_valid_targets or self._safe_open_targets)

    def _objective(
        self,
        geometry: _TargetGeometry,
        previous: np.ndarray,
    ) -> Callable[[np.ndarray], float]:
        def objective(candidate: np.ndarray) -> float:
            controls = np.asarray(candidate, dtype=np.float64)
            if controls.shape != previous.shape or not np.all(np.isfinite(controls)):
                return math.inf
            predicted = _predict_fingertips(geometry, controls)
            fingertip_error = float(
                np.sum(np.square(predicted - geometry.fingertips))
            )
            smoothness = float(np.sum(np.square(controls - previous)))
            limit_penalty = self._joint_limit_penalty(controls)
            return (
                self.weights.fingertip * fingertip_error
                + self.weights.joint_limit * limit_penalty
                + self.weights.smoothness * smoothness
            )

        return objective

    def _joint_limit_penalty(self, controls: np.ndarray) -> float:
        controls_by_name = dict(zip(FINGERTIP_NAMES, controls, strict=True))
        penalty = 0.0
        for finger in self.config.fingers:
            control = float(controls_by_name[finger.name])
            for target in finger.targets:
                raw_value = target.open_value + (
                    target.closed_value - target.open_value
                ) * control
                below = max(0.0, target.limit.minimum - raw_value)
                above = max(0.0, raw_value - target.limit.maximum)
                penalty += below * below + above * above
        return penalty

    def _run_optimizer(
        self,
        objective: Callable[[np.ndarray], float],
        initial: np.ndarray,
    ) -> _OptimizerResult:
        try:
            from scipy.optimize import minimize
        except ImportError:
            return _projected_gradient_minimize(
                objective,
                initial,
                max_iterations=self.max_iterations,
            )

        result = minimize(
            objective,
            np.clip(initial, 0.0, 1.0),
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * len(FINGERTIP_NAMES),
            options={"maxiter": self.max_iterations},
        )
        return _OptimizerResult(
            controls=np.asarray(result.x, dtype=np.float64),
            objective_value=float(result.fun),
            success=bool(result.success),
            backend="scipy",
        )

    def _map_controls(self, controls: np.ndarray) -> dict[str, float]:
        controls_by_name = dict(zip(FINGERTIP_NAMES, controls, strict=True))
        targets = {
            target.name: target.clipped_value() for target in self.config.static_targets
        }
        for finger in self.config.fingers:
            control = float(np.clip(controls_by_name[finger.name], 0.0, 1.0))
            for target in finger.targets:
                targets[target.name] = target.map_control(control)
        return targets


def _target_geometry(landmarks: np.ndarray) -> _TargetGeometry:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise OptimizationRetargeterError(
            f"landmarks must have shape [21, 3], got {points.shape}."
        )
    if not np.all(np.isfinite(points)):
        raise OptimizationRetargeterError("landmarks must contain only finite values.")

    origin = points[0]
    across = points[5] - points[17]
    toward = points[9] - origin
    palm_width = float(np.linalg.norm(across))
    if palm_width <= _EPSILON:
        raise OptimizationRetargeterError("Cannot normalize a hand with zero palm width.")
    x_axis = _unit(across, name="palm x-axis")
    z_axis = _unit(np.cross(x_axis, toward), name="palm normal")
    y_axis = _unit(np.cross(z_axis, x_axis), name="palm y-axis")
    axes = np.stack((x_axis, y_axis, z_axis), axis=1)
    local = ((points - origin) @ axes) / palm_width

    fingertips = np.empty((len(FINGERTIP_NAMES), 3), dtype=np.float64)
    bases = np.empty_like(fingertips)
    lengths = np.empty(len(FINGERTIP_NAMES), dtype=np.float64)
    for index, finger_name in enumerate(FINGERTIP_NAMES):
        chain = _FINGER_CHAINS[finger_name]
        fingertips[index] = local[chain[-1]]
        bases[index] = local[chain[0]]
        lengths[index] = sum(
            float(np.linalg.norm(points[end] - points[start]))
            for start, end in zip(chain[:-1], chain[1:], strict=True)
        ) / palm_width
    if np.any(lengths <= _EPSILON) or not np.all(np.isfinite(lengths)):
        raise OptimizationRetargeterError("Each finger must have a finite chain length.")
    return _TargetGeometry(fingertips, bases, lengths)


def _predict_fingertips(
    geometry: _TargetGeometry,
    controls: np.ndarray,
) -> np.ndarray:
    predicted = geometry.finger_bases.copy()
    for index, finger_name in enumerate(FINGERTIP_NAMES):
        extension = 1.0 - float(controls[index])
        length = float(geometry.chain_lengths[index])
        if finger_name == "thumb":
            direction = geometry.fingertips[index] - geometry.finger_bases[index]
            norm = float(np.linalg.norm(direction))
            if norm <= _EPSILON:
                direction = np.asarray([1.0, 0.0, 0.0])
            else:
                direction = direction / norm
        else:
            direction = np.asarray([0.0, 1.0, 0.0])
        predicted[index] += direction * length * extension
    return predicted


def _geometric_controls(geometry: _TargetGeometry) -> np.ndarray:
    controls = np.empty(len(FINGERTIP_NAMES), dtype=np.float64)
    for index, finger_name in enumerate(FINGERTIP_NAMES):
        displacement = geometry.fingertips[index] - geometry.finger_bases[index]
        if finger_name == "thumb":
            extension = float(np.linalg.norm(displacement))
        else:
            extension = float(displacement[1])
        controls[index] = np.clip(
            1.0 - extension / float(geometry.chain_lengths[index]),
            0.0,
            1.0,
        )
    return controls


def _projected_gradient_minimize(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    *,
    max_iterations: int,
) -> _OptimizerResult:
    controls = np.clip(np.asarray(initial, dtype=np.float64), 0.0, 1.0)
    value = float(objective(controls))
    if not math.isfinite(value):
        return _OptimizerResult(controls, value, False, "projected-gradient")

    step_size = 0.25
    epsilon = 1e-6
    for _ in range(max_iterations):
        gradient = np.empty_like(controls)
        for index in range(controls.size):
            upper = controls.copy()
            lower = controls.copy()
            upper[index] = min(1.0, upper[index] + epsilon)
            lower[index] = max(0.0, lower[index] - epsilon)
            span = upper[index] - lower[index]
            gradient[index] = 0.0 if span <= 0.0 else (
                objective(upper) - objective(lower)
            ) / span
        candidate = np.clip(controls - step_size * gradient, 0.0, 1.0)
        candidate_value = float(objective(candidate))
        if math.isfinite(candidate_value) and candidate_value < value:
            if np.linalg.norm(candidate - controls) <= 1e-8:
                controls = candidate
                value = candidate_value
                break
            controls = candidate
            value = candidate_value
            step_size = min(0.5, step_size * 1.05)
        else:
            step_size *= 0.5
            if step_size <= 1e-8:
                break
    return _OptimizerResult(controls, value, math.isfinite(value), "projected-gradient")


def _landmarks_and_confidence(value: object | None) -> tuple[np.ndarray | None, float]:
    if value is None:
        return None, 0.0
    if isinstance(value, np.ndarray):
        return value, 1.0
    if not bool(getattr(value, "detected", True)):
        return None, 0.0
    image_landmarks = getattr(value, "image_landmarks", None)
    world_landmarks = getattr(value, "world_landmarks", None)
    landmarks = image_landmarks if image_landmarks is not None else world_landmarks
    try:
        confidence = float(getattr(value, "confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    return landmarks, min(1.0, max(0.0, confidence))


def _validate_weights(weights: OptimizationWeights) -> None:
    for name, value in (
        ("fingertip", weights.fingertip),
        ("joint_limit", weights.joint_limit),
        ("smoothness", weights.smoothness),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise OptimizationRetargeterError(
                f"Optimization weight '{name}' must be finite and non-negative."
            )
    if weights.fingertip == 0.0:
        raise OptimizationRetargeterError("fingertip weight must be greater than zero.")


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise OptimizationRetargeterError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OptimizationRetargeterError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise OptimizationRetargeterError(f"{name} must be a finite number.")
    return result


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= _EPSILON:
        raise OptimizationRetargeterError(f"Cannot compute {name} from landmarks.")
    return np.asarray(vector / norm, dtype=np.float64)
