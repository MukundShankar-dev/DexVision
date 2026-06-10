"""Map normalized human finger curls to robot hand target controls."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexvision.features.hand_features import FINGER_CURL_FIELDS, HandFeatures, no_hand_features


class CurlRetargeterError(ValueError):
    """Raised when curl retargeting config or input is invalid."""


@dataclass(frozen=True)
class TargetLimit:
    """Inclusive scalar control range for one robot target."""

    minimum: float
    maximum: float

    def clip(self, value: float) -> float:
        """Return ``value`` clipped into this limit range."""

        if value < self.minimum:
            return self.minimum
        if value > self.maximum:
            return self.maximum
        return value


@dataclass(frozen=True)
class CurlTarget:
    """One robot target driven by a normalized finger curl."""

    name: str
    open_value: float
    closed_value: float
    limit: TargetLimit

    def map_curl(self, curl: float) -> float:
        """Interpolate from open to closed target and clip to limits."""

        target = self.open_value + (self.closed_value - self.open_value) * curl
        return self.limit.clip(target)


@dataclass(frozen=True)
class StaticTarget:
    """One robot target that does not depend on finger curl."""

    name: str
    value: float
    limit: TargetLimit

    def clipped_value(self) -> float:
        """Return the configured value clipped to limits."""

        return self.limit.clip(self.value)


@dataclass(frozen=True)
class FingerCurlMapping:
    """Mapping from a ``HandFeatures`` curl field to robot targets."""

    name: str
    feature: str
    scale: float
    offset: float
    targets: tuple[CurlTarget, ...]


@dataclass(frozen=True)
class CurlRetargeterConfig:
    """Validated curl retargeter configuration."""

    min_confidence: float
    static_targets: tuple[StaticTarget, ...]
    fingers: tuple[FingerCurlMapping, ...]


class CurlRetargeter:
    """Convert ``HandFeatures`` finger curls to robot target dictionaries.

    The output keys are the configured robot control target names. For the
    Level 1 Shadow Hand model, these names are MuJoCo actuator names accepted by
    ``MujocoEnv.set_joint_targets``.
    """

    def __init__(self, config: CurlRetargeterConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "CurlRetargeter":
        """Load a retargeter from a YAML file."""

        return cls.from_mapping(load_curl_retargeter_config(config_path))

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any]) -> "CurlRetargeter":
        """Build a retargeter from a parsed config mapping."""

        return cls(_coerce_config(raw_config))

    def map(
        self,
        features_or_landmarks: HandFeatures | None,
        robot_state: object | None = None,
    ) -> dict[str, float]:
        """Map hand features to clipped robot targets.

        Args:
            features_or_landmarks: ``HandFeatures`` for one frame. ``None`` is
                treated as a missing hand and maps to the configured open pose.
            robot_state: Reserved for the module contract; unused by this
                stateless curl retargeter.

        Returns:
            A new ``dict`` of robot target name to scalar target value.
        """

        del robot_state
        features = _sanitize_features(features_or_landmarks)
        low_confidence = features.confidence < self.config.min_confidence

        targets = {
            static_target.name: static_target.clipped_value()
            for static_target in self.config.static_targets
        }
        for finger in self.config.fingers:
            raw_curl = 0.0 if low_confidence else getattr(features, finger.feature)
            curl = _clip01(finger.offset + finger.scale * _clip01(raw_curl))
            for target in finger.targets:
                targets[target.name] = target.map_curl(curl)
        return targets


def load_curl_retargeter_config(config_path: str | Path) -> Mapping[str, Any]:
    """Load a curl-retargeter config file.

    Full YAML is accepted when PyYAML is installed. The bundled config remains
    JSON-compatible YAML so this loader also works in minimal environments.
    """

    path = Path(config_path)
    if not path.exists():
        raise CurlRetargeterError(f"Curl retargeter config does not exist: {path}")
    if not path.is_file():
        raise CurlRetargeterError(f"Curl retargeter config path is not a file: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        try:
            raw_config = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CurlRetargeterError(
                "PyYAML is not installed, and the curl retargeter config is not "
                f"JSON-compatible YAML: {path}. Install PyYAML or use the bundled "
                "config format."
            ) from exc
    else:
        try:
            raw_config = yaml.safe_load(text)
        except Exception as exc:  # pragma: no cover - exact PyYAML exception varies.
            raise CurlRetargeterError(
                f"Failed to parse curl retargeter config '{path}': {exc}"
            ) from exc

    if not isinstance(raw_config, Mapping):
        raise CurlRetargeterError(f"Curl retargeter config must be a mapping: {path}")
    return raw_config


def _coerce_config(raw_config: Mapping[str, Any]) -> CurlRetargeterConfig:
    raw_retargeting = raw_config.get("retargeting", raw_config)
    if not isinstance(raw_retargeting, Mapping):
        raise CurlRetargeterError("Curl retargeter config must contain a 'retargeting' mapping.")

    retargeter_type = raw_retargeting.get("type", "curl")
    if retargeter_type != "curl":
        raise CurlRetargeterError(f"Unsupported retargeter type: {retargeter_type!r}.")

    min_confidence = _coerce_float(
        raw_retargeting.get("min_confidence", 0.0),
        field_name="min_confidence",
    )
    if not 0.0 <= min_confidence <= 1.0:
        raise CurlRetargeterError("min_confidence must be in [0.0, 1.0].")

    static_targets = _coerce_static_targets(raw_retargeting.get("static_targets", {}))
    fingers = _coerce_fingers(raw_retargeting.get("fingers"))
    _validate_unique_target_names(static_targets, fingers)

    return CurlRetargeterConfig(
        min_confidence=min_confidence,
        static_targets=static_targets,
        fingers=fingers,
    )


def _coerce_static_targets(raw_targets: object) -> tuple[StaticTarget, ...]:
    if not isinstance(raw_targets, Mapping):
        raise CurlRetargeterError("static_targets must be a mapping.")

    targets: list[StaticTarget] = []
    for raw_name, raw_target in raw_targets.items():
        name = _coerce_name(raw_name, field_name="static target name")
        if isinstance(raw_target, int | float):
            value = _coerce_float(raw_target, field_name=f"static_targets.{name}")
            limit = TargetLimit(value, value)
        elif isinstance(raw_target, Mapping):
            value = _coerce_float(
                raw_target.get("value"),
                field_name=f"static_targets.{name}.value",
            )
            limit = _coerce_limit(raw_target, field_name=f"static_targets.{name}")
        else:
            raise CurlRetargeterError(
                f"static_targets.{name} must be a number or a mapping with value/min/max."
            )
        targets.append(StaticTarget(name=name, value=value, limit=limit))
    return tuple(targets)


def _coerce_fingers(raw_fingers: object) -> tuple[FingerCurlMapping, ...]:
    if not isinstance(raw_fingers, Mapping) or not raw_fingers:
        raise CurlRetargeterError("retargeting.fingers must be a non-empty mapping.")

    fingers: list[FingerCurlMapping] = []
    for raw_name, raw_mapping in raw_fingers.items():
        finger_name = _coerce_name(raw_name, field_name="finger name")
        if not isinstance(raw_mapping, Mapping):
            raise CurlRetargeterError(f"retargeting.fingers.{finger_name} must be a mapping.")

        feature = _coerce_name(
            raw_mapping.get("feature", f"{finger_name}_curl"),
            field_name=f"retargeting.fingers.{finger_name}.feature",
        )
        if feature not in FINGER_CURL_FIELDS:
            allowed = ", ".join(FINGER_CURL_FIELDS)
            raise CurlRetargeterError(
                f"retargeting.fingers.{finger_name}.feature must be one of: {allowed}."
            )

        scale = _coerce_float(
            raw_mapping.get("scale", 1.0),
            field_name=f"retargeting.fingers.{finger_name}.scale",
        )
        offset = _coerce_float(
            raw_mapping.get("offset", 0.0),
            field_name=f"retargeting.fingers.{finger_name}.offset",
        )
        raw_targets = raw_mapping.get("targets")
        if not isinstance(raw_targets, Mapping) or not raw_targets:
            raise CurlRetargeterError(
                f"retargeting.fingers.{finger_name}.targets must be a non-empty mapping."
            )

        targets: list[CurlTarget] = []
        for raw_target_name, raw_target in raw_targets.items():
            target_name = _coerce_name(
                raw_target_name,
                field_name=f"retargeting.fingers.{finger_name}.targets name",
            )
            if not isinstance(raw_target, Mapping):
                raise CurlRetargeterError(
                    f"retargeting.fingers.{finger_name}.targets.{target_name} must be a mapping."
                )
            open_value = _coerce_float(
                raw_target.get("open"),
                field_name=f"retargeting.fingers.{finger_name}.targets.{target_name}.open",
            )
            closed_value = _coerce_float(
                raw_target.get("closed"),
                field_name=f"retargeting.fingers.{finger_name}.targets.{target_name}.closed",
            )
            limit = _coerce_limit(
                raw_target,
                field_name=f"retargeting.fingers.{finger_name}.targets.{target_name}",
            )
            targets.append(
                CurlTarget(
                    name=target_name,
                    open_value=open_value,
                    closed_value=closed_value,
                    limit=limit,
                )
            )

        fingers.append(
            FingerCurlMapping(
                name=finger_name,
                feature=feature,
                scale=scale,
                offset=offset,
                targets=tuple(targets),
            )
        )
    return tuple(fingers)


def _coerce_limit(raw_mapping: Mapping[str, Any], *, field_name: str) -> TargetLimit:
    minimum = _coerce_float(raw_mapping.get("min"), field_name=f"{field_name}.min")
    maximum = _coerce_float(raw_mapping.get("max"), field_name=f"{field_name}.max")
    if minimum > maximum:
        raise CurlRetargeterError(f"{field_name}.min must be <= {field_name}.max.")
    return TargetLimit(minimum=minimum, maximum=maximum)


def _validate_unique_target_names(
    static_targets: tuple[StaticTarget, ...],
    fingers: tuple[FingerCurlMapping, ...],
) -> None:
    seen: set[str] = set()
    for target in static_targets:
        if target.name in seen:
            raise CurlRetargeterError(f"Duplicate retargeting target name: {target.name}")
        seen.add(target.name)

    for finger in fingers:
        for target in finger.targets:
            if target.name in seen:
                raise CurlRetargeterError(f"Duplicate retargeting target name: {target.name}")
            seen.add(target.name)


def _sanitize_features(features: HandFeatures | None) -> HandFeatures:
    if features is None:
        return no_hand_features()
    return HandFeatures(
        thumb_curl=_clip01(features.thumb_curl),
        index_curl=_clip01(features.index_curl),
        middle_curl=_clip01(features.middle_curl),
        ring_curl=_clip01(features.ring_curl),
        pinky_curl=_clip01(features.pinky_curl),
        pinch_thumb_index=_clip01(features.pinch_thumb_index),
        palm_roll_proxy=_clip_signed(features.palm_roll_proxy),
        palm_pitch_proxy=_clip_signed(features.palm_pitch_proxy),
        confidence=_clip01(features.confidence),
    )


def _coerce_name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CurlRetargeterError(f"{field_name} must be a non-empty string.")
    return value


def _coerce_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, int | float):
        raise CurlRetargeterError(f"{field_name} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise CurlRetargeterError(f"{field_name} must be finite.")
    return number


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _clip_signed(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(-1.0, float(value)))
