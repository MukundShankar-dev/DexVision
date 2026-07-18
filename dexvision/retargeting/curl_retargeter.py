"""Map normalized human finger control signals to robot hand target controls."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexvision.features.hand_features import (
    FINGER_CONTROL_FIELDS,
    FingerState,
    HandFeatures,
    NON_THUMB_FINGER_NAMES,
    no_hand_features,
)


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
    """One robot target driven by a normalized finger control value."""

    name: str
    open_value: float
    closed_value: float
    limit: TargetLimit

    def map_curl(self, curl: float) -> float:
        """Interpolate from open to closed target and clip to limits."""

        target = self.open_value + (self.closed_value - self.open_value) * curl
        return self.limit.clip(target)

    def map_control(self, control: float) -> float:
        """Interpolate from open to closed target and clip to limits."""

        return self.map_curl(control)


@dataclass(frozen=True)
class PinchOverlayTarget:
    """One target blended toward a pinch-specific value."""

    name: str
    closed_value: float
    limit: TargetLimit

    def blend(self, current_value: float, control: float) -> float:
        """Blend from the current retargeted value toward the pinch value."""

        clipped_current = self.limit.clip(current_value)
        target = clipped_current + (self.closed_value - clipped_current) * control
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
    """Mapping from a ``HandFeatures`` control field to robot targets."""

    name: str
    feature: str
    scale: float
    offset: float
    targets: tuple[CurlTarget, ...]


@dataclass(frozen=True)
class PinchOverlay:
    """Optional thumb-index overlay driven by thumb/index tip distance."""

    activation_below: float
    closed_at: float
    min_index_bend: float
    max_other_bend: float
    targets: tuple[PinchOverlayTarget, ...]

    def control_from_features(self, features: HandFeatures) -> float:
        """Return pinch control when the hand shape looks like a pinch."""

        if features.index_bend < self.min_index_bend:
            return 0.0
        other_bend = max(features.middle_bend, features.ring_bend, features.pinky_bend)
        if other_bend > self.max_other_bend:
            return 0.0
        return self.control_from_distance(features.pinch_thumb_index)

    def control_from_distance(self, pinch_thumb_index: float) -> float:
        """Return 0 for non-pinches and 1 near a full thumb-index pinch."""

        distance = _clip01(pinch_thumb_index)
        if distance >= self.activation_below:
            return 0.0
        if distance <= self.closed_at:
            return 1.0
        span = self.activation_below - self.closed_at
        if span <= 0.0:
            return 1.0
        return _clip01((self.activation_below - distance) / span)


@dataclass(frozen=True)
class CurlRetargeterConfig:
    """Validated curl retargeter configuration."""

    min_confidence: float
    static_targets: tuple[StaticTarget, ...]
    fingers: tuple[FingerCurlMapping, ...]
    pinch_overlay: PinchOverlay | None = None


class CurlRetargeter:
    """Convert ``HandFeatures`` finger control signals to robot target dictionaries.

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
            raw_control = 0.0 if low_confidence else _feature_value(features, finger.feature)
            control = _clip01(finger.offset + finger.scale * _clip01(raw_control))
            for target in finger.targets:
                targets[target.name] = target.map_control(control)
        if self.config.pinch_overlay is not None and not low_confidence and features.confidence > 0.0:
            pinch_control = self.config.pinch_overlay.control_from_features(features)
            if pinch_control > 0.0:
                for target in self.config.pinch_overlay.targets:
                    targets[target.name] = target.blend(targets[target.name], pinch_control)
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
    pinch_overlay = _coerce_pinch_overlay(
        raw_retargeting.get("pinch_overlay"),
        known_target_names=_target_names(static_targets, fingers),
    )

    return CurlRetargeterConfig(
        min_confidence=min_confidence,
        static_targets=static_targets,
        fingers=fingers,
        pinch_overlay=pinch_overlay,
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

        default_feature = (
            f"{finger_name}_bend"
            if finger_name in NON_THUMB_FINGER_NAMES
            else f"{finger_name}_curl"
        )
        feature = _coerce_name(
            raw_mapping.get("feature", default_feature),
            field_name=f"retargeting.fingers.{finger_name}.feature",
        )
        if feature not in FINGER_CONTROL_FIELDS:
            allowed = ", ".join(FINGER_CONTROL_FIELDS)
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


def _coerce_pinch_overlay(
    raw_overlay: object,
    *,
    known_target_names: set[str],
) -> PinchOverlay | None:
    if raw_overlay is None:
        return None
    if not isinstance(raw_overlay, Mapping):
        raise CurlRetargeterError("retargeting.pinch_overlay must be a mapping.")

    feature = _coerce_name(
        raw_overlay.get("feature", "pinch_thumb_index"),
        field_name="retargeting.pinch_overlay.feature",
    )
    if feature != "pinch_thumb_index":
        raise CurlRetargeterError("retargeting.pinch_overlay.feature must be pinch_thumb_index.")

    activation_below = _coerce_float(
        raw_overlay.get("activation_below", 0.45),
        field_name="retargeting.pinch_overlay.activation_below",
    )
    closed_at = _coerce_float(
        raw_overlay.get("closed_at", 0.30),
        field_name="retargeting.pinch_overlay.closed_at",
    )
    if not 0.0 <= closed_at < activation_below <= 1.0:
        raise CurlRetargeterError(
            "retargeting.pinch_overlay must satisfy 0 <= closed_at < activation_below <= 1."
        )
    min_index_bend = _coerce_float(
        raw_overlay.get("min_index_bend", 0.0),
        field_name="retargeting.pinch_overlay.min_index_bend",
    )
    max_other_bend = _coerce_float(
        raw_overlay.get("max_other_bend", 1.0),
        field_name="retargeting.pinch_overlay.max_other_bend",
    )
    if not 0.0 <= min_index_bend <= 1.0:
        raise CurlRetargeterError("retargeting.pinch_overlay.min_index_bend must be in [0, 1].")
    if not 0.0 <= max_other_bend <= 1.0:
        raise CurlRetargeterError("retargeting.pinch_overlay.max_other_bend must be in [0, 1].")

    raw_targets = raw_overlay.get("targets")
    if not isinstance(raw_targets, Mapping) or not raw_targets:
        raise CurlRetargeterError("retargeting.pinch_overlay.targets must be a non-empty mapping.")

    targets: list[PinchOverlayTarget] = []
    seen: set[str] = set()
    for raw_name, raw_target in raw_targets.items():
        name = _coerce_name(raw_name, field_name="retargeting.pinch_overlay.targets name")
        if name in seen:
            raise CurlRetargeterError(f"Duplicate pinch_overlay target name: {name}")
        seen.add(name)
        if name not in known_target_names:
            raise CurlRetargeterError(
                f"retargeting.pinch_overlay target {name!r} is not a configured target."
            )
        if not isinstance(raw_target, Mapping):
            raise CurlRetargeterError(
                f"retargeting.pinch_overlay.targets.{name} must be a mapping."
            )
        closed_value = _coerce_float(
            raw_target.get("closed"),
            field_name=f"retargeting.pinch_overlay.targets.{name}.closed",
        )
        limit = _coerce_limit(
            raw_target,
            field_name=f"retargeting.pinch_overlay.targets.{name}",
        )
        targets.append(
            PinchOverlayTarget(
                name=name,
                closed_value=closed_value,
                limit=limit,
            )
        )

    return PinchOverlay(
        activation_below=activation_below,
        closed_at=closed_at,
        min_index_bend=min_index_bend,
        max_other_bend=max_other_bend,
        targets=tuple(targets),
    )


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


def _target_names(
    static_targets: tuple[StaticTarget, ...],
    fingers: tuple[FingerCurlMapping, ...],
) -> set[str]:
    names = {target.name for target in static_targets}
    for finger in fingers:
        names.update(target.name for target in finger.targets)
    return names


def _sanitize_features(features: HandFeatures | None) -> HandFeatures:
    if features is None:
        return no_hand_features()
    return HandFeatures(
        thumb=_sanitize_named_finger(features, "thumb"),
        index=_sanitize_named_finger(features, "index"),
        middle=_sanitize_named_finger(features, "middle"),
        ring=_sanitize_named_finger(features, "ring"),
        pinky=_sanitize_named_finger(features, "pinky"),
        palm=getattr(features, "palm", None),
        pinch_thumb_index=_clip_pinch_distance(
            float(getattr(features, "pinch_thumb_index", 1.0))
        ),
        palm_roll_proxy=_clip_signed(float(getattr(features, "palm_roll_proxy", 0.0))),
        palm_pitch_proxy=_clip_signed(float(getattr(features, "palm_pitch_proxy", 0.0))),
        confidence=_clip01(float(getattr(features, "confidence", 0.0))),
    )


def _sanitize_named_finger(features: object, finger: str) -> FingerState:
    state = getattr(features, finger, None)
    if isinstance(state, FingerState):
        return _sanitize_finger_state(state)

    curl = _clip01(float(getattr(features, f"{finger}_curl", 0.0)))
    bend = getattr(features, f"{finger}_bend", None)
    extension = 1.0 - curl if bend is None else 1.0 - _clip01(float(bend))
    return FingerState(
        curl=curl,
        extension=_clip01(extension),
        abduction=None,
        is_up=False,
        valid=True,
    )


def _sanitize_finger_state(state: FingerState) -> FingerState:
    return FingerState(
        curl=_clip01(state.curl),
        extension=_clip01(state.extension),
        abduction=None if state.abduction is None else _clip_signed(state.abduction),
        is_up=bool(state.is_up),
        valid=bool(state.valid),
    )


def _feature_value(features: HandFeatures, feature: str) -> float:
    if hasattr(features, feature):
        return _clip01(getattr(features, feature))
    if feature.endswith("_bend"):
        legacy_feature = f"{feature.removesuffix('_bend')}_curl"
        if hasattr(features, legacy_feature):
            return _clip01(getattr(features, legacy_feature))
    raise CurlRetargeterError(f"HandFeatures does not provide configured feature {feature!r}.")


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


def _clip_pinch_distance(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return _clip01(value)


def _clip_signed(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(-1.0, float(value)))
