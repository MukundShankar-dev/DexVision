"""Smoothing utilities for hand-control features."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, Literal

import numpy as np

from dexvision.features.hand_features import HandFeatures, no_hand_features


CONTROL_FEATURE_FIELDS: Final[tuple[str, ...]] = (
    "thumb_curl",
    "index_curl",
    "middle_curl",
    "ring_curl",
    "pinky_curl",
    "pinch_thumb_index",
    "palm_roll_proxy",
    "palm_pitch_proxy",
)
LowConfidenceBehavior = Literal["hold", "decay"]


@dataclass(frozen=True)
class SmoothingConfig:
    """Configuration for exponential moving average hand-feature smoothing."""

    alpha: float = 0.35
    min_confidence: float = 0.2
    low_confidence_behavior: LowConfidenceBehavior = "hold"
    decay_alpha: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in the range (0.0, 1.0].")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in the range [0.0, 1.0].")
        if self.low_confidence_behavior not in ("hold", "decay"):
            raise ValueError("low_confidence_behavior must be 'hold' or 'decay'.")
        if not 0.0 <= self.decay_alpha <= 1.0:
            raise ValueError("decay_alpha must be in the range [0.0, 1.0].")


class FeatureSmoother:
    """Exponential moving average smoother for scalar ``HandFeatures`` values.

    Missing or low-confidence input freezes the last stable control values by
    default while still reporting the current low confidence.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.35,
        min_confidence: float = 0.2,
        low_confidence_behavior: LowConfidenceBehavior = "hold",
        decay_alpha: float = 0.05,
    ) -> None:
        self.config = SmoothingConfig(
            alpha=alpha,
            min_confidence=min_confidence,
            low_confidence_behavior=low_confidence_behavior,
            decay_alpha=decay_alpha,
        )
        self._state: HandFeatures | None = None

    @property
    def state(self) -> HandFeatures | None:
        """Return the most recent smoothed feature vector, if initialized."""

        return self._state

    def reset(self) -> None:
        """Clear smoothing history."""

        self._state = None

    def update(self, features: HandFeatures | None) -> HandFeatures:
        """Update the smoother and return finite smoothed features.

        Args:
            features: Current hand features, or ``None`` when tracking is
                unavailable.

        Returns:
            A finite ``HandFeatures`` instance.
        """

        current = sanitize_hand_features(features) if features is not None else no_hand_features()

        if self._state is None:
            self._state = (
                current
                if current.confidence >= self.config.min_confidence
                else replace(no_hand_features(), confidence=current.confidence)
            )
            return self._state

        if current.confidence < self.config.min_confidence:
            self._state = self._handle_low_confidence(current)
            return self._state

        self._state = _ema_features(self._state, current, self.config.alpha)
        return self._state

    def _handle_low_confidence(self, current: HandFeatures) -> HandFeatures:
        previous = self._state or no_hand_features()
        if self.config.low_confidence_behavior == "decay":
            decayed = _ema_features(previous, no_hand_features(), self.config.decay_alpha)
            return replace(decayed, confidence=current.confidence)

        return replace(previous, confidence=current.confidence)


def sanitize_hand_features(features: HandFeatures) -> HandFeatures:
    """Clip non-finite or out-of-range feature values to safe bounds."""

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


def _ema_features(previous: HandFeatures, current: HandFeatures, alpha: float) -> HandFeatures:
    return HandFeatures(
        thumb_curl=_ema(previous.thumb_curl, current.thumb_curl, alpha),
        index_curl=_ema(previous.index_curl, current.index_curl, alpha),
        middle_curl=_ema(previous.middle_curl, current.middle_curl, alpha),
        ring_curl=_ema(previous.ring_curl, current.ring_curl, alpha),
        pinky_curl=_ema(previous.pinky_curl, current.pinky_curl, alpha),
        pinch_thumb_index=_ema(previous.pinch_thumb_index, current.pinch_thumb_index, alpha),
        palm_roll_proxy=_ema(previous.palm_roll_proxy, current.palm_roll_proxy, alpha),
        palm_pitch_proxy=_ema(previous.palm_pitch_proxy, current.palm_pitch_proxy, alpha),
        confidence=_ema(previous.confidence, current.confidence, alpha),
    )


def _ema(previous: float, current: float, alpha: float) -> float:
    return float((alpha * current) + ((1.0 - alpha) * previous))


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _clip_signed(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))
