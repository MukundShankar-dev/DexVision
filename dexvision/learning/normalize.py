"""Per-dimension min-max scaling to [-1, 1], fitted on training data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

MIN_RANGE = 1e-6  # dimensions that never vary map to 0 and decode back to their constant


@dataclass(frozen=True)
class Normalizer:
    low: np.ndarray  # [D]
    high: np.ndarray  # [D]

    @classmethod
    def fit(cls, data: np.ndarray) -> Normalizer:
        """``data`` [N, D] -> per-dimension bounds."""
        data = np.asarray(data, dtype=np.float64)
        return cls(data.min(axis=0), data.max(axis=0))

    def _scale(self) -> tuple[np.ndarray, np.ndarray]:
        span = self.high - self.low
        constant = span < MIN_RANGE
        center = np.where(constant, self.low, (self.high + self.low) / 2)
        half = np.where(constant, 1.0, span / 2)
        return center, half

    def encode(self, x):
        """[..., D] -> [..., D] in [-1, 1] on the training range. Works on numpy or torch."""
        center, half = self._scale()
        if isinstance(x, torch.Tensor):
            center = torch.as_tensor(center, dtype=x.dtype, device=x.device)
            half = torch.as_tensor(half, dtype=x.dtype, device=x.device)
        return (x - center) / half

    def decode(self, x):
        center, half = self._scale()
        if isinstance(x, torch.Tensor):
            center = torch.as_tensor(center, dtype=x.dtype, device=x.device)
            half = torch.as_tensor(half, dtype=x.dtype, device=x.device)
        return x * half + center

    def state(self) -> dict[str, list[float]]:
        return {"low": self.low.tolist(), "high": self.high.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> Normalizer:
        return cls(np.asarray(state["low"], dtype=np.float64),
                   np.asarray(state["high"], dtype=np.float64))
