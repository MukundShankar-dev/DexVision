"""Pick-specific causal phase diagnostics, without retrospective expert labels."""
from __future__ import annotations

import numpy as np


def pick_phases(held: np.ndarray, heights: np.ndarray, threshold: float) -> np.ndarray:
    """Current contact-derived retention and measured height -> acquire/lift/stabilize."""
    held, heights = np.asarray(held), np.asarray(heights, float)
    if held.shape != heights.shape or not np.isfinite(heights).all():
        raise ValueError("Invalid pick phase inputs")
    return np.where(held > 0.5, np.where(heights >= threshold, "stabilize", "lift"), "acquire")


def phase_errors(predictions, targets, masks, phases) -> dict:
    """Per-phase normalized MSE with explicit frame and active-target counts."""
    predictions, targets, masks = map(np.asarray, (predictions, targets, masks))
    phases = np.asarray(phases)
    if predictions.shape != targets.shape or masks.shape != targets.shape or len(phases) != len(targets):
        raise ValueError("Phase error shapes disagree")
    if not all(np.isfinite(v).all() for v in (predictions, targets, masks)):
        raise ValueError("Non-finite phase error inputs")
    result = {}
    for phase in sorted(set(phases)):
        take = phases == phase
        count = int(masks[take].sum())
        result[str(phase)] = {"frames": int(take.sum()), "active_values": count,
            "masked_mse": float((((predictions[take] - targets[take]) ** 2) * masks[take]).sum() / count) if count else None}
    return result
