"""Display-only calibrated projection; never wrap or mutate inference state."""

from __future__ import annotations
from dataclasses import dataclass
from dexvision.perception.body_center_model import BodyCenterEstimator
from dexvision.evaluation.evaluate_perception import review_overlay


@dataclass(frozen=True)
class CalibratedProjection:
    calibration: dict

    def project(self, points):
        return BodyCenterEstimator.project(self, points)


@dataclass(frozen=True)
class OverlayView:
    model: dict
    pose: CalibratedProjection


def review_dual_overlay(rgb, estimates, row, latency, baseline):
    """Only the renderer receives this view, after frozen RGB inference."""
    view = OverlayView(baseline.model, CalibratedProjection(baseline.pose.calibration))
    return review_overlay(rgb, estimates, row, latency, view)
