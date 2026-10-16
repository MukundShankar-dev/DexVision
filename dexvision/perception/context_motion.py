"""Versioned neural RGB-context inference; older context engines remain frozen."""

from __future__ import annotations

import numpy as np

from dexvision.perception.body_center_model import (
    BodyCenterEstimator,
)
from dexvision.perception.height_model import decode_height
from dexvision.perception.object_pose import PoseEstimate
from dexvision.perception.perception_baseline import PerceptionBaseline
from dexvision.perception.predictive_tracker import PredictiveTracker

from dexvision.perception.context_baseline import (
    context_features as current_context,
    prepare_context,
)

FEATURE_VERSION = "level5/rgb-motion706-v1"
FEATURE_COUNT = 706


def context_features(
    rgb, detection, timestamp, previous, *, maximum_gap_s, prepared=None
):
    """Current context370 plus336 past-to-current RGB rates; no future/state.

    The stored current370 context is acquisition-time evidence. Missing or expired
    past context has explicit availability0 in the component temporal block and
    zero numerical padding, never a claimed zero physical motion measurement.
    """
    available = previous is not None and 0 < timestamp - previous[0] <= maximum_gap_s
    component_previous = (previous[0], previous[1][:16]) if available else None
    current, _ = current_context(
        rgb,
        detection,
        timestamp,
        component_previous,
        maximum_gap_s=maximum_gap_s,
        prepared=prepared,
    )
    rates = (
        (current[34:] - previous[1][34:]) / (timestamp - previous[0])
        if available
        else np.zeros(336)
    )
    return np.r_[current, rates], current


class NeuralContextPose(BodyCenterEstimator):
    def __init__(self, calibration, models, *, maximum_gap_s):
        if any(m.get("feature_version") != FEATURE_VERSION for m in models.values()):
            raise ValueError(
                "Context pose requires exact RGB-motion706 feature contract."
            )
        super().__init__(calibration, models, maximum_gap_s=maximum_gap_s)
        # Compile portable arrays once; never mutate the hash-bound JSON model.
        self.models = {
            c: {
                **m,
                "mean": np.asarray(m["mean"]),
                "spread": np.asarray(m["spread"]),
                "target_mean": np.asarray(m["target_mean"]),
                "target_spread": np.asarray(m["target_spread"]),
                "layers": [
                    {
                        "weight": np.asarray(layer["weight"]),
                        "bias": np.asarray(layer["bias"]),
                    }
                    for layer in m["layers"]
                ],
            }
            for c, m in models.items()
        }

    def reset(self):
        super().reset()
        self._context_rgb = None
        self._context_timestamp = None
        self._context_maps = None

    def estimate_rgb(self, rgb, track):
        if track.ambiguous or track.detection.class_id not in self.models:
            return PoseEstimate(
                track,
                None,
                None,
                None,
                None,
                None,
                None,
                "ambiguous_or_missing_context_model",
            )
        if self._context_rgb is not rgb or self._context_timestamp != track.timestamp:
            self._context_maps = prepare_context(rgb)
            self._context_rgb, self._context_timestamp = rgb, track.timestamp
        features, current = context_features(
            rgb,
            track.detection,
            track.timestamp,
            self.feature_history.get(track.track_id),
            maximum_gap_s=self.maximum_gap_s,
            prepared=self._context_maps,
        )
        self.feature_history[track.track_id] = track.timestamp, current.copy()
        model = self.models[track.detection.class_id]
        from dexvision.perception.context_motion_network import predict_network

        prediction = predict_network(model, features)
        try:
            position = decode_height(prediction, features, self.calibration)
        except ValueError:
            return PoseEstimate(
                track, None, None, None, None, None, None, "invalid_metric_prediction"
            )
        prior = self.previous.get(track.track_id)
        velocity = None
        if prior is not None and 0 < track.timestamp - prior[0] <= self.maximum_gap_s:
            velocity = tuple((position - prior[1]) / (track.timestamp - prior[0]))
        self.previous[track.track_id] = track.timestamp, position.copy()
        return PoseEstimate(
            track,
            tuple(position),
            None,
            velocity,
            None,
            None,
            None,
            "full_orientation_and_semantic_identity_unavailable",
        )


class NeuralContextBaseline(PerceptionBaseline):
    """Same frozen detector/ray decoder with richer visible RGB evidence only."""

    def __init__(self, recipe: dict, model: dict):
        import json
        from pathlib import Path

        super().__init__(recipe, model)
        if recipe["tracking_mode"] == "predictive":
            self.tracker = PredictiveTracker(
                **recipe["tracker"], **recipe["predictive_tracker"]
            )
        calibration = json.loads(
            (Path(recipe["visual_root"]) / "calibration.json").read_text()
        )
        self.image_shape = (
            calibration["image_size"][1],
            calibration["image_size"][0],
            3,
        )
        if "body_center_models" in model:
            self.pose = NeuralContextPose(
                calibration,
                model["body_center_models"],
                maximum_gap_s=recipe["tracker"]["maximum_gap_s"],
            )

    def infer(self, rgb: np.ndarray, timestamp: float) -> tuple:
        if rgb.shape != self.image_shape:
            raise ValueError(
                "RGB size differs from the frozen metric camera calibration."
            )
        return super().infer(rgb, timestamp)
