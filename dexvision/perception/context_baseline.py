"""RGB-only object and scene context for metric body-origin hypotheses."""

from __future__ import annotations

import cv2
import numpy as np

from dexvision.perception.body_center_model import (
    BodyCenterEstimator,
    component_features,
    temporal_features,
)
from dexvision.perception.height_model import (
    decode_height,
    predict_tree,
    predict_nearest,
)
from dexvision.perception.object_pose import PoseEstimate
from dexvision.perception.perception_baseline import PerceptionBaseline
from dexvision.perception.predictive_tracker import PredictiveTracker

FEATURE_VERSION = "level5/rgb-context370-v1"
FEATURE_COUNT = 370


def prepare_context(rgb: np.ndarray):
    """Compute the shared visible-pixel maps once per image, without labels."""
    value = rgb.max(axis=2).astype(np.float32)
    intensity = max(32.0, float(np.median(value)))
    pixels = np.stack(
        (
            np.clip(value / intensity, 0.0, 4.0) / 4.0,
            value < 0.8 * intensity,
            value > 2.0 * intensity,
        ),
        axis=-1,
    ).astype(np.float32)
    global_rgb = cv2.resize(pixels, (8, 6), interpolation=cv2.INTER_AREA).ravel()
    return pixels, global_rgb


def context_features(
    rgb: np.ndarray,
    detection,
    timestamp: float,
    previous,
    *,
    maximum_gap_s: float,
    prepared=None,
):
    """[370]: causal component34 + global144 + local192 RGB-derived context.

    Context channels are value/median clipped at4 and divided by4, dark-pixel
    occupancy below0.8median, and bright occupancy above2median. The median
    intensity has a32/255 floor. Float32 area grids preserve subpixel means;
    global8x6 and local8x8 cover the frame and a128px object-centred crop.
    No labels, condition, robot state, phase, semantic id or future pixels enter.
    """
    current = component_features(rgb, detection)
    temporal = temporal_features(
        current, timestamp, previous, maximum_gap_s=maximum_gap_s
    )
    pixels, global_rgb = prepare_context(rgb) if prepared is None else prepared
    x0, y0, x1, y1 = detection.box
    x, y = (x0 + x1) // 2, (y0 + y1) // 2
    local = pixels[
        max(0, y - 64) : min(rgb.shape[0], y + 64),
        max(0, x - 64) : min(rgb.shape[1], x + 64),
    ]
    local_rgb = cv2.resize(local, (8, 8), interpolation=cv2.INTER_AREA).ravel()
    return np.r_[temporal, global_rgb, local_rgb], current


class ContextPose(BodyCenterEstimator):
    def __init__(self, calibration, models, *, maximum_gap_s):
        if any(m.get("feature_version") != FEATURE_VERSION for m in models.values()):
            raise ValueError(
                "Context pose requires exact RGB-context370 feature contract."
            )
        super().__init__(calibration, models, maximum_gap_s=maximum_gap_s)

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
        if model["kind"] == "deterministic_world_height_tree":
            prediction = predict_tree(model, features)[0]
        elif model["kind"] == "deterministic_world_height_nearest":
            prediction = predict_nearest(model, features)[0]
        else:
            from dexvision.perception.body_center_model import predict_kernel

            prediction = predict_kernel(model, features)[0]
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


class ContextBaseline(PerceptionBaseline):
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
            self.pose = ContextPose(
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
