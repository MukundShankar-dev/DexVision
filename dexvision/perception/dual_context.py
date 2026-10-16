"""Independent current-RGB localization and causal motion localization.

A current-frame position is a diagnostic hypothesis even when visual identity
is ambiguous. Ambiguity invalidity and all full-control rejections remain intact.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from dexvision.perception.context_motion import (
    context_features,
    prepare_context,
    NeuralContextPose,
)
from dexvision.perception.context_motion_network import predict_network
from dexvision.perception.height_model import decode_height
from dexvision.perception.object_pose import PoseEstimate
from dexvision.perception.perception_baseline import PerceptionBaseline
from dexvision.perception.predictive_tracker import PredictiveTracker

INFERENCE_VARIANT = "level5/dual-rgb-localization-v1"


def current_frame_features(features):
    """Keep current RGB only; [706] with explicit zero padding for history."""
    x = np.asarray(features).copy()
    if x.shape != (706,):
        raise ValueError("Exact RGB706 vector required.")
    x[16:34] = 0
    x[370:] = 0
    return x


def compose_history(current, timestamp, previous, maximum_gap_s):
    """Exact old RGB706 construction from cached current-only RGB370 values."""
    from dexvision.perception.body_center_model import temporal_features

    available = previous is not None and 0 < timestamp - previous[0] <= maximum_gap_s
    component = temporal_features(
        current[:16],
        timestamp,
        (previous[0], previous[1][:16]) if available else None,
        maximum_gap_s=maximum_gap_s,
    )
    rates = (
        (current[34:] - previous[1][34:]) / (timestamp - previous[0])
        if available
        else np.zeros(336)
    )
    return np.r_[component, current[34:], rates]


class FeatureHistory:
    """Shared TRAIN/runtime causal routing; uncertain class history is cleared."""

    def __init__(self, maximum_gap_s):
        self.maximum_gap_s = maximum_gap_s
        self.reset()

    def reset(self):
        self.history = {}
        self.classes = {}

    def observe(self, rgb, tracks, *, force_current=False, current_contexts=None):
        uncertain = {t.detection.class_id for t in tracks if t.ambiguous}
        for tid in list(self.history):
            if self.classes.get(tid) in uncertain:
                self.history.pop(tid, None)
        prepared = prepare_context(rgb) if current_contexts is None else None
        result = []
        for index, t in enumerate(tracks):
            self.classes[t.track_id] = t.detection.class_id
            previous = self.history.get(t.track_id)
            if current_contexts is None:
                _, current = context_features(
                    rgb,
                    t.detection,
                    t.timestamp,
                    None,
                    maximum_gap_s=self.maximum_gap_s,
                    prepared=prepared,
                )
            else:
                current = current_contexts[index]
            x = compose_history(current, t.timestamp, previous, self.maximum_gap_s)
            temporal = (
                bool(x[16])
                and t.detection.class_id not in uncertain
                and not force_current
            )
            if not temporal:
                x = current_frame_features(x)
            if t.detection.class_id not in uncertain:
                self.history[t.track_id] = t.timestamp, current.copy()
            result.append((x, temporal))
        return result


class DualContextPose:
    def __init__(self, calibration, model, maximum_gap_s):
        if model.get("inference_variant") != INFERENCE_VARIANT:
            raise ValueError("Exact dual-head localization contract required.")
        self.calibration = calibration
        self.maximum_gap_s = maximum_gap_s
        # Reuse the immutable portable-array compiler, never its ambiguity route.
        self.temporal_models = NeuralContextPose(
            calibration, model["body_center_models"], maximum_gap_s=maximum_gap_s
        ).models
        self.current_models = NeuralContextPose(
            calibration, model["current_frame_models"], maximum_gap_s=maximum_gap_s
        ).models
        if set(self.temporal_models) != set(self.current_models):
            raise ValueError("Both heads must cover the same visual classes.")
        self.features = FeatureHistory(maximum_gap_s)
        self.reset()

    def reset(self):
        self.features.reset()
        self.previous = {}

    def estimate_frame(self, rgb, tracks, *, force_current=False):
        uncertain = {t.detection.class_id for t in tracks if t.ambiguous}
        for tid in list(self.previous):
            if self.features.classes.get(tid) in uncertain:
                self.previous.pop(tid, None)
        features = self.features.observe(rgb, tracks, force_current=force_current)
        estimates = []
        for track, (x, temporal) in zip(tracks, features, strict=True):
            models = self.temporal_models if temporal else self.current_models
            c = track.detection.class_id
            if c not in models:
                estimates.append(
                    PoseEstimate(
                        track,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        "missing_localization_model",
                    )
                )
                continue
            try:
                position = decode_height(
                    predict_network(models[c], x), x, self.calibration
                )
            except ValueError:
                self.previous.pop(track.track_id, None)
                estimates.append(
                    PoseEstimate(
                        track,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        "invalid_metric_prediction",
                    )
                )
                continue
            prior = self.previous.get(track.track_id)
            velocity = None
            if (
                temporal
                and prior is not None
                and 0 < track.timestamp - prior[0] <= self.maximum_gap_s
            ):
                velocity = tuple((position - prior[1]) / (track.timestamp - prior[0]))
            # Current-frame routing never publishes a motion estimate or motion history.
            if temporal and not track.ambiguous:
                self.previous[track.track_id] = track.timestamp, position.copy()
            else:
                self.previous.pop(track.track_id, None)
            reason = (
                "ambiguous_identity_position_hypothesis"
                if track.ambiguous
                else "full_orientation_and_semantic_identity_unavailable"
            )
            estimates.append(
                PoseEstimate(
                    track, tuple(position), None, velocity, None, None, None, reason
                )
            )
        return tuple(estimates)


class DualContextBaseline(PerceptionBaseline):
    def __init__(self, recipe, model, *, force_current=False):
        super().__init__(recipe, model)
        if recipe["tracking_mode"] != "predictive":
            raise ValueError("Frozen predictive tracking required.")
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
        self.pose = DualContextPose(
            calibration, model, recipe["tracker"]["maximum_gap_s"]
        )
        self.force_current = force_current

    def infer(self, rgb, timestamp):
        if rgb.shape != self.image_shape:
            raise ValueError("RGB size differs from frozen camera calibration.")
        tracks = self.tracker.update(self.detector.detect(rgb), timestamp)
        return self.pose.estimate_frame(rgb, tracks, force_current=self.force_current)
