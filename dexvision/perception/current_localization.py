"""Current RGB body-origin localization; physical motion remains unavailable."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from dexvision.perception.body_center_model import BodyCenterEstimator
from dexvision.perception.context_motion import (
    NeuralContextPose,
    context_features,
    prepare_context,
)
from dexvision.perception.context_motion_network import predict_network
from dexvision.perception.height_model import decode_height
from dexvision.perception.object_pose import PoseEstimate
from dexvision.perception.perception_baseline import PerceptionBaseline
from dexvision.perception.predictive_tracker import PredictiveTracker

ENGINE = "level5/current-rgb-localization-v1"


class CurrentPose(BodyCenterEstimator):
    def __init__(self, calibration, models, maximum_gap_s, *, feature_cache=None):
        super().__init__(calibration, models, maximum_gap_s=maximum_gap_s)
        self.models = NeuralContextPose(
            calibration, models, maximum_gap_s=maximum_gap_s
        ).models
        self.feature_cache = feature_cache

    def estimate_frame(self, rgb, tracks):
        prepared = None
        outputs = []
        pixel_hash = (
            hashlib.sha256(rgb.tobytes()).hexdigest()
            if self.feature_cache is not None
            else None
        )
        for track in tracks:
            c = track.detection.class_id
            if c not in self.models:
                outputs.append(
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
            key = None
            if self.feature_cache is not None:
                key = (
                    pixel_hash,
                    c,
                    tuple(track.detection.box),
                    hashlib.sha256(track.detection.mask.tobytes()).hexdigest(),
                )
            x = self.feature_cache.get(key) if self.feature_cache is not None else None
            if x is None:
                if prepared is None:
                    prepared = prepare_context(rgb)
                x, _ = context_features(
                    rgb,
                    track.detection,
                    track.timestamp,
                    None,
                    maximum_gap_s=self.maximum_gap_s,
                    prepared=prepared,
                )
                if x[16:34].any() or x[370:].any():
                    raise ValueError("Current-only features cannot contain history.")
                if self.feature_cache is not None:
                    x.setflags(write=False)
                    self.feature_cache[key] = x
            try:
                position = decode_height(
                    predict_network(self.models[c], x), x, self.calibration
                )
            except ValueError:
                outputs.append(
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
            reason = (
                "ambiguous_identity_position_hypothesis"
                if track.ambiguous
                else "full_orientation_semantic_identity_and_motion_unavailable"
            )
            outputs.append(
                PoseEstimate(
                    track, tuple(position), None, None, None, None, None, reason
                )
            )
        return tuple(outputs)


class CurrentLocalizationBaseline(PerceptionBaseline):
    """Tracker remains causal; localization receives only each current RGB frame."""

    def __init__(self, recipe, model):
        if (
            model.get("feature_version") != ENGINE
            or model.get("inference_variant") != ENGINE
            or model.get("localization_mode") != "current_only_no_motion"
        ):
            raise ValueError("Frozen current-only model required.")
        super().__init__(recipe, model)
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
        # This optional VAL-only cache contains pure RGB/mask/box feature values.
        # It has no labels, timestamps, model output or episode history; default
        # inference and fresh confirmation use uncached current RGB extraction.
        self.pose = CurrentPose(
            calibration,
            model["body_center_models"],
            recipe["tracker"]["maximum_gap_s"],
            feature_cache=recipe.get("_eval_current_feature_cache"),
        )

    def infer(self, rgb, timestamp):
        if rgb.shape != self.image_shape:
            raise ValueError("RGB size differs from frozen camera calibration.")
        tracks = self.tracker.update(self.detector.detect(rgb), timestamp)
        return self.pose.estimate_frame(rgb, tracks)
