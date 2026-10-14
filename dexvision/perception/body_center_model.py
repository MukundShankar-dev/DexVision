"""Training-only kernel metric body-centre calibration from RGB components."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from dexvision.perception.metric_geometry import convex_polygon, polygon_features


FEATURE_NAMES = (
    "centroid_u", "centroid_v", "bbox_width", "bbox_height", "std_u", "std_v",
    "covariance", "area", "solidity", "bbox_center_minus_centroid_u",
    "bbox_center_minus_centroid_v", "hue_median", "saturation_median", "value_median",
    "value_p10", "value_p90",
)
FEATURE_SCALES = np.array([640, 480, 64, 64, 32, 32, 32, 4096, 1, 32, 32, 180, 255, 255, 255, 255], float)


def component_features(rgb: np.ndarray, detection) -> np.ndarray:
    """Return [16] fixed-scale RGB/mask geometry; no annotation or scene state."""
    y, x = np.nonzero(detection.mask)
    if not len(x):
        raise ValueError("Empty visible component.")
    pixels = cv2.cvtColor(rgb[detection.mask].reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    box = np.asarray(detection.box, float)
    centroid = np.array([x.mean(), y.mean()])
    if len(x) >= 3:
        try:
            polygon = convex_polygon(np.column_stack((x, y)))
            moments = polygon_features(polygon)
            area = abs(cv2.contourArea(polygon.astype(np.float32)))
        except ValueError:
            moments, area = np.r_[centroid, 0., 0., 0.], float(len(x))
    else:
        moments, area = np.r_[centroid, 0., 0., 0.], float(len(x))
    extent = box[2:]-box[:2]
    center_offset = (box[:2]+box[2:]-1)/2-centroid
    features = np.r_[centroid, extent, moments[2:], len(x), len(x)/max(area, len(x)),
                     center_offset, np.median(pixels, axis=0),
                     np.percentile(pixels[:, 2], [10, 90])]
    return features/FEATURE_SCALES


def temporal_features(current: np.ndarray, timestamp: float, previous,
                      *, maximum_gap_s: float) -> np.ndarray:
    """[34] current RGB + available/dt + prior RGB, never prior truth labels.

    Missing history is explicitly masked by availability=0; its numeric padding
    is not a measurement or a zero physical velocity. Old RGB remains causal
    context, not a newly timestamped observation.
    """
    available = previous is not None and 0 < timestamp-previous[0] <= maximum_gap_s
    dt = round((timestamp-previous[0])/maximum_gap_s, 6) if available else 0.
    prior_features = previous[1] if available else np.zeros(len(FEATURE_NAMES))
    return np.r_[current, float(available), dt, prior_features]


def metric_target(position: np.ndarray, features: np.ndarray, calibration: dict) -> np.ndarray:
    """Evaluation/training label -> [body centre u/v offset px, camera depth m]."""
    extrinsic = np.asarray(calibration["camera_from_world"], float)
    camera = extrinsic[:3, :3] @ position + extrinsic[:3, 3]
    projected = np.asarray(calibration["intrinsics"], float) @ camera
    offset = projected[:2]/projected[2]-features[:2]*FEATURE_SCALES[:2]
    return np.r_[offset/32., camera[2]]


def decode_metric(prediction: np.ndarray, features: np.ndarray, calibration: dict) -> np.ndarray:
    """Learned centre offsets/depth -> metre/world position via fixed pinhole."""
    prediction = np.asarray(prediction, float)
    if prediction.shape != (3,) or not np.isfinite(prediction).all() or prediction[2] <= 0:
        raise ValueError("Invalid learned metric centre/depth.")
    pixel = features[:2]*FEATURE_SCALES[:2]+prediction[:2]*32.
    ray = np.linalg.solve(np.asarray(calibration["intrinsics"], float), np.r_[pixel, 1.])
    world = np.linalg.inv(np.asarray(calibration["camera_from_world"], float)) @ np.r_[ray*prediction[2], 1.]
    return world[:3]


def fit_kernel(inputs: np.ndarray, targets: np.ndarray, *, bandwidth: float,
               regularization: float, maximum_centres: int, weights: np.ndarray) -> dict:
    """Deterministic Nyström ridge fit, all training rows, frozen centre budget.

    Fixed input scales above; spread statistics fit TRAIN only with a floor.
    Centre selection uses ordered feature-space farthest points, no validation
    rows, semantic identity, source episode ordinal or random seed.
    """
    inputs, targets = np.asarray(inputs, float), np.asarray(targets, float)
    mean = inputs.mean(0)
    spread = np.maximum(inputs.std(0), .02)
    normalized = (inputs-mean)/spread
    selected = [int(np.argmin(np.linalg.norm(normalized, axis=1)))]
    nearest = np.full(len(inputs), np.inf)
    for _ in range(min(maximum_centres, len(inputs))-1):
        distance = np.sum((normalized-normalized[selected[-1]])**2, axis=1)
        nearest = np.minimum(nearest, distance)
        index = int(nearest.argmax())
        if nearest[index] < 1e-10:
            break
        selected.append(index)
    centres = normalized[selected]
    kernel = np.exp(-cdist(normalized, centres, "sqeuclidean")/(2*bandwidth**2))
    design = np.column_stack((np.ones(len(inputs)), kernel))
    weighted = design*weights[:, None]
    penalty = np.eye(design.shape[1])*regularization
    penalty[0, 0] = 1e-10
    coefficients = np.linalg.solve(design.T @ weighted+penalty, weighted.T @ targets)
    return {"mean": mean.tolist(), "spread": spread.tolist(), "centres": centres.tolist(),
            "coefficients": coefficients.tolist(), "bandwidth": bandwidth,
            "regularization": regularization, "training_rows": len(inputs)}


def predict_kernel(model: dict, inputs: np.ndarray) -> np.ndarray:
    normalized = (np.asarray(inputs, float)-model["mean"])/model["spread"]
    kernel = np.exp(-cdist(normalized.reshape(-1, len(model["mean"])),
                          np.asarray(model["centres"]), "sqeuclidean")/(2*model["bandwidth"]**2))
    return np.column_stack((np.ones(len(kernel)), kernel)) @ np.asarray(model["coefficients"])


class BodyCenterEstimator:
    """RGB-only learned metric position with explicit unavailable pose fields."""

    def __init__(self, calibration: dict, models: dict, *, maximum_gap_s: float):
        self.calibration = calibration
        self.models = models
        self.maximum_gap_s = maximum_gap_s
        self.reset()

    def reset(self):
        self.previous = {}
        self.feature_history = {}

    def project(self, points):
        extrinsic = np.asarray(self.calibration["camera_from_world"])
        camera = np.asarray(points) @ extrinsic[:3, :3].T+extrinsic[:3, 3]
        projected = camera @ np.asarray(self.calibration["intrinsics"]).T
        return projected[:, :2]/projected[:, 2, None]

    def estimate_rgb(self, rgb, track):
        from dexvision.perception.object_pose import PoseEstimate
        if track.ambiguous or track.detection.class_id not in self.models:
            return PoseEstimate(track, None, None, None, None, None, None, "ambiguous_or_missing_metric_model")
        features = component_features(rgb, track.detection)
        model = self.models[track.detection.class_id]
        current_features = features
        if model.get("temporal_context"):
            features = temporal_features(features, track.timestamp, self.feature_history.get(track.track_id),
                                         maximum_gap_s=self.maximum_gap_s)
        self.feature_history[track.track_id] = track.timestamp, current_features.copy()
        if model.get("kind") == "deterministic_world_height_nearest":
            from dexvision.perception.height_model import decode_height, predict_nearest
            prediction = predict_nearest(model, features)[0]
            decoder = decode_height
        elif model.get("kind") == "deterministic_world_height_kernel":
            from dexvision.perception.height_model import decode_height
            prediction = predict_kernel(model, features)[0]
            decoder = decode_height
        elif model.get("kind") == "deterministic_world_height_tree":
            from dexvision.perception.height_model import decode_height, predict_tree
            prediction = predict_tree(model, features)[0]
            decoder = decode_height
        else:
            prediction = predict_kernel(model, features)[0]
            decoder = decode_metric
        try:
            position = decoder(prediction, features, self.calibration)
        except ValueError:
            return PoseEstimate(track, None, None, None, None, None, None, "invalid_metric_prediction")
        velocity = None
        prior = self.previous.get(track.track_id)
        if prior is not None and 0 < track.timestamp-prior[0] <= self.maximum_gap_s:
            velocity = tuple((position-prior[1])/(track.timestamp-prior[0]))
        self.previous[track.track_id] = track.timestamp, position.copy()
        return PoseEstimate(track, tuple(position), None, velocity, None, None, None,
                            "full_orientation_and_semantic_identity_unavailable")
