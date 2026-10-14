"""Deterministic RGB body-centre height regression and calibrated ray decoder."""

from __future__ import annotations

import numpy as np

from dexvision.perception.body_center_model import FEATURE_SCALES


def height_target(position: np.ndarray, features: np.ndarray, calibration: dict) -> np.ndarray:
    """Training-only label: body centre offsets /32px and WORLD height /0.1m."""
    from dexvision.perception.body_center_model import metric_target
    target = metric_target(position, features, calibration)
    target[2] = position[2]/.1
    return target


def decode_height(prediction: np.ndarray, features: np.ndarray, calibration: dict) -> np.ndarray:
    """Decode inferred body pixel centre plus learned height, not board height.

    Camera right/down/forward and world/metre conventions are frozen. Neither
    position nor orientation labels, task state or robot telemetry enter here.
    """
    prediction = np.asarray(prediction, float)
    if prediction.shape != (3,) or not np.isfinite(prediction).all():
        raise ValueError("Invalid inferred body centre/height.")
    pixel = features[:2]*FEATURE_SCALES[:2]+prediction[:2]*32.
    ray = np.linalg.solve(np.asarray(calibration["intrinsics"], float), np.r_[pixel, 1.])
    transform = np.linalg.inv(np.asarray(calibration["camera_from_world"], float))
    direction = transform[:3, :3] @ ray
    if abs(direction[2]) < 1e-8:
        raise ValueError("Body ray parallel to inferred height plane.")
    depth = (prediction[2]*.1-transform[2, 3])/direction[2]
    if depth <= 0:
        raise ValueError("Inferred body centre behind camera.")
    return transform[:3, 3]+direction*depth


def fit_tree(inputs: np.ndarray, targets: np.ndarray, weights: np.ndarray,
             *, maximum_depth: int, minimum_leaf_samples: int) -> dict:
    """Full-budget deterministic weighted CART: all frozen RGB/context features eligible.

    Split by weighted multi-output squared error; ties use feature then ordered
    threshold. No random seed, validation labels or episode id enters a node.
    Trees retain training leaves, each with a weighted body-centre prediction.
    """
    inputs, targets, weights = np.asarray(inputs, float), np.asarray(targets, float), np.asarray(weights, float)
    if inputs.ndim != 2 or targets.shape != (len(inputs), 3) or weights.shape != (len(inputs),):
        raise ValueError("Invalid training tree shapes.")
    if not np.isfinite(inputs).all() or not np.isfinite(targets).all() or np.any(weights <= 0):
        raise ValueError("Invalid training tree values.")

    def build(indices, depth):
        x, y, w = inputs[indices], targets[indices], weights[indices]
        mean = np.average(y, axis=0, weights=w)
        node = {"prediction": mean.tolist(), "training_rows": len(indices)}
        if depth >= maximum_depth or len(indices) < 2*minimum_leaf_samples:
            return node
        total_w = w.sum()
        total_y = (y*w[:, None]).sum(0)
        parent_loss = np.sum(w[:, None]*(y-mean)**2)
        best = None
        for feature in range(inputs.shape[1]):
            order = np.argsort(x[:, feature], kind="stable")
            sorted_x, sorted_y, sorted_w = x[order, feature], y[order], w[order]
            left_w = np.cumsum(sorted_w)[:-1]
            left_y = np.cumsum(sorted_y*sorted_w[:, None], axis=0)[:-1]
            improvement = np.sum(left_y**2/left_w[:, None] +
                                 (total_y-left_y)**2/(total_w-left_w)[:, None], axis=1) - np.sum(total_y**2)/total_w
            allowed = ((sorted_x[:-1] < sorted_x[1:]) &
                       (np.arange(1, len(indices)) >= minimum_leaf_samples) &
                       (np.arange(1, len(indices)) <= len(indices)-minimum_leaf_samples))
            improvement[~allowed] = -np.inf
            split = int(improvement.argmax())
            gain = improvement[split]
            if gain > max(1e-12, parent_loss*1e-10) and (best is None or gain > best[0]+1e-12):
                threshold = sorted_x[split]+(sorted_x[split+1]-sorted_x[split])/2
                # Adjacent floats have no representable midpoint. <= must
                # retain the scored partition and never create an empty child.
                if threshold >= sorted_x[split+1]:
                    threshold = sorted_x[split]
                best = gain, feature, float(threshold)
        if best is None:
            return node
        _, feature, threshold = best
        left = x[:, feature] <= threshold
        node.update(feature=feature, threshold=threshold,
                    left=build(indices[left], depth+1), right=build(indices[~left], depth+1))
        return node

    return {"kind": "deterministic_world_height_tree", "root": build(np.arange(len(inputs)), 0),
            "maximum_depth": maximum_depth, "minimum_leaf_samples": minimum_leaf_samples,
            "training_rows": len(inputs), "target_units": "pixel_offsets/32_world_height/0.1m"}


def predict_tree(model: dict, inputs: np.ndarray) -> np.ndarray:
    result = []
    for row in np.atleast_2d(inputs):
        node = model["root"]
        while "feature" in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        result.append(node["prediction"])
    return np.asarray(result, float)


def fit_nearest(inputs: np.ndarray, targets: np.ndarray, weights: np.ndarray, *, neighbours: int) -> dict:
    """TRAIN-only deterministic local regression; no identity or source features."""
    mean, spread = inputs.mean(0), np.maximum(inputs.std(0), .02)
    return {"kind": "deterministic_world_height_nearest", "mean": mean.tolist(), "spread": spread.tolist(),
            "inputs": ((inputs-mean)/spread).tolist(), "targets": targets.tolist(),
            "weights": weights.tolist(), "neighbours": neighbours, "training_rows": len(inputs)}


def predict_nearest(model: dict, inputs: np.ndarray) -> np.ndarray:
    x = (np.atleast_2d(inputs)-model["mean"])/model["spread"]
    training = np.asarray(model["inputs"])
    targets, weights = np.asarray(model["targets"]), np.asarray(model["weights"])
    predictions = []
    for row in x:
        distance = np.sum((training-row)**2, axis=1)
        selected = np.argsort(distance, kind="stable")[:model["neighbours"]]
        local_weights = weights[selected]/np.maximum(distance[selected], 1e-8)
        predictions.append(np.average(targets[selected], axis=0, weights=local_weights))
    return np.asarray(predictions)
