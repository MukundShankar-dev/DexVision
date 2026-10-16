"""Portable learned context network weights and explicit TRAIN-only transforms."""

from __future__ import annotations
import numpy as np


def predict_network(model: dict, features: np.ndarray) -> np.ndarray:
    """RGB motion706 plus visual class one-hot; no task/phase/state inputs."""
    if model["kind"] != "rgb_motion_world_height_mlp":
        raise ValueError("Exact neural context model kind required.")
    x = (np.asarray(features, float) - model["mean"]) / model["spread"]
    if x.shape != (706,) or not np.isfinite(x).all():
        raise ValueError("Expected finite motion706 features.")
    identity = np.zeros(4)
    identity[model["class_index"]] = 1.0
    x = np.r_[x, identity]
    for index, layer in enumerate(model["layers"]):
        x = np.asarray(layer["weight"]) @ x + layer["bias"]
        if index + 1 < len(model["layers"]):
            x = np.maximum(x, 0.0)
    return x * model["target_spread"] + model["target_mean"]
