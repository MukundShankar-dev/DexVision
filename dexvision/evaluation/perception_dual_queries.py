"""Model-independent cached VAL features; strict per-phase scalar/batch parity."""

from __future__ import annotations

import numpy as np
from dexvision.evaluation.perception_phase_queries import phase_query_pass
from dexvision.perception.context_motion_network import predict_network
from dexvision.perception.dual_context import current_frame_features
from dexvision.perception.height_model import decode_height


def predict_batch(model, features):
    x = (np.asarray(features, float) - model["mean"]) / model["spread"]
    identity = np.zeros((len(x), 4))
    identity[:, model["class_index"]] = 1
    x = np.c_[x, identity]
    for i, layer in enumerate(model["layers"]):
        x = x @ np.asarray(layer["weight"]).T + layer["bias"]
        if i + 1 < len(model["layers"]):
            x = np.maximum(x, 0)
    return x * model["target_spread"] + model["target_mean"]


def score_query_cache(recipe, model, *, force_current=False):
    import json
    from pathlib import Path

    with np.load(recipe["validation_query_cache"], allow_pickle=False) as c:
        features = c["features"].copy()
        targets = c["targets"].copy()
        matched = c["matched"].copy()
        phases = c["phase"].copy()
        sources = c["source"].copy()
        temporal = c["temporal"].copy()
        ambiguous = c["ambiguous"].copy()
    calibration = json.loads(
        (Path(recipe["visual_root"]) / "calibration.json").read_text()
    )
    errors = np.full(len(features), np.nan)
    temporal &= ~ambiguous
    if force_current:
        temporal[:] = False
    for route, heads in (
        (True, model["body_center_models"]),
        (False, model["current_frame_models"]),
    ):
        indices = np.flatnonzero(matched & (temporal == route))
        if not len(indices):
            continue
        x = features[indices]
        if not route:
            x = np.stack([current_frame_features(v) for v in x])
        head = heads["rigid_cuboid"]
        prediction = predict_batch(head, x)
        for j in np.linspace(0, len(indices) - 1, min(8, len(indices)), dtype=int):
            np.testing.assert_allclose(
                prediction[j], predict_network(head, x[j]), atol=1e-10, rtol=1e-9
            )
        for i, v, p in zip(indices, x, prediction, strict=True):
            try:
                a = decode_height(p, v, calibration)
                b = decode_height(targets[i], v, calibration)
                errors[i] = np.linalg.norm(a - b)
            except ValueError:
                pass
    result = {}
    for phase in recipe["query_phases"]:
        mask = phases == phase
        es = errors[mask & np.isfinite(errors)]
        g = {
            "visible_queries": int(mask.sum()),
            "matches": int(matched[mask].sum()),
            "poses": len(es),
            "visible_physical_source_episodes": len(set(sources[mask])),
            "median_m": float(np.median(es)) if len(es) else None,
            "p95_m": float(np.percentile(es, 95)) if len(es) else None,
        }
        g["passed"] = phase_query_pass(g)
        result[phase] = g
    return {
        "groups": result,
        "passed": all(g["passed"] for g in result.values()),
        "force_current_frame": force_current,
        "no_model_dependent_cache": True,
        "scope": "Fixed exact-runtime RGB features/routing and association computed before candidate fitting; independent per-phase gates; forced-current stress never changes identity/control invalidity.",
    }
