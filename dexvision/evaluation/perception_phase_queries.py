"""Additional phase-window validation; never consumes a confirmation dataset."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from dexvision.learning.perception_development import development_rows
from dexvision.perception.context_motion import NeuralContextBaseline
from dexvision.perception.perception_baseline import read_rgb
from dexvision.evaluation.perception_metrics import match_detections


def score_phase_queries(recipe, model):
    rows = development_rows(recipe)
    selected = sorted(
        (r for r in rows if r["split"] == "validation"),
        key=lambda r: (r["source_episode_id"], r["timestamp"]),
    )
    baseline = NeuralContextBaseline(recipe, model)
    episode = None
    visible = matched = 0
    errors = []
    sources = set()
    for row in selected:
        if row["source_episode_id"] != episode:
            baseline.reset()
        episode = row["source_episode_id"]
        estimates = baseline.infer(
            read_rgb(Path(recipe["visual_root"]), row), row["timestamp"]
        )
        if not row["phase_window_query"]:
            continue
        owner = row["evaluation_only_source_owner"]
        for i, label in enumerate(row["annotations"]):
            if label["object_id"] != owner["entity_id"] or not label["visible"]:
                continue
            visible += 1
            sources.add(owner["evaluation_only_parent_episode"])
            matches = match_detections(row["annotations"], estimates)
            if i not in matches:
                continue
            matched += 1
            e = estimates[matches[i]]
            if e.position is not None:
                errors.append(
                    float(
                        np.linalg.norm(
                            np.asarray(e.position) - label["pose_6d"]["translation_m"]
                        )
                    )
                )
    median = float(np.median(errors)) if errors else None
    p95 = float(np.percentile(errors, 95)) if errors else None
    result = {
        "visible_queries": visible,
        "visible_physical_source_episodes": len(sources),
        "matches": matched,
        "poses": len(errors),
        "median_m": median,
        "p95_m": p95,
        "scope": "additional cuboid stabilization query-only validation; whole physical sources remain split-owned; no test input; original full-sequence identity/other gates remain mandatory",
    }

    result["passed"] = phase_query_pass(result)
    return result


def phase_query_pass(result):
    return (
        result["visible_queries"] >= 8
        and result["visible_physical_source_episodes"] >= 4
        and result["matches"] >= 0.95 * result["visible_queries"]
        and result["poses"] >= 0.99 * result["matches"]
        and result["median_m"] is not None
        and result["p95_m"] is not None
        and result["median_m"] <= 0.010
        and result["p95_m"] <= 0.025
    )
