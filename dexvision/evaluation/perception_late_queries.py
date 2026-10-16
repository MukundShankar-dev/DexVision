"""Strict query-only late-phase validation; no confirmation input permitted."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from dexvision.learning.perception_development import development_rows
from dexvision.perception.context_motion import NeuralContextBaseline
from dexvision.perception.perception_baseline import read_rgb
from dexvision.evaluation.perception_metrics import match_detections
from dexvision.evaluation.perception_phase_queries import phase_query_pass


def late_phase_pass(groups):
    return all(
        p in groups and phase_query_pass(groups[p]) for p in ("release", "settle")
    )


def score_late_queries(recipe, model):
    rows = development_rows(recipe)
    rows = sorted(
        (r for r in rows if r["split"] == "validation"),
        key=lambda r: (r["source_episode_id"], r["timestamp"]),
    )
    baseline = NeuralContextBaseline(recipe, model)
    episode = None
    stats = {
        p: {"visible_queries": 0, "matches": 0, "errors": [], "sources": set()}
        for p in ("release", "settle")
    }
    for row in rows:
        if row["source_episode_id"] != episode:
            baseline.reset()
        episode = row["source_episode_id"]
        estimates = baseline.infer(
            read_rgb(Path(recipe["visual_root"]), row), row["timestamp"]
        )
        if not row["phase_window_query"]:
            continue
        owner = row["evaluation_only_source_owner"]
        phase = next(
            p["phase"]
            for p in owner["phase_intervals"]
            if p["start_frame"] <= row["source_frame_index"] < p["end_frame"]
        )
        if phase not in stats:
            raise ValueError("Undeclared late query phase.")
        s = stats[phase]
        matches = match_detections(row["annotations"], estimates)
        for i, label in enumerate(row["annotations"]):
            if label["object_id"] != owner["entity_id"] or not label["visible"]:
                continue
            s["visible_queries"] += 1
            s["sources"].add(owner["evaluation_only_parent_episode"])
            if i not in matches:
                continue
            s["matches"] += 1
            e = estimates[matches[i]]
            if e.position is not None:
                s["errors"].append(
                    float(
                        np.linalg.norm(
                            np.asarray(e.position) - label["pose_6d"]["translation_m"]
                        )
                    )
                )
    result = {}
    for p, s in stats.items():
        es = s["errors"]
        result[p] = {
            "visible_queries": s["visible_queries"],
            "matches": s["matches"],
            "poses": len(es),
            "visible_physical_source_episodes": len(s["sources"]),
            "median_m": float(np.median(es)) if es else None,
            "p95_m": float(np.percentile(es, 95)) if es else None,
        }
        result[p]["passed"] = phase_query_pass(result[p])
    return {
        "groups": result,
        "passed": late_phase_pass(result),
        "scope": "TRAIN/validation-only query windows; each release/settle group must independently meet unchanged coverage/recall/pose gates; phase labels read only after RGB inference",
    }
