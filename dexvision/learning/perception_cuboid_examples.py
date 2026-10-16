"""Cuboid-only TRAIN amendment harvest; other class supervision stays in prior cache."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from dexvision.learning.perception_development import (
    load_repair_recipe,
)
from dexvision.perception.context_motion import (
    NeuralContextBaseline as ContextBaseline,
    context_features,
    prepare_context,
    FEATURE_VERSION,
    FEATURE_COUNT,
)
from dexvision.perception.height_model import height_target
from dexvision.perception.perception_baseline import (
    read_rgb,
)
from dexvision.evaluation.perception_metrics import box_iou


def load_context_recipe(path):
    recipe = load_repair_recipe(path)
    if recipe["engine"] != FEATURE_VERSION or recipe["feature_count"] != FEATURE_COUNT:
        raise ValueError("Exact RGB context feature contract is required.")
    return recipe


def phase_sample_weight(episode, phase, primary, counts, phase_episodes):
    """Equal role/phase totals and equal episodes inside each role/phase.

    Expert phase is supervision for loss weighting only, never an input.
    """
    return (4.0 if primary else 1.0) / (
        counts[episode, phase, primary] * len(phase_episodes[primary, phase])
    )


def training_examples(recipe, rows, detector):
    baseline = ContextBaseline(recipe, detector)
    calibration = json.loads(
        (Path(recipe["visual_root"]) / "calibration.json").read_text()
    )
    values, missing = defaultdict(list), Counter()
    episode, history = None, {}
    for row in sorted(
        (r for r in rows if r["split"] == "train"),
        key=lambda r: (r["source_episode_id"], r["timestamp"]),
    ):
        if row["source_episode_id"] != episode:
            baseline.reset()
            history.clear()
        episode = row["source_episode_id"]
        rgb = read_rgb(Path(recipe["visual_root"]), row)
        tracks = baseline.tracker.update(
            baseline.detector.detect(rgb), row["timestamp"]
        )
        prepared = prepare_context(rgb)
        features = {}
        for index, track in enumerate(tracks):
            if track.ambiguous or track.detection.class_id != "rigid_cuboid":
                continue
            feature, current = context_features(
                rgb,
                track.detection,
                row["timestamp"],
                history.get(track.track_id),
                maximum_gap_s=recipe["tracker"]["maximum_gap_s"],
                prepared=prepared,
            )
            history[track.track_id] = row["timestamp"], current.copy()
            features[index] = feature
        owner = row["evaluation_only_source_owner"]
        phase = next(
            (
                p["phase"]
                for p in owner["phase_intervals"]
                if p["start_frame"] <= row["source_frame_index"] < p["end_frame"]
            ),
            "unlabelled",
        )
        available = set(features)
        for label in row["annotations"]:
            c = label["class_id"]
            if not label["visible"] or c != "rigid_cuboid":
                continue
            candidates = [
                (box_iou(tracks[i].detection.box, label["box_xyxy"]), i)
                for i in available
                if tracks[i].detection.class_id == c
            ]
            if not candidates or max(candidates)[0] < 0.5:
                missing[c] += 1
                continue
            _, index = max(candidates)
            available.remove(index)
            x = features[index]
            target = height_target(
                np.asarray(label["pose_6d"]["translation_m"]), x, calibration
            )
            values[c].append(
                (x, target, episode, phase, label["object_id"] == owner["entity_id"])
            )
    result, audit = {}, {}
    if set(values) != {"rigid_cuboid"}:
        raise ValueError("Missing cuboid TRAIN context examples.")
    for c, entries in values.items():
        counts = Counter((v[2], v[3], v[4]) for v in entries)
        phase_episodes = defaultdict(set)
        for value in entries:
            phase_episodes[value[4], value[3]].add(value[2])
        weights = np.asarray(
            [
                phase_sample_weight(v[2], v[3], v[4], counts, phase_episodes)
                for v in entries
            ]
        )
        weights *= len(weights) / weights.sum()
        x, y = np.stack([v[0] for v in entries]), np.stack([v[1] for v in entries])
        result[c] = x, y, weights
        audit[c] = {
            "rows": len(entries),
            "primary_rows": sum(v[4] for v in entries),
            "missing": missing[c],
            "feature_count": x.shape[1],
            "feature_min": x.min(0).tolist(),
            "feature_max": x.max(0).tolist(),
            "feature_std": x.std(0).tolist(),
            "target_min": y.min(0).tolist(),
            "target_max": y.max(0).tolist(),
            "source": "TRAIN RGB only; equal primary/nonprimary-role x phase x episode loss totals; phase never enters features",
        }
    return result, audit
