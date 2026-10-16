"""Harvest exact revised-frontend TRAIN features and fixed validation queries."""

from __future__ import annotations

from collections import OrderedDict, Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from dexvision.evaluation.perception_metrics import CORE_CLASSES, box_iou
from dexvision.learning.perception_balanced_examples import balanced_examples
from dexvision.learning.perception_development import development_rows
from dexvision.logging.visual_stream import write_json
from dexvision.perception.context_motion import context_features, prepare_context
from dexvision.perception.dual_context import FeatureHistory, current_frame_features
from dexvision.perception.height_model import height_target
from dexvision.perception.perception_baseline import PerceptionBaseline, read_rgb
from dexvision.perception.predictive_tracker import PredictiveTracker


class RGBCache:
    """Bounded current-RGB preprocessing cache; no labels or temporal values."""

    def __init__(self, recipe, detector, maximum_entries=256):
        self.detector = PerceptionBaseline(recipe, detector).detector
        self.root = Path(recipe["visual_root"])
        self.maximum_entries = maximum_entries
        self.entries = OrderedDict()

    def get(self, row):
        key = row["rgb"], row["rgb_sha256"]
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        rgb = read_rgb(self.root, row)
        detections = self.detector.detect(rgb)
        prepared = prepare_context(rgb)
        currents = [
            context_features(rgb, d, 0.0, None, maximum_gap_s=0.51, prepared=prepared)[
                1
            ]
            for d in detections
        ]
        # Stored features depend solely on current verified pixels and detections.
        value = detections, currents
        self.entries[key] = value
        while len(self.entries) > self.maximum_entries:
            self.entries.popitem(last=False)
        return value


def harvested_rows(recipe, detector, split, *, cube_only=False):
    rows = sorted(
        (r for r in development_rows(recipe) if r["split"] == split),
        key=lambda r: (r["source_episode_id"], r["timestamp"]),
    )
    calibration = json.loads(
        (Path(recipe["visual_root"]) / "calibration.json").read_text()
    )
    cache = RGBCache(recipe, detector)
    tracker = PredictiveTracker(**recipe["tracker"], **recipe["predictive_tracker"])
    history = FeatureHistory(recipe["tracker"]["maximum_gap_s"])
    episode = None
    for index, row in enumerate(rows):
        if row["source_episode_id"] != episode:
            tracker.reset()
            history.reset()
        episode = row["source_episode_id"]
        detections, currents = cache.get(row)
        tracks = tracker.update(detections, row["timestamp"])
        # Tracker preserves detection ordering; cached RGB370 has no history.
        if len(tracks) != len(currents) or any(
            t.detection is not d for t, d in zip(tracks, detections, strict=True)
        ):
            raise ValueError("RGB cache/tracker detection order mismatch.")
        features = history.observe(None, tracks, current_contexts=currents)
        if not row.get("phase_window_query", True):
            continue
        # Labels are read only after every RGB feature and routing decision.
        owner = row["evaluation_only_source_owner"]
        phase = next(
            (
                p["phase"]
                for p in owner["phase_intervals"]
                if p["start_frame"] <= row["source_frame_index"] < p["end_frame"]
            ),
            "unlabelled",
        )
        source = owner.get("evaluation_only_parent_episode", episode)
        available = set(range(len(tracks)))
        for label in row["annotations"]:
            c = label["class_id"]
            if (
                not label["visible"]
                or c not in CORE_CLASSES
                or (cube_only and c != "rigid_cuboid")
            ):
                continue
            candidates = [
                (box_iou(tracks[i].detection.box, label["box_xyxy"]), i)
                for i in available
                if tracks[i].detection.class_id == c
            ]
            matched = bool(candidates) and max(candidates)[0] >= 0.5
            x = y = None
            temporal = ambiguous = False
            if matched:
                _, i = max(candidates)
                available.remove(i)
                x, temporal = features[i]
                ambiguous = tracks[i].ambiguous
                y = height_target(
                    np.asarray(label["pose_6d"]["translation_m"]), x, calibration
                )
            yield (
                c,
                x,
                y,
                source,
                phase,
                label["object_id"] == owner["entity_id"],
                temporal,
                ambiguous,
                row["source_frame_index"],
            )
        if (index + 1) % 20000 == 0:
            print("RGB harvest", split, index + 1, "/", len(rows), flush=True)


def harvest_training(recipe, detector):
    temporal, current = defaultdict(list), defaultdict(list)
    missing = Counter()
    streams = [
        (recipe, False),
        (
            {
                **recipe,
                "visual_root": recipe["motion_root"],
                "development_protocol": recipe["motion_protocol"],
                "source_manifest": recipe["motion_manifest"],
            },
            True,
        ),
    ]
    streams += [
        ({**recipe, **stream}, True) for stream in recipe["dense_development_streams"]
    ]
    audit = {}
    for stream, cube_only in streams:
        for (
            c,
            x,
            y,
            source,
            phase,
            primary,
            has_history,
            ambiguous,
            frame,
        ) in harvested_rows(stream, detector, "train", cube_only=cube_only):
            if x is None:
                missing[c] += 1
                continue
            current[c].append((current_frame_features(x), y, source, phase, primary))
            if has_history and not ambiguous:
                temporal[c].append((x, y, source, phase, primary))
    if set(current) != set(CORE_CLASSES) or set(temporal) != set(CORE_CLASSES):
        raise ValueError("Both TRAIN heads require all visual classes.")
    result = {}
    for name, entries in (("current", current), ("temporal", temporal)):
        result[name] = {c: balanced_examples(v) for c, v in entries.items()}
        audit[name] = {
            c: {
                "rows": len(v),
                "primary_rows": sum(e[4] for e in v),
                "physical_sources": len({e[2] for e in v}),
                "role_phase_counts": dict(Counter(str((e[4], e[3])) for e in v)),
            }
            for c, v in entries.items()
        }
    audit.update(
        TRAIN_only=True,
        missing_detections=dict(missing),
        exact_runtime_feature_routing=True,
        no_consumed_test_input=True,
        weighting=recipe["weighting"],
    )
    return result, audit


def unique_queries(values):
    """One denominator per physical source/frame; inconsistent duplicates fail."""
    selected = {}
    for value in values:
        key = value[0], value[3], value[8]
        old = selected.setdefault(key, value)
        if old[4:] != value[4:] or (old[1] is None) != (value[1] is None):
            raise ValueError("Inconsistent repeated physical query metadata.")
        if old[1] is not None and (
            not np.array_equal(old[1], value[1]) or not np.array_equal(old[2], value[2])
        ):
            raise ValueError(
                "Repeated physical query changed exact RGB features/targets."
            )
    return list(selected.values())


def save_query_cache(recipes, detector, output):
    """Fixed VAL associations/features, no model inference or selection."""
    selected = unique_queries(
        v
        for recipe in recipes
        for v in harvested_rows(recipe, detector, "validation", cube_only=True)
        if v[5]
    )
    count = len(selected)
    np.savez_compressed(
        output,
        features=np.stack(
            [v[1] if v[1] is not None else np.zeros(706) for v in selected]
        ),
        targets=np.stack([v[2] if v[2] is not None else np.zeros(3) for v in selected]),
        source=np.asarray([v[3] for v in selected]),
        source_frame=np.asarray([v[8] for v in selected]),
        phase=np.asarray([v[4] for v in selected]),
        matched=np.asarray([v[1] is not None for v in selected]),
        temporal=np.asarray([v[6] for v in selected]),
        ambiguous=np.asarray([v[7] for v in selected]),
    )
    return count


def harvest(config):
    from dexvision.learning.perception_motion_examples import load_context_recipe

    recipe = load_context_recipe(config)
    detector = json.loads(Path(recipe["detector_model"]).read_text())
    detector = {
        k: v
        for k, v in detector.items()
        if k not in ("body_center_models", "current_frame_models")
    }
    recipe = {
        **recipe,
        "tracking_mode": "predictive",
        "tracker": detector["tracker_config"],
    }
    output = Path(recipe["harvest_dir"])
    output.mkdir(parents=True, exist_ok=False)
    examples, audit = harvest_training(recipe, detector)
    np.savez_compressed(
        output / "train_examples.npz",
        **{
            f"{head}_{c}_{name}": v[i]
            for head, classes in examples.items()
            for c, v in classes.items()
            for i, name in enumerate(("features", "targets", "weights"))
        },
    )
    write_json(output / "training_audit.json", audit)
    motion = {
        **recipe,
        "visual_root": recipe["motion_root"],
        "development_protocol": recipe["motion_protocol"],
        "source_manifest": recipe["motion_manifest"],
    }
    streams = [motion] + [
        {**recipe, **stream} for stream in recipe["dense_development_streams"]
    ]
    count = save_query_cache(streams, detector, output / "validation_queries.npz")
    write_json(
        output / "query_audit.json",
        {
            "visible_primary_queries": count,
            "validation_only": True,
            "no_fitted_models": True,
            "deduplicated_physical_source_frame": True,
            "exact_runtime_routing": True,
        },
    )
    from dexvision.perception.perception_baseline import save_artifact_index

    save_artifact_index(output)
    return audit
