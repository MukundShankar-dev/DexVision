"""Role/phase-balanced TRAIN-only context refit; V12 remains immutable."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from dexvision.evaluation.perception_context import score_context
from dexvision.learning.perception_development import (
    development_rows,
    load_repair_recipe,
)
from dexvision.logging.visual_stream import write_json, digest_file
from dexvision.perception.context_baseline import (
    ContextBaseline,
    context_features,
    prepare_context,
    FEATURE_VERSION,
    FEATURE_COUNT,
)
from dexvision.perception.height_model import height_target, fit_tree
from dexvision.perception.perception_baseline import (
    read_rgb,
    model_digest,
    load_model,
    save_artifact_index,
)
from dexvision.evaluation.perception_metrics import CORE_CLASSES, box_iou


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
            if track.ambiguous:
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
            if not label["visible"] or c not in CORE_CLASSES:
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
    if set(values) != set(CORE_CLASSES):
        raise ValueError("Missing core TRAIN context examples.")
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


def strict_validation(report):
    primary = report["primary_object_metrics"]
    groups = report["primary_class_phase"]
    required = [
        f"{c}/{p}"
        for c in ("rigid_cuboid", "rigid_cylinder")
        for p in ("transport", "place")
    ]
    coverage = all(
        groups.get(k, {}).get("visible_eligible_instances", 0) >= 8 for k in required
    )
    supported = [g for g in groups.values() if g["visible_eligible_instances"] >= 8]
    return (
        report["all_numerical_gates_pass"]
        and report["all_primary_gates_pass"]
        and coverage
        and primary["identity_switches"] == 0
        and primary["position_hypothesis_count"]
        >= 0.99 * primary["detections_matched_iou_0_5"]
        and all(
            g["position_error_p95_m"] is not None
            and g["position_error_p95_m"] <= 0.025
            and g["position_error_median_m"] <= 0.010
            and g["recall"] >= 0.95
            for g in supported
        )
    )


def train_phase_context(config):
    recipe = load_context_recipe(config)
    rows = development_rows(recipe)
    output = Path(recipe["model_dir"])
    output.mkdir(parents=True, exist_ok=False)
    parent = load_repair_recipe(recipe["detector_recipe"])
    detector = {
        k: v for k, v in load_model(parent).items() if k != "body_center_models"
    }
    for rule in detector["fragment_merge"].values():
        rule["maximum_gap_px"] = recipe["detector_fragment_gap_px"]
    candidates = []
    for mode in recipe["tracking_candidates"]:
        current = {
            **recipe,
            "tracking_mode": mode,
            "tracker": detector["tracker_config"],
        }
        examples, audit = training_examples(current, rows, detector)
        write_json(output / f"training_audit_{mode}.json", audit)
        for depth in recipe["context_tree"]["depths"]:
            for leaf in recipe["context_tree"]["leaves"]:
                models = {
                    c: fit_tree(x, y, w, maximum_depth=depth, minimum_leaf_samples=leaf)
                    for c, (x, y, w) in examples.items()
                }
                for model in models.values():
                    model["feature_version"] = FEATURE_VERSION
                model = {
                    **detector,
                    "body_center_models": models,
                    "feature_version": FEATURE_VERSION,
                    "tracking_mode": mode,
                    "method": "rgb_context_world_height_tree",
                }
                report = score_context(current, model, rows, "validation")
                passes = strict_validation(report)
                name = f"{mode}_d{depth}_l{leaf}"
                write_json(
                    output / f"{name}.json",
                    {"model": model, "report": report, "strict_pass": passes},
                )
                print(
                    name,
                    "strict=",
                    passes,
                    "primary_p95=",
                    report["primary_object_metrics"]["position_error_p95_m"],
                    flush=True,
                )
                candidates.append(
                    (
                        int(passes),
                        -max(
                            g["position_error_p95_m"] or 1.0
                            for g in report["primary_class_phase"].values()
                            if g["visible_eligible_instances"] >= 8
                        ),
                        -(
                            report["primary_object_metrics"]["position_error_p95_m"]
                            or 1.0
                        ),
                        -len(candidates),
                        model,
                        report,
                    )
                )
    selected = max(candidates, key=lambda r: r[:4])
    model, report = selected[-2:]
    report["strict_validation_pass"] = strict_validation(report)
    write_json(output / "selected.json", model)
    write_json(output / "validation.json", report)
    write_json(
        output / "selection.json",
        {
            "rule": recipe["candidate_selection"],
            "model_sha256": model_digest(model),
            "recipe_input_digests": recipe["input_digests"],
            "recipe_sha256": digest_file(Path(config)),
            "implementation_digests": recipe["implementation_digests"],
            "test_data_used_for_fitting_or_selection": False,
            "strict_validation_pass": strict_validation(report),
            "candidate_count": len(candidates),
        },
    )
    save_artifact_index(output)
    return report
