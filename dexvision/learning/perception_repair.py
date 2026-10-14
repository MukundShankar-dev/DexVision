"""Versioned 5.6 deterministic detector selection and body-centre regression."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from dexvision.evaluation.evaluate_perception import score_split
from dexvision.evaluation.perception_audit import accepted_inventory, feature_parity_diagnostics
from dexvision.evaluation.perception_metrics import CORE_CLASSES, box_iou
from dexvision.logging.visual_stream import digest_file, write_json
from dexvision.perception.body_center_model import component_features, fit_kernel, metric_target, temporal_features
from dexvision.perception.perception_baseline import (
    PerceptionBaseline, model_digest, read_rgb, save_artifact_index,
    training_palette, visual_rows,
)


def training_examples(recipe: dict, rows: list[dict], detector_model: dict) -> tuple[dict, dict]:
    """Pair inferred TRAIN detections with training labels AFTER detection.

    No labels, semantic ids, source ordinals or physical phase enter features.
    The source association and primary weight are supervision only;
    duplicates remain present, with no truth-based inference shortcut.
    """
    baseline = PerceptionBaseline(recipe, detector_model)
    calibration = json.loads((Path(recipe["visual_root"])/"calibration.json").read_text(encoding="utf-8"))
    examples = defaultdict(list)
    missing = Counter()
    history, prior_episode = {}, None
    for row in sorted(rows, key=lambda r: (r["source_episode_id"], r["timestamp"])):
        if row["split"] != "train":
            continue
        if row["source_episode_id"] != prior_episode:
            baseline.reset()
            history.clear()
        prior_episode = row["source_episode_id"]
        rgb = read_rgb(Path(recipe["visual_root"]), row)
        detections = baseline.detector.detect(rgb)
        features_by_index = {}
        tracks = baseline.tracker.update(detections, row["timestamp"])
        for index, track in enumerate(tracks):
            if track.ambiguous:
                continue
            current = component_features(rgb, track.detection)
            features_by_index[index] = (temporal_features(current, row["timestamp"], history.get(track.track_id),
                maximum_gap_s=recipe["tracker"]["maximum_gap_s"]) if recipe.get("temporal_context") else current)
            history[track.track_id] = row["timestamp"], current.copy()
        available = set(features_by_index)
        for annotation in row["annotations"]:
            category = annotation["class_id"]
            if not annotation["visible"] or category not in CORE_CLASSES:
                continue
            candidates = [(box_iou(detections[i].box, annotation["box_xyxy"]), i)
                          for i in available if detections[i].class_id == category]
            if not candidates or max(candidates)[0] < .5:
                missing[category] += 1
                continue
            _, index = max(candidates)
            available.remove(index)
            features = features_by_index[index]
            if recipe.get("metric_target") == "world_height":
                from dexvision.perception.height_model import height_target
                target = height_target(np.asarray(annotation["pose_6d"]["translation_m"]), features, calibration)
            else:
                target = metric_target(np.asarray(annotation["pose_6d"]["translation_m"]), features, calibration)
            examples[category].append((features, target, row["source_episode_id"],
                annotation["object_id"] == row["evaluation_only_source_owner"]["entity_id"]))
    if set(examples) != set(CORE_CLASSES):
        raise ValueError("A core class has no training metric examples.")
    result, audit = {}, {}
    for category, values in examples.items():
        episodes = Counter(v[2] for v in values)
        weights = np.array([(recipe.get("primary_training_weight", 1.) if v[3] else 1.)/episodes[v[2]]
                            for v in values])
        weights *= len(weights)/weights.sum()
        result[category] = (np.stack([v[0] for v in values]), np.stack([v[1] for v in values]), weights)
        audit[category] = {"matched_training_rows": len(values), "missing_training_detections": missing[category],
                           "training_episodes": sorted(episodes), "primary_training_rows": sum(v[3] for v in values),
                           "weighting": "episode-normalized with frozen primary supervision multiplier",
                           "primary_training_weight": recipe.get("primary_training_weight", 1.),
                           "feature_count": result[category][0].shape[1]}
    return result, audit


def train_repair(config: str | Path, recipe: dict) -> dict:
    directory = Path(recipe["model_dir"])
    directory.mkdir(parents=True, exist_ok=False)
    rows = visual_rows(recipe, allowed_splits=("train", "validation"))
    if any(r["split"] == "test" for r in rows):
        raise ValueError("Held-out rows cannot enter repair.")
    write_json(directory/"field_source_inventory.json", accepted_inventory(recipe))
    write_json(directory/"feature_diagnostics.json", feature_parity_diagnostics(recipe))
    fitted = training_palette(recipe, rows)
    button_hues = []
    for row in rows:
        if row["split"] != "train" or not any(a["class_id"] == "button" and a["visible"] for a in row["annotations"]):
            continue
        rgb, masks = read_rgb(Path(recipe["visual_root"]), row, with_training_mask=True)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        for annotation in row["annotations"]:
            if annotation["class_id"] == "button" and annotation["visible"]:
                button_hues.extend(hsv[..., 0][masks == annotation["mask_value"]][::recipe["palette_pixel_stride"]].tolist())
    palette = {**fitted["palette"], "button": float(np.median(button_hues))}
    candidates = []
    for merge_gap in recipe.get("candidate_fragment_merge_gap_px", [None]):
        for hue_offsets in recipe.get("candidate_puck_hue_offsets", [None]):
            for minimum_value in recipe.get("candidate_puck_minimum_value", [None]):
                for minimum_pixels in recipe.get("candidate_minimum_pixels", [recipe["minimum_pixels"]]):
                    for margin in recipe["candidate_hue_margins"]:
                        for saturation in recipe["candidate_puck_minimum_saturation"]:
                            model = {"palette": palette, "hue_margin": margin, "minimum_pixels": minimum_pixels,
                                     "class_minimum_saturation": {"rigid_flat_puck": saturation},
                                     "method": "training_hue_palette_class_saturation_components"}
                            if merge_gap is not None:
                                model["fragment_merge"] = {"rigid_flat_puck": {"maximum_gap_px": merge_gap,
                                    "maximum_extent_px": recipe["fragment_maximum_extent_px"]}}
                            if hue_offsets is not None:
                                model["class_hue_offsets"] = {"rigid_flat_puck": hue_offsets}
                            if minimum_value is not None:
                                model["class_minimum_value"] = {"rigid_flat_puck": minimum_value}
                            if "confidence_area_pixels" in recipe:
                                model["confidence_area_pixels"] = recipe["confidence_area_pixels"]
                            report = score_split(recipe, model, rows, "validation", detector_only=True)
                            write_json(directory/f"detector_p{minimum_pixels}_h{margin}_s{saturation}_v{minimum_value}_o{hue_offsets}_g{merge_gap}.json", {"model": model, "report": report})
                            worst = min(report["breakdowns"]["class_id"][c]["recall"] or 0. for c in CORE_CLASSES)
                            primary = report["primary_object_metrics"]["recall"] or 0.
                            primary_worst = min(report["primary_object_breakdowns"]["class_id"].get(c, {}).get("recall") or 0.
                                                for c in CORE_CLASSES)
                            passing = worst >= .95 and (primary_worst >= .95 if recipe.get("require_primary_gates") else True)
                            if recipe.get("detector_selection") == "passing_gates_then_precision":
                                candidates.append((int(passing), -report["false_positive_components"], primary, worst,
                                                   report["recall"], -margin, -minimum_pixels, -saturation, model))
                            else:
                                candidates.append((worst, report["recall"], -report["false_positive_components"],
                                                   -margin, -saturation, 0, 0, 0, model))
    detector_model = max(candidates, key=lambda row: row[:-1])[-1]
    write_json(directory/"selected_detector.json", detector_model)
    examples, audit = training_examples(recipe, rows, detector_model)
    write_json(directory/"training_examples.json", audit)
    pose_candidates = []
    if recipe.get("metric_target") == "world_height":
        from dexvision.perception.height_model import fit_tree
        for depth in recipe["tree"]["depth_candidates"]:
            for leaf in recipe["tree"]["minimum_leaf_candidates"]:
                models = {category: fit_tree(x, y, w, maximum_depth=depth, minimum_leaf_samples=leaf)
                          for category, (x, y, w) in examples.items()}
                if recipe.get("temporal_context"):
                    for fitted_model in models.values():
                        fitted_model["temporal_context"] = True
                model = {**detector_model, "body_center_models": models, "method": "rgb_component_body_center_world_height_trees"}
                report = score_split(recipe, model, rows, "validation")
                write_json(directory/f"pose_d{depth}_l{leaf}.json", {"model": model, "report": report})
                passes = report["all_numerical_gates_pass"] and (
                    report["all_primary_gates_pass"] if recipe.get("require_primary_gates") else True)
                selection_report = report["primary_object_metrics"] if recipe.get("require_primary_gates") else report
                pose_candidates.append((int(passes),
                    -(selection_report["position_error_p95_m"] or float("inf")),
                    -(report["position_error_median_m"] or float("inf")), -depth, -leaf, model, report))
        extra_candidates = []
        for bandwidth in recipe.get("height_kernel", {}).get("bandwidth_candidates", []):
            for regularization in recipe["height_kernel"]["regularization_candidates"]:
                models = {c: fit_kernel(x, y, bandwidth=bandwidth, regularization=regularization,
                    maximum_centres=recipe["height_kernel"]["maximum_centres"], weights=w)
                    for c, (x,y,w) in examples.items()}
                for fitted_model in models.values():
                    fitted_model.update(kind="deterministic_world_height_kernel", temporal_context=True)
                extra_candidates.append((f"kernel_b{bandwidth}_r{regularization}", models))
        if recipe.get("height_nearest"):
            from dexvision.perception.height_model import fit_nearest
            for neighbours in recipe["height_nearest"]["neighbour_candidates"]:
                models = {c: fit_nearest(x,y,w,neighbours=neighbours) for c,(x,y,w) in examples.items()}
                for fitted_model in models.values():
                    fitted_model["temporal_context"] = True
                extra_candidates.append((f"nearest_k{neighbours}", models))
        for ordinal, (name, models) in enumerate(extra_candidates):
            model = {**detector_model, "body_center_models": models,
                     "method": "rgb_component_body_center_world_height_models"}
            report = score_split(recipe, model, rows, "validation")
            write_json(directory/f"pose_{name}.json", {"model": model, "report": report})
            passes = report["all_numerical_gates_pass"] and report["all_primary_gates_pass"]
            pose_candidates.append((int(passes),
                -(report["primary_object_metrics"]["position_error_p95_m"] or float("inf")),
                -(report["position_error_median_m"] or float("inf")), -1000-ordinal, 0, model, report))
        selected = max(pose_candidates, key=lambda row: row[:5])
        return save_selected(config, recipe, directory, selected)
    for bandwidth in recipe["kernel"]["bandwidth_candidates"]:
        for regularization in recipe["kernel"]["regularization_candidates"]:
            models = {category: fit_kernel(x, y, bandwidth=bandwidth, regularization=regularization,
                       maximum_centres=recipe["kernel"]["maximum_centres"], weights=w)
                      for category, (x, y, w) in examples.items()}
            model = {**detector_model, "body_center_models": models,
                     "method": "rgb_component_kernel_body_center_offsets_depth"}
            report = score_split(recipe, model, rows, "validation")
            name = f"pose_b{bandwidth}_r{regularization}"
            write_json(directory/f"{name}.json", {"model": model, "report": report})
            gates = int(report["all_numerical_gates_pass"])
            p95 = report["position_error_p95_m"] or float("inf")
            median = report["position_error_median_m"] or float("inf")
            pose_candidates.append((gates, -p95, -median, -bandwidth, -regularization, model, report))
    selected = max(pose_candidates, key=lambda row: row[:5])
    return save_selected(config, recipe, directory, selected)


def save_selected(config, recipe, directory, selected):
    model, report = selected[-2:]
    write_json(directory/"selected.json", model)
    write_json(directory/"validation.json", report)
    write_json(directory/"selection.json", {"rule": recipe["candidate_selection"],
        "model_sha256": model_digest(model), "recipe_input_digests": recipe["input_digests"],
        "recipe_sha256": digest_file(Path(config)), "implementation_digests": recipe["implementation_digests"],
        "test_data_used_for_fitting_or_selection": False, "test_rows_decoded": 0,
        "evaluation_scope": "training_validation_only", "training_seeds": "deterministic_no_randomness"})
    save_artifact_index(directory)
    return {"model_dir": directory.as_posix(), "selected_detector": {k:v for k,v in model.items() if k != "body_center_models"},
            "selected_model_parameters": {k:v for k,v in model["body_center_models"]["rigid_cuboid"].items()
                                          if k in ("bandwidth", "regularization", "maximum_depth", "minimum_leaf_samples")},
            "validation_numerical_gates": report["numerical_gates"],
            "recall": report["recall"], "median_error_m": report["position_error_median_m"],
            "p95_error_m": report["position_error_p95_m"], "test_started": False,
            "primary_validation": report["primary_object_metrics"],
            "all_primary_gates_pass": report["all_primary_gates_pass"], "baseline_qualified": False}
