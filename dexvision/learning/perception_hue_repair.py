"""TRAIN-only hue refinement with exact frozen three-seed MLP weight reuse."""

from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import json
import cv2
import numpy as np
from dexvision.learning.perception_motion_examples import load_context_recipe
from dexvision.learning.perception_development import development_rows
from dexvision.learning.perception_context_phase import strict_validation
from dexvision.evaluation.perception_motion import score_context
from dexvision.evaluation.perception_phase_queries import score_phase_queries
from dexvision.evaluation.perception_late_queries import score_late_queries
from dexvision.perception.perception_baseline import (
    read_rgb,
    model_digest,
    save_artifact_index,
)
from dexvision.logging.visual_stream import write_json, digest_file


def hue_interval(histogram, lower_quantile, upper_quantile, padding):
    counts = np.asarray(histogram)
    if (
        counts.shape != (180,)
        or np.any(counts < 0)
        or not counts.sum()
        or not 0 <= lower_quantile < upper_quantile <= 1
        or padding < 0
    ):
        raise ValueError("Invalid TRAIN hue support.")
    c = np.cumsum(counts)
    n = c[-1]
    a = int(np.searchsorted(c, max(1, lower_quantile * n), side="left")) - 90
    b = int(np.searchsorted(c, max(1, upper_quantile * n), side="left")) - 90
    return [max(-90, a - padding), min(89, b + padding)]


def fit_cube_hue(recipe, streams, palette):
    histogram = np.zeros(180, np.int64)
    count = 0
    for root, rows in streams:
        for row in rows:
            if row["split"] != "train" or not row.get("phase_window_query", True):
                continue
            labels = [
                a
                for a in row["annotations"]
                if a["visible"] and a["class_id"] == "rigid_cuboid"
            ]
            if not labels:
                continue
            rgb, mask = read_rgb(Path(root), row, with_training_mask=True)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            for label in labels:
                allowed = (
                    (mask == label["mask_value"])
                    & (hsv[..., 1] >= recipe["minimum_saturation"])
                    & (hsv[..., 2] >= recipe["minimum_value"])
                )
                h = hsv[..., 0][allowed].astype(float)
                offset = ((h - palette + 90) % 180 - 90).astype(int)
                histogram += np.bincount(offset + 90, minlength=180)
            count += 1
    spec = recipe["cube_hue_fit"]
    interval = hue_interval(histogram, **spec)
    return interval, {
        "TRAIN_only": True,
        "no_validation_pixels_fit": True,
        "RGB_frames": count,
        "eligible_pixels": int(histogram.sum()),
        "signed_hue_histogram": histogram.tolist(),
        "palette_hue": palette,
        "fitted_offsets": interval,
        "fit_spec": spec,
    }


def pose_weights_digest(model):
    return model_digest(model["body_center_models"])


def select_hue(config):
    recipe = load_context_recipe(config)
    output = Path(recipe["model_dir"])
    output.mkdir(parents=True, exist_ok=False)
    rows = development_rows(recipe)
    late = {
        **recipe,
        "visual_root": recipe["late_root"],
        "development_protocol": recipe["late_protocol"],
        "source_manifest": recipe["late_manifest"],
    }
    late_rows = development_rows(late)
    stabil = {
        **recipe,
        "visual_root": recipe["amendment_root"],
        "development_protocol": recipe["amendment_protocol"],
        "source_manifest": recipe["amendment_manifest"],
    }
    parents = [
        json.loads(Path(p).read_text()) for p in recipe["frozen_pose_candidates"]
    ]
    if len(parents) != 3 or {
        next(iter(p["model"]["body_center_models"].values()))["seed"] for p in parents
    } != {3, 7, 11}:
        raise ValueError("Exactly three distinct frozen pose seeds required.")
    first = parents[0]["model"]
    offsets, audit = fit_cube_hue(
        recipe,
        [(recipe["visual_root"], rows), (recipe["late_root"], late_rows)],
        first["palette"]["rigid_cuboid"],
    )
    write_json(output / "hue_fit.json", audit)
    candidates = []
    for path, parent in zip(recipe["frozen_pose_candidates"], parents, strict=True):
        model = deepcopy(parent["model"])
        before = pose_weights_digest(model)
        spec = next(iter(model["body_center_models"].values()))
        seed = spec["seed"]
        if (
            spec["epochs"] != 3000
            or len(parent["train_loss"]) != 3000
            or seed not in [3, 7, 11]
        ):
            raise ValueError("Exact full-budget V22 pose candidate required.")
        model["class_hue_offsets"] = {
            **model.get("class_hue_offsets", {}),
            "rigid_cuboid": offsets,
        }
        assert before == pose_weights_digest(model)
        model["detector_amendment"] = (
            "TRAIN cuboid hue support only; exact frozen V22 pose weights retained"
        )
        current = {
            **recipe,
            "tracking_mode": model["tracking_mode"],
            "tracker": model["tracker_config"],
        }
        s = {
            **stabil,
            "tracking_mode": model["tracking_mode"],
            "tracker": model["tracker_config"],
        }
        late_recipe = {
            **late,
            "tracking_mode": model["tracking_mode"],
            "tracker": model["tracker_config"],
        }
        report = score_context(current, model, rows, "validation")
        report["additional_phase_validation"] = score_phase_queries(s, model)
        report["late_phase_validation"] = score_late_queries(late_recipe, model)
        passed = (
            strict_validation(report)
            and report["additional_phase_validation"]["passed"]
            and report["late_phase_validation"]["passed"]
        )
        report["strict_validation_pass"] = passed
        write_json(
            output / f"candidate_seed{seed}.json",
            {
                "model": model,
                "report": report,
                "strict_pass": passed,
                "neural_weight_updates": 0,
                "parent_candidate": path,
                "parent_candidate_sha256": digest_file(Path(path)),
                "pose_weights_sha256": before,
                "original_completed_epochs": 3000,
            },
        )
        print(
            "seed",
            seed,
            "strict",
            passed,
            "hue_offsets",
            offsets,
            "late",
            report["late_phase_validation"],
            flush=True,
        )
        worst = max(
            g["position_error_p95_m"] or 1.0
            for g in report["primary_class_phase"].values()
            if g["visible_eligible_instances"] >= 8
        )
        candidates.append(
            (
                int(passed),
                -worst,
                -report["primary_object_metrics"]["position_error_p95_m"],
                -len(candidates),
                model,
                report,
            )
        )
    chosen = max(candidates, key=lambda x: x[:4])
    model, report = chosen[-2:]
    write_json(output / "selected.json", model)
    write_json(output / "validation.json", report)
    write_json(
        output / "selection.json",
        {
            "rule": recipe["candidate_selection"],
            "model_sha256": model_digest(model),
            "recipe_sha256": digest_file(Path(config)),
            "recipe_input_digests": recipe["input_digests"],
            "implementation_digests": recipe["implementation_digests"],
            "test_data_used_for_fitting_or_selection": False,
            "strict_validation_pass": report["strict_validation_pass"],
            "candidate_count": 3,
            "neural_weight_updates": 0,
            "pose_weights_sha256": pose_weights_digest(model),
        },
    )
    save_artifact_index(output)
    from dexvision.perception.perception_baseline import load_model

    load_model(recipe)
    return report
