"""Three full dual-head metric fits on separately frozen TRAIN/validation data."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from dexvision.evaluation.perception_dual import score_context
from dexvision.evaluation.perception_dual_queries import score_query_cache
from dexvision.evaluation.perception_metrics import CORE_CLASSES
from dexvision.learning.perception_context import strict_validation
from dexvision.learning.perception_metric_network import fit_network
from dexvision.learning.perception_motion_examples import load_context_recipe
from dexvision.learning.perception_development import development_rows
from dexvision.logging.visual_stream import write_json, digest_file
from dexvision.perception.context_motion import FEATURE_VERSION
from dexvision.perception.dual_context import INFERENCE_VARIANT
from dexvision.perception.perception_baseline import (
    model_digest,
    save_artifact_index,
    load_model,
)


def train_dual(config):
    recipe = load_context_recipe(config)
    if recipe.get("inference_variant") != INFERENCE_VARIANT:
        raise ValueError("Exact dual-head contract required.")
    output = Path(recipe["model_dir"])
    output.mkdir(parents=True, exist_ok=False)
    detector = json.loads(Path(recipe["detector_model"]).read_text())
    detector = {
        k: v
        for k, v in detector.items()
        if k not in ("body_center_models", "current_frame_models")
    }
    current = {
        **recipe,
        "tracking_mode": "predictive",
        "tracker": detector["tracker_config"],
    }
    with np.load(recipe["train_feature_cache"], allow_pickle=False) as cache:
        examples = {
            head: {
                c: tuple(
                    cache[f"{head}_{c}_{name}"].copy()
                    for name in ("features", "targets", "weights")
                )
                for c in CORE_CLASSES
            }
            for head in ("current", "temporal")
        }
    rows = development_rows(recipe)
    calibration = json.loads(
        (Path(recipe["visual_root"]) / "calibration.json").read_text()
    )
    candidates = []
    for spec in recipe["network_candidates"]:
        print("Candidate", spec["seed"], "temporal head", flush=True)
        temporal, temporal_loss = fit_network(
            examples["temporal"], calibration=calibration, **spec
        )
        print("Candidate", spec["seed"], "current RGB head", flush=True)
        single, single_loss = fit_network(
            examples["current"], calibration=calibration, **spec
        )
        model = {
            **detector,
            "body_center_models": temporal,
            "current_frame_models": single,
            "feature_version": FEATURE_VERSION,
            "tracking_mode": "predictive",
            "inference_variant": INFERENCE_VARIANT,
            "method": "separately_trained_current_and_causal_RGB_metric_localization; ambiguous identity stays invalid",
        }
        write_json(
            output / f"weights_seed{spec['seed']}.json",
            {
                "model": model,
                "temporal_train_loss": temporal_loss,
                "current_train_loss": single_loss,
            },
        )
        report = score_context(current, model, rows, "validation")
        query = score_query_cache(current, model)
        stress = score_query_cache(current, model, force_current=True)
        passed = strict_validation(report) and query["passed"] and stress["passed"]
        report.update(
            additional_motion_phase_validation=query,
            current_frame_stress_validation=stress,
            strict_validation_pass=passed,
        )
        name = f"candidate_seed{spec['seed']}"
        write_json(
            output / f"{name}.json",
            {
                "model": model,
                "report": report,
                "strict_pass": passed,
                "temporal_train_loss": temporal_loss,
                "current_train_loss": single_loss,
            },
        )
        worst = max(
            [
                g["position_error_p95_m"] or 1.0
                for g in report["primary_class_phase"].values()
                if g["visible_eligible_instances"] >= 8
            ]
            + [g["p95_m"] or 1.0 for g in query["groups"].values()]
            + [g["p95_m"] or 1.0 for g in stress["groups"].values()]
        )
        print(
            name,
            "strict",
            passed,
            "worst p95",
            worst,
            "primary",
            report["primary_object_metrics"]["position_error_p95_m"],
            flush=True,
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
    selected = max(candidates, key=lambda v: v[:4])
    model, report = selected[-2:]
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
            "candidate_count": len(candidates),
            "fitted_heads_per_candidate": 2,
            "epochs_each": 3000,
        },
    )
    save_artifact_index(output)
    load_model(recipe)
    return report
