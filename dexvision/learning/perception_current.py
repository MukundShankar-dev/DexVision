"""Frozen current-head reuse and full validation; no optimizer updates."""

from __future__ import annotations
import json
from pathlib import Path
from copy import deepcopy
from dexvision.learning.perception_development import (
    load_repair_recipe,
    development_rows,
)
from dexvision.learning.perception_context import strict_validation
from dexvision.evaluation.perception_dual_queries import score_query_cache
from dexvision.perception.current_localization import ENGINE
from dexvision.perception.perception_baseline import (
    model_digest,
    save_artifact_index,
    load_model,
)
from dexvision.logging.visual_stream import digest_file, write_json


def load_current_recipe(path):
    recipe = load_repair_recipe(path)
    if recipe["engine"] != ENGINE or recipe["feature_count"] != 706:
        raise ValueError("Exact current-RGB localization contract required.")
    return recipe


def source_fit(path):
    result = json.loads(Path(path).read_text())
    if (
        len(result["current_train_loss"]) != 3000
        or len(result["temporal_train_loss"]) != 3000
    ):
        raise ValueError("Both frozen source budgets must finish.")
    if any(
        m["epochs"] != 3000
        for name in ("current_frame_models", "body_center_models")
        for m in result["model"][name].values()
    ):
        raise ValueError("Incomplete source head budget.")
    return result


def select_current(config):
    from dexvision.evaluation.perception_current import score_context

    recipe = load_current_recipe(config)
    output = Path(recipe["model_dir"])
    output.mkdir(parents=True, exist_ok=False)
    sources = [source_fit(p) for p in recipe["frozen_candidates"]]
    seeds = [
        next(iter(v["model"]["current_frame_models"].values()))["seed"] for v in sources
    ]
    if set(seeds) != {3, 7, 11} or len(seeds) != 3:
        raise ValueError("Three independent completed source seeds required.")
    rows = development_rows(recipe)
    candidates = []
    cache = {}
    for path, result, seed in zip(
        recipe["frozen_candidates"], sources, seeds, strict=True
    ):
        before = model_digest(result["model"]["current_frame_models"])
        model = {
            k: deepcopy(v)
            for k, v in result["model"].items()
            if k not in ("body_center_models", "current_frame_models")
        }
        model.update(
            body_center_models=deepcopy(result["model"]["current_frame_models"]),
            feature_version=ENGINE,
            inference_variant=ENGINE,
            localization_mode="current_only_no_motion",
            method="current_RGB706_padded_MLP; calibrated_UV_world_height_decoder; no physical motion estimate",
        )
        assert model_digest(model["body_center_models"]) == before
        print("Current-only full sequence validation seed", seed, flush=True)
        current = {
            **recipe,
            "tracking_mode": "predictive",
            "tracker": model["tracker_config"],
            "_eval_current_feature_cache": cache,
        }
        report = score_context(current, model, rows, "validation")
        query = score_query_cache(
            recipe,
            {**model, "current_frame_models": model["body_center_models"]},
            force_current=True,
        )
        passed = strict_validation(report) and query["passed"]
        report.update(
            strict_validation_pass=passed,
            current_frame_phase_validation=query,
            physical_motion_available=False,
        )
        write_json(
            output / f"candidate_seed{seed}.json",
            {
                "model": model,
                "report": report,
                "strict_pass": passed,
                "source_candidate": path,
                "source_sha256": digest_file(Path(path)),
                "exact_active_weights_sha256": before,
                "neural_weight_updates": 0,
            },
        )
        worst = max(
            [
                g["position_error_p95_m"]
                if g["position_error_p95_m"] is not None
                else 1.0
                for g in report["primary_class_phase"].values()
                if g["visible_eligible_instances"] >= 8
            ]
            + [
                g["p95_m"] if g["p95_m"] is not None else 1.0
                for g in query["groups"].values()
            ]
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
        print(
            "Current seed",
            seed,
            "strict",
            passed,
            "worst_p95",
            worst,
            "primary_p95",
            report["primary_object_metrics"]["position_error_p95_m"],
            flush=True,
        )
    selected = max(candidates, key=lambda r: r[:4])
    model, report = selected[-2:]
    write_json(output / "selected.json", model)
    write_json(output / "validation.json", report)
    write_json(
        output / "selection.json",
        {
            "model_sha256": model_digest(model),
            "recipe_sha256": digest_file(Path(config)),
            "recipe_input_digests": recipe["input_digests"],
            "implementation_digests": recipe["implementation_digests"],
            "test_data_used_for_fitting_or_selection": False,
            "strict_validation_pass": report["strict_validation_pass"],
            "candidate_count": 3,
            "rule": recipe["candidate_selection"],
            "neural_weight_updates": 0,
            "physical_motion_available": False,
        },
    )
    save_artifact_index(output)
    load_model(recipe)
    return report
