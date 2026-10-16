"""Frozen coverage amendment; reuse exact prior TRAIN features, never tests."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from dexvision.learning.perception_motion import fit_network
from dexvision.learning.perception_motion_examples import (
    load_context_recipe,
    training_examples,
)
from dexvision.learning.perception_context_phase import strict_validation
from dexvision.learning.perception_development import (
    development_rows,
    load_repair_recipe,
)
from dexvision.evaluation.perception_motion import score_context
from dexvision.evaluation.perception_metrics import CORE_CLASSES
from dexvision.perception.context_motion import FEATURE_VERSION, FEATURE_COUNT
from dexvision.perception.perception_baseline import (
    load_model,
    model_digest,
    save_artifact_index,
)
from dexvision.logging.visual_stream import digest_file, write_json


def merge_training_examples(
    previous: dict, amendment: dict, *, amendment_fraction: float
) -> dict:
    """Allocation-weighted parent totals for cuboids; other classes use prior TRAIN.

    Parent-specific role/phase/episode weighting is retained within each half.
    The source outcome never enters this rule. No validation/test rows enter.
    """
    if not 0 < amendment_fraction < 1:
        raise ValueError("Amendment mass must be strictly between zero and one.")
    if set(previous) != set(CORE_CLASSES) or set(amendment) != set(CORE_CLASSES):
        raise ValueError("All core TRAIN classes must be present.")
    for values in (previous, amendment):
        for x, y, w in values.values():
            if (
                x.ndim != 2
                or x.shape[1] != FEATURE_COUNT
                or y.shape != (len(x), 3)
                or w.shape != (len(x),)
                or not len(x)
            ):
                raise ValueError("Exact TRAIN feature/target/weight shapes required.")
            if not all(np.isfinite(v).all() for v in (x, y, w)) or np.any(w <= 0):
                raise ValueError("Invalid TRAIN examples.")
    merged = dict(previous)
    c = "rigid_cuboid"
    a, b = previous[c], amendment[c]
    merged[c] = (
        np.concatenate((a[0], b[0])),
        np.concatenate((a[1], b[1])),
        np.r_[
            (1 - amendment_fraction) * a[2] / a[2].sum(),
            amendment_fraction * b[2] / b[2].sum(),
        ],
    )
    return merged


def train_coverage(config):
    recipe = load_context_recipe(config)
    rows = development_rows(recipe)
    output = Path(recipe["model_dir"])
    output.mkdir(parents=True, exist_ok=False)
    ancestor = load_context_recipe(recipe["train_feature_cache_recipe"])
    for key in (
        "engine",
        "feature_count",
        "tracker",
        "predictive_tracker",
        "detector_fragment_gap_px",
        "minimum_confidence",
        "maximum_age_s",
        "detector_recipe",
    ):
        if recipe[key] != ancestor[key]:
            raise ValueError("Cached TRAIN transform/detector differs: " + key)
    for key in (
        "dexvision/perception/context_motion.py",
        "dexvision/perception/context_motion_network.py",
        "dexvision/learning/perception_motion_examples.py",
    ):
        if (
            recipe["implementation_digests"][key]
            != ancestor["implementation_digests"][key]
        ):
            raise ValueError("Cached TRAIN feature code differs.")
    with np.load(recipe["train_feature_cache"], allow_pickle=False) as cached:
        previous = {
            c: tuple(
                cached[f"{c}_{name}"].copy()
                for name in ("features", "targets", "weights")
            )
            for c in CORE_CLASSES
        }
    parent = load_repair_recipe(recipe["detector_recipe"])
    detector = {
        k: v for k, v in load_model(parent).items() if k != "body_center_models"
    }
    for c, rule in detector["fragment_merge"].items():
        rule["maximum_gap_px"] = recipe["detector_fragment_gap_px"][c]
    current = {
        **recipe,
        "tracking_mode": "predictive",
        "tracker": detector["tracker_config"],
    }
    amended = {
        **current,
        "visual_root": recipe["amendment_root"],
        "development_protocol": recipe["amendment_protocol"],
        "source_manifest": recipe["amendment_manifest"],
    }
    new_rows = development_rows(amended)
    extra, audit = training_examples(amended, new_rows, detector)
    examples = merge_training_examples(
        previous, extra, amendment_fraction=recipe["cuboid_amendment_fraction"]
    )
    write_json(
        output / "training_audit.json",
        {
            "amendment": audit,
            "previous_train_cache_sha256": digest_file(
                Path(recipe["train_feature_cache"])
            ),
            "weighting": recipe["weighting"],
            "no_validation_or_test_fit": True,
            "combined_rows": {c: len(v[0]) for c, v in examples.items()},
        },
    )
    np.savez_compressed(
        output / "train_examples.npz",
        **{
            f"{c}_{name}": v[i]
            for c, v in examples.items()
            for i, name in enumerate(("features", "targets", "weights"))
        },
    )
    candidates = []
    for spec in recipe["network_candidates"]:
        models, history = fit_network(examples, **spec)
        model = {
            **detector,
            "body_center_models": models,
            "feature_version": FEATURE_VERSION,
            "tracking_mode": "predictive",
            "method": "rgb_motion_world_height_mlp",
        }
        report = score_context(current, model, rows, "validation")
        passes = strict_validation(report)
        name = f"mlp_{'_'.join(map(str, spec['hidden']))}_s{spec['seed']}"
        write_json(
            output / f"{name}.json",
            {
                "model": model,
                "report": report,
                "strict_pass": passes,
                "train_loss": history,
            },
        )
        print(
            name,
            "strict=",
            passes,
            "primary_p95=",
            report["primary_object_metrics"]["position_error_p95_m"],
            flush=True,
        )
        worst = max(
            g["position_error_p95_m"] or 1.0
            for g in report["primary_class_phase"].values()
            if g["visible_eligible_instances"] >= 8
        )
        candidates.append(
            (
                int(passes),
                -worst,
                -report["primary_object_metrics"]["position_error_p95_m"],
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
            "recipe_sha256": digest_file(Path(config)),
            "recipe_input_digests": recipe["input_digests"],
            "implementation_digests": recipe["implementation_digests"],
            "test_data_used_for_fitting_or_selection": False,
            "strict_validation_pass": strict_validation(report),
            "candidate_count": len(candidates),
        },
    )
    save_artifact_index(output)
    load_model(recipe)
    return report
