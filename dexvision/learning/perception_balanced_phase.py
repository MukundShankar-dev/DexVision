"""Frozen coverage amendment; reuse exact prior TRAIN features, never tests."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from dexvision.learning.perception_metric_network import fit_network
from dexvision.learning.perception_motion_examples import (
    load_context_recipe,
)
from dexvision.learning.perception_balanced_examples import balanced_examples
from dexvision.evaluation.perception_phase_queries import score_phase_queries
from dexvision.learning.perception_context_phase import strict_validation
from dexvision.learning.perception_development import (
    development_rows,
    load_repair_recipe,
)
from dexvision.evaluation.perception_motion import score_context
from dexvision.evaluation.perception_metrics import CORE_CLASSES
from dexvision.perception.context_motion import FEATURE_VERSION
from dexvision.perception.perception_baseline import (
    load_model,
    model_digest,
    save_artifact_index,
)
from dexvision.logging.visual_stream import digest_file, write_json


def save_raw_cache(path, entries):
    path = Path(path)
    if path.exists():
        raise ValueError("Raw TRAIN cache exists; preserve it.")
    np.savez_compressed(
        path,
        features=np.stack([v[0] for v in entries]),
        targets=np.stack([v[1] for v in entries]),
        episode=np.asarray([v[2] for v in entries]),
        phase=np.asarray([v[3] for v in entries]),
        primary=np.asarray([v[4] for v in entries]),
    )


def read_raw_cache(path):
    with np.load(path, allow_pickle=False) as c:
        return list(
            zip(
                c["features"],
                c["targets"],
                c["episode"].tolist(),
                c["phase"].tolist(),
                c["primary"].tolist(),
                strict=True,
            )
        )


def verify_base_cache(base_entries, prior):
    x, y, _ = balanced_examples(base_entries)
    if not np.array_equal(x, prior[0]) or not np.array_equal(y, prior[1]):
        raise ValueError("Reharvest changed original frozen TRAIN features or targets.")


def read_balanced_cache(recipe):
    entries = read_raw_cache(recipe["base_raw_cache"])
    phase = read_raw_cache(recipe["phase_raw_cache"])
    with np.load(recipe["train_feature_cache"], allow_pickle=False) as c:
        previous = {
            k: tuple(c[f"{k}_{n}"].copy() for n in ("features", "targets", "weights"))
            for k in CORE_CLASSES
        }
    verify_base_cache(entries, previous["rigid_cuboid"])
    previous["rigid_cuboid"] = balanced_examples(entries + phase)
    from collections import Counter

    totals = Counter()
    for v, w in zip(entries + phase, previous["rigid_cuboid"][2], strict=True):
        totals[str((v[4], v[3]))] += float(w / previous["rigid_cuboid"][2].sum())
    return previous, {
        "TRAIN_only": True,
        "base_rows": len(entries),
        "additional_queries": len(phase),
        "original_features_and_targets_exact": True,
        "role_phase_mass": dict(totals),
        "weighting": recipe["weighting"],
        "no_validation_or_test_fit": True,
    }


def harvest(config, stream, output):
    from dexvision.learning.perception_balanced_examples import recover_entries

    recipe = load_context_recipe(config)
    if stream not in ("base", "phase"):
        raise ValueError("Unknown TRAIN recovery stream.")
    with np.load(recipe["train_feature_cache"], allow_pickle=False) as c:
        x, y = c["rigid_cuboid_features"].copy(), c["rigid_cuboid_targets"].copy()
    if stream == "phase":
        count = len(x)
        with np.load(recipe["combined_train_feature_cache"], allow_pickle=False) as c:
            if not np.array_equal(
                c["rigid_cuboid_features"][:count], x
            ) or not np.array_equal(c["rigid_cuboid_targets"][:count], y):
                raise ValueError("Phase cache changed original TRAIN prefix.")
            x, y = (
                c["rigid_cuboid_features"][count:].copy(),
                c["rigid_cuboid_targets"][count:].copy(),
            )
        recipe = {
            **recipe,
            "visual_root": recipe["amendment_root"],
            "development_protocol": recipe["amendment_protocol"],
            "source_manifest": recipe["amendment_manifest"],
        }
    rows = development_rows(recipe)
    calibration = json.loads(
        (Path(recipe["visual_root"]) / "calibration.json").read_text()
    )
    values = recover_entries(rows, x, y, calibration)
    save_raw_cache(output, values)
    return {
        "stream": stream,
        "TRAIN_rows": len(values),
        "exact_features_targets_unchanged": True,
        "ordered_source_positions_match": True,
        "no_new_inference": True,
    }


def train_coverage(config):
    recipe = load_context_recipe(config)
    rows = development_rows(recipe)
    output = Path(recipe["model_dir"])
    output.mkdir(parents=True, exist_ok=False)
    base = load_repair_recipe(recipe["detector_recipe"])
    detector = {k: v for k, v in load_model(base).items() if k != "body_center_models"}
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
    examples, audit = read_balanced_cache(recipe)
    write_json(output / "training_audit.json", audit)
    np.savez_compressed(
        output / "train_examples.npz",
        **{
            f"{c}_{name}": v[i]
            for c, v in examples.items()
            for i, name in enumerate(("features", "targets", "weights"))
        },
    )
    calibration = json.loads(
        (Path(recipe["visual_root"]) / "calibration.json").read_text()
    )
    candidates = []
    for spec in recipe["network_candidates"]:
        models, history = fit_network(examples, calibration=calibration, **spec)
        model = {
            **detector,
            "body_center_models": models,
            "feature_version": FEATURE_VERSION,
            "tracking_mode": "predictive",
            "method": "rgb_motion_world_height_mlp_metric_loss",
        }
        report = score_context(current, model, rows, "validation")
        auxiliary = score_phase_queries(amended, model)
        report["additional_phase_validation"] = auxiliary
        passes = strict_validation(report) and auxiliary["passed"]
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
    report["strict_validation_pass"] = (
        strict_validation(report) and report["additional_phase_validation"]["passed"]
    )
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
        },
    )
    save_artifact_index(output)
    load_model(recipe)
    return report
