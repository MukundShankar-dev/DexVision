"""One-time confirmation with explicit late-phase coverage and unchanged gates."""

from __future__ import annotations
from pathlib import Path
import json
from dexvision.evaluation.perception_current import (
    selected_context,
    score_context,
    select_context_review,
)
from dexvision.evaluation.perception_phase_queries import phase_query_pass
from dexvision.logging.visual_stream import write_json


def fresh_late_evidence(report, instances):
    groups = {}
    for phase in ("release", "settle"):
        g = report.get("primary_class_phase", {}).get("rigid_cuboid/" + phase, {})
        episodes = {
            r["episode"]
            for r in instances
            if r["primary_object"]
            and r["eligible"]
            and r["class_id"] == "rigid_cuboid"
            and r["phase_stratum"] == phase
        }
        result = {
            "visible_queries": g.get("visible_eligible_instances", 0),
            "matches": g.get("detections_matched_iou_0_5", 0),
            "poses": g.get("position_hypothesis_count", 0),
            "visible_physical_source_episodes": len(episodes),
            "median_m": g.get("position_error_median_m"),
            "p95_m": g.get("position_error_p95_m"),
        }
        result["passed"] = phase_query_pass(result)
        groups[phase] = result
    return {
        "groups": groups,
        "passed": all(g["passed"] for g in groups.values()),
        "minimum_visible_per_phase": 8,
        "minimum_physical_sources_per_phase": 4,
        "no_pooled_or_unsupported_late_phase_acceptance": True,
    }


def confirm_current_context(protocol_path, dataset_dir, output_dir):
    from dexvision.learning.perception_context import strict_validation
    from dexvision.evaluation.perception_audit import (
        accepted_inventory,
        feature_parity_diagnostics,
    )
    from dexvision.logging.perception_development import load_protocol, verify_dataset
    from dexvision.logging.visual_stream import digest_file
    from dexvision.perception.perception_baseline import (
        save_artifact_index,
        model_digest,
    )

    path, root, output = Path(protocol_path), Path(dataset_dir), Path(output_dir)
    protocol = load_protocol(path)
    if (
        protocol["dataset_role"] != "confirmation"
        or protocol["selected_engine"] != "level5/current-rgb-localization-v1"
    ):
        raise ValueError("Wrong fresh confirmation engine/role.")
    if protocol.get("required_late_coverage") != {
        "phases": ["release", "settle"],
        "minimum_visible": 8,
        "minimum_sources": 4,
    }:
        raise ValueError(
            "Frozen late-phase confirmation coverage requirements missing."
        )
    if protocol.get("selected_localization_mode") != "current_only_no_motion":
        raise ValueError("Frozen dual-head runtime missing.")
    attempt = path.with_name("confirmation_attempt.json")
    if attempt.exists():
        raise ValueError("Confirmation consumed; output path changes cannot retry.")
    if output.exists():
        raise ValueError("Existing confirmation output.")
    verify_dataset(root, path)
    recipe, model = selected_context(protocol["selected_recipe"])
    if model_digest(model) != protocol["selected_model_sha256"]:
        raise ValueError("Frozen selected model changed.")
    with attempt.open("x") as handle:
        json.dump(
            {
                "protocol_sha256": digest_file(path),
                "model_sha256": model_digest(model),
                "consumed": True,
            },
            handle,
        )
    owners = json.loads((root / "owners.json").read_text())
    rows = [
        json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()
    ]
    cases = {c["episode_id"]: c["session_id"] for c in protocol["cases"]}
    if {r["source_episode_id"] for r in rows} != set(cases):
        raise ValueError("Missing or unexpected frozen confirmation episodes.")
    for row in rows:
        if row["recording_session_id"] != cases[row["source_episode_id"]]:
            raise ValueError("Confirmation session ownership mismatch.")
        if row["split"] != "test":
            raise ValueError("Confirmation has non-test rows.")
        row["evaluation_only_source_owner"] = owners[row["source_episode_id"]]
    output.mkdir(parents=True, exist_ok=False)
    report = score_context(
        {**recipe, "visual_root": str(root), "review_scope": "fresh_confirmation"},
        model,
        rows,
        "test",
        output_dir=output,
    )
    instances = json.loads((output / "instance_metrics.json").read_text())
    late = fresh_late_evidence(report, instances)
    passed = strict_validation(report) and late["passed"]
    report.update(fresh_late_phase_evidence=late, baseline_qualified=False)
    report.update(
        automated_baseline_pass=passed,
        strict_validation_pass=passed,
        model_sha256=model_digest(model),
        recipe_sha256=recipe["_recipe_sha256"],
        manual_acceptance="pending" if passed else "not_triggered",
        checkpoint_complete=False,
        future_checkpoint_started=False,
    )
    report.update(
        accepted_bundles=accepted_inventory(recipe),
        feature_diagnostics=feature_parity_diagnostics(recipe),
    )
    if report["automated_baseline_pass"]:
        select_context_review(output, protocol)
    write_json(output / "report.json", report)
    save_artifact_index(output)
    return report
