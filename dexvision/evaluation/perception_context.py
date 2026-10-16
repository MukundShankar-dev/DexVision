"""Context-only 5.6 scoring, validation and reserved fresh confirmation."""

from __future__ import annotations

import time
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from dexvision.evaluation.evaluate_perception import (
    synthetic_rejection_check,
    review_overlay,
    serialize_estimate,
    write_review,
)
from dexvision.evaluation.perception_confirmation import TypedBaseline
from dexvision.evaluation.perception_metrics import (
    CORE_CLASSES,
    breakdowns,
    match_detections,
    summarize_instances,
)
from dexvision.logging.visual_stream import write_json
from dexvision.perception.object_pose import PoseEstimate
from dexvision.perception.perception_baseline import read_rgb
from dexvision.perception.context_baseline import ContextBaseline
from dexvision.perception.world_estimate import world_estimate_from_dict


def score_context(
    recipe: dict,
    model: dict,
    rows: list[dict],
    split: str,
    *,
    output_dir: Path | None = None,
    detector_only: bool = False,
) -> dict:
    """Truth is read only AFTER independent inference for association/scoring.

    Ordering preserves source timestamp gaps and resets all history per episode.
    A model is never given the task, source ids, GT box/mask/pose or test labels.
    """
    selected = sorted(
        (r for r in rows if r["split"] == split),
        key=lambda r: (r["source_episode_id"], r["timestamp"]),
    )
    if not selected:
        raise ValueError(f"Missing {split} visual data.")
    baseline = ContextBaseline(recipe, model)
    wrapper = TypedBaseline(baseline) if output_dir is not None else None
    root = Path(recipe["visual_root"])
    metric_rows, latencies, gaps, frame_outputs = [], [], [], []
    prior_episode, previous_timestamp = None, None
    previous_truth, previous_associations = {}, {}
    panel_candidates = {}
    video = None
    videos = []
    episode_index = -1
    last_overlay = None
    false_positives = 0
    try:
        for row in selected:
            episode = row["source_episode_id"]
            if episode != prior_episode:
                baseline.reset()
                previous_truth, previous_associations = {}, {}
                previous_timestamp = None
                if video is not None:
                    video.release()
                episode_index += 1
                if output_dir is not None:
                    path = output_dir / f"sequence_{episode_index:02d}.avi"
                    video = cv2.VideoWriter(
                        str(path),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        recipe["review_fps"],
                        (640, 480),
                    )
                    if not video.isOpened():
                        raise ValueError(
                            "MJPEG video encoder unavailable; temporal review cannot be silently omitted"
                        )
                    videos.append(
                        {
                            "episode": episode,
                            "condition": row["condition"],
                            "path": path.name,
                            "timestamps": [],
                        }
                    )
            rgb = read_rgb(root, row)
            started = time.perf_counter()
            if detector_only:
                tracks = baseline.tracker.update(
                    baseline.detector.detect(rgb), row["timestamp"]
                )
                estimates = tuple(
                    PoseEstimate(t, None, None, None, None, None, None, "not_scored")
                    for t in tracks
                )
            else:
                estimates = (
                    wrapper.infer(rgb, row["timestamp"])
                    if wrapper
                    else baseline.infer(rgb, row["timestamp"])
                )
            latency = time.perf_counter() - started
            latencies.append(latency)
            # Privileged annotations are used ONLY below this inference boundary.
            matches = match_detections(row["annotations"], estimates)
            owner = row["evaluation_only_source_owner"]
            frame = row["source_frame_index"]
            skill_stratum = next(
                (
                    s["skill_name"]
                    for s in owner["segments"]
                    if s["start_frame"] <= frame < s["end_frame"]
                ),
                owner["skill"],
            )
            phase_stratum = next(
                (
                    s["phase"]
                    for s in owner["phase_intervals"]
                    if s["start_frame"] <= frame < s["end_frame"]
                ),
                "unlabelled",
            )
            false_positives += len(estimates) - len(matches)
            if previous_timestamp is not None:
                gaps.append(row["timestamp"] - previous_timestamp)
            for i, annotation in enumerate(row["annotations"]):
                if annotation["class_id"] not in CORE_CLASSES:
                    continue
                estimate = estimates[matches[i]] if i in matches else None
                truth = np.asarray(annotation["pose_6d"]["translation_m"])
                position_delta, position_error, velocity_error, yaw_error = (
                    None,
                    None,
                    None,
                    None,
                )
                identifier = annotation["object_id"]
                identity_switch = False
                prior = previous_truth.get(identifier)
                if estimate is not None and estimate.position is not None:
                    position_delta = (np.asarray(estimate.position) - truth).tolist()
                    position_error = float(np.linalg.norm(position_delta))
                    if prior is not None and estimate.linear_velocity is not None:
                        dt = row["timestamp"] - prior[0]
                        if 0 < dt <= recipe["tracker"]["maximum_gap_s"]:
                            velocity_error = float(
                                np.linalg.norm(
                                    np.asarray(estimate.linear_velocity)
                                    - (truth - prior[1]) / dt
                                )
                            )
                    if (
                        annotation["class_id"] == "rigid_cuboid"
                        and estimate.yaw is not None
                    ):
                        rotation = np.asarray(annotation["pose_6d"]["rotation_matrix"])
                        if (
                            rotation[2, 2]
                            >= recipe["diagnostics"]["upright_cosine_min"]
                        ):
                            target_yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                            period = np.pi / 2 if identifier == "block_small" else np.pi
                            yaw_error = float(
                                abs(
                                    (estimate.yaw - target_yaw + period / 2) % period
                                    - period / 2
                                )
                            )
                if estimate is not None and not estimate.track.ambiguous:
                    old = previous_associations.get(identifier)
                    if (
                        old is not None
                        and row["timestamp"] - old[0]
                        <= recipe["tracker"]["maximum_gap_s"]
                    ):
                        identity_switch = old[1] != estimate.track.track_id
                    previous_associations[identifier] = (
                        row["timestamp"],
                        estimate.track.track_id,
                    )
                previous_truth[identifier] = (row["timestamp"], truth)
                metric_rows.append(
                    {
                        "episode": episode,
                        "timestamp": row["timestamp"],
                        "source_frame": row["source_frame_index"],
                        "class_id": annotation["class_id"],
                        "object_id": identifier,
                        "condition": row["condition"],
                        "skill_stratum": skill_stratum,
                        "phase_stratum": phase_stratum,
                        "primary_object": identifier == owner["entity_id"],
                        "eligible": bool(annotation["visible"]),
                        "matched": estimate is not None,
                        "certified_pose": False,
                        "position_error_m": position_error,
                        "position_delta_m": position_delta,
                        "velocity_error_mps": velocity_error,
                        "symmetry_yaw_error_rad": yaw_error,
                        "identity_switch": identity_switch,
                        "reacquired": bool(estimate and estimate.track.reacquired),
                        "rejection_reason": estimate.rejection_reason
                        if estimate
                        else "missing_detection",
                    }
                )
            if output_dir is not None:
                overlay = review_overlay(rgb, estimates, row, latency, baseline)
                if video is not None:
                    # Hold previous frame across real source gaps; timestamps on
                    # the video make duplicate exclusions and stale cadence visible.
                    if previous_timestamp is not None and last_overlay is not None:
                        repeats = max(
                            0,
                            round(
                                (row["timestamp"] - previous_timestamp)
                                * recipe["review_fps"]
                            )
                            - 1,
                        )
                        for _ in range(repeats):
                            video.write(cv2.cvtColor(last_overlay, cv2.COLOR_RGB2BGR))
                    video.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    videos[-1]["timestamps"].append(row["timestamp"])
                for i in matches:
                    category = row["annotations"][i]["class_id"]
                    key = row["condition"], category
                    if key not in panel_candidates:
                        panel_candidates[key] = (overlay.copy(), row, category)
                frame_outputs.append(
                    {
                        "episode": episode,
                        "frame": row["source_frame_index"],
                        "timestamp": row["timestamp"],
                        "latency_s": latency,
                        "estimates": [serialize_estimate(e) for e in estimates],
                    }
                )
                last_overlay = overlay
            prior_episode, previous_timestamp = episode, row["timestamp"]
    finally:
        if video is not None:
            video.release()
    rejection = synthetic_rejection_check()
    report = summarize_instances(
        metric_rows, isolation_rejection_fraction=rejection["rejection_fraction"]
    )
    report.update(
        split=split,
        frame_count=len(selected),
        source_episode_count=episode_index + 1,
        false_positive_components=false_positives,
        breakdowns=breakdowns(metric_rows, rejection["rejection_fraction"]),
        synthetic_rejection=rejection,
        inference_latency_median_s=float(np.median(latencies)),
        inference_latency_p95_s=float(np.percentile(latencies, 95)),
        deliveries_exceeding_parent_maximum_age=sum(
            t > recipe["maximum_age_s"] for t in latencies
        ),
        source_gap_median_s=float(np.median(gaps)) if gaps else None,
        source_gap_maximum_s=max(gaps) if gaps else None,
        source_intervals_exceeding_parent_maximum_age=sum(
            g > recipe["maximum_age_s"] for g in gaps
        ),
        cadence_claim="offline latency; historical export cannot supply 0.1s maximum-age delivery between sampled frames",
        full_orientation="unsupported",
        angular_velocity="unsupported",
        fixture_depth="unsupported",
        semantic_identity="unsupported_visual_ids_only",
        phase_and_contact="unsupported",
        typed_world_state="blocked_missing_full_pose_semantic_ids_and_sensor_fields",
        pose_claim=(
            "learned_body_center_pixel_offsets_and_world_height; calibrated_ray_decoder; full_orientation_unavailable"
            if "world_height" in model.get("method", "")
            else "learned_body_center_pixel_offsets_and_camera_depth; calibrated_ray_decoder; full_orientation_unavailable"
            if "body_center_models" in model
            else "upright_nominal_size_silhouette_hypotheses_only; no certified control input"
        ),
    )
    primary_rows = [r for r in metric_rows if r["primary_object"]]
    report["primary_object_metrics"] = summarize_instances(
        primary_rows, isolation_rejection_fraction=rejection["rejection_fraction"]
    )
    report["primary_object_breakdowns"] = breakdowns(
        primary_rows, rejection["rejection_fraction"]
    )
    class_phase = defaultdict(list)
    for item in primary_rows:
        class_phase[item["class_id"] + "/" + item["phase_stratum"]].append(item)
    report["primary_class_phase"] = {
        key: summarize_instances(
            items, isolation_rejection_fraction=rejection["rejection_fraction"]
        )
        for key, items in class_phase.items()
    }
    report["all_primary_gates_pass"] = report["primary_object_metrics"][
        "all_numerical_gates_pass"
    ] and all(
        c in report["primary_object_breakdowns"]["class_id"]
        and report["primary_object_breakdowns"]["class_id"][c]["numerical_gates"][
            "recall"
        ]
        for c in CORE_CLASSES
    )
    class_reports = report["breakdowns"]["class_id"]
    report["numerical_gates"]["all_core_classes_recall"] = all(
        c in class_reports and class_reports[c]["numerical_gates"]["recall"]
        for c in CORE_CLASSES
    )
    report["all_numerical_gates_pass"] = all(report["numerical_gates"].values())
    if output_dir is not None:
        write_json(output_dir / "instance_metrics.json", metric_rows)
        write_json(output_dir / "inferred_frames.json", frame_outputs)
        write_review(output_dir, recipe, panel_candidates, videos)
        for frame in wrapper.frames:
            world_estimate_from_dict(frame["world"])
            if not frame["accepted_control_state_blocked"]:
                raise ValueError("Partial RGB evidence unexpectedly admitted control.")
        write_json(output_dir / "typed_inferred_frames.json", wrapper.frames)
    return report


def selected_context(config):
    from dexvision.learning.perception_context import load_context_recipe
    from dexvision.perception.perception_baseline import load_model

    recipe = load_context_recipe(config)
    model = load_model(recipe)
    if (
        model["feature_version"] != recipe["engine"]
        or model["tracking_mode"] not in recipe["tracking_candidates"]
    ):
        raise ValueError("Selected context feature/tracking contract mismatch.")
    return {
        **recipe,
        "tracking_mode": model["tracking_mode"],
        "tracker": model["tracker_config"],
    }, model


def validate_context(config, output_dir):
    from dexvision.learning.perception_development import development_rows
    from dexvision.perception.perception_baseline import (
        save_artifact_index,
        model_digest,
    )
    from dexvision.learning.perception_context import strict_validation
    from dexvision.evaluation.perception_audit import (
        accepted_inventory,
        feature_parity_diagnostics,
    )

    recipe, model = selected_context(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = development_rows(recipe)
    report = score_context(recipe, model, rows, "validation", output_dir=output)
    report.update(
        automated_baseline_pass=strict_validation(report),
        strict_validation_pass=strict_validation(report),
        model_sha256=model_digest(model),
        recipe_sha256=recipe["_recipe_sha256"],
        checkpoint_complete=False,
    )
    report.update(
        accepted_bundles=accepted_inventory(recipe),
        feature_diagnostics=feature_parity_diagnostics(recipe),
    )
    write_json(output / "report.json", report)
    save_artifact_index(output)
    return report


def select_context_review(output, protocol):
    """Fixed nominal/boundary slots plus worst-pose episodes; never best-only."""
    review_path = output / "manual_review.json"
    review = json.loads(review_path.read_text())
    rules = protocol["manual_review_sampling"]
    episodes = {
        c["episode_id"]
        for c in protocol["cases"]
        if c["seed"] % 100 in rules["repetitions"]
    }
    metrics = json.loads((output / "instance_metrics.json").read_text())
    for c in CORE_CLASSES:
        available = [
            r
            for r in metrics
            if r["primary_object"]
            and r["eligible"]
            and r["class_id"] == c
            and r["position_error_m"] is not None
        ]
        if available:
            episodes.add(max(available, key=lambda r: r["position_error_m"])["episode"])
    write_json(output / "all_sequences.json", review)
    review["all_sequences_manifest"] = "all_sequences.json"
    review["all_sequence_count"] = len(review["videos"])
    review["videos"] = [v for v in review["videos"] if v["episode"] in episodes]
    review["sampling_rule"] = rules
    write_json(review_path, review)


def confirm_context(protocol_path, dataset_dir, output_dir):
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
        or protocol["selected_engine"] != "level5/rgb-context370-v1"
    ):
        raise ValueError("Wrong fresh confirmation engine/role.")
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
    report.update(
        automated_baseline_pass=strict_validation(report),
        strict_validation_pass=strict_validation(report),
        model_sha256=model_digest(model),
        recipe_sha256=recipe["_recipe_sha256"],
        manual_acceptance="pending" if strict_validation(report) else "not_triggered",
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


def freeze_context_confirmation(config, output_dir):
    import yaml
    from dexvision.learning.perception_context import strict_validation
    from dexvision.evaluation.perception_development import freeze_confirmation

    spec = yaml.safe_load(Path(config).read_text())
    recipe, model = selected_context(spec["selected_recipe"])
    if spec["manual_review_sampling"] != {
        "repetitions": [0, 1],
        "include_worst_primary_pose_per_class": True,
    }:
        raise ValueError(
            "Explicit fixed slots plus worst-pose manual review are required."
        )
    report = json.loads((Path(recipe["model_dir"]) / "validation.json").read_text())
    if not strict_validation(report) or spec["selected_engine"] != recipe["engine"]:
        raise ValueError(
            "Strict context validation/coverage must pass before fresh confirmation."
        )
    return freeze_confirmation(config, output_dir)
