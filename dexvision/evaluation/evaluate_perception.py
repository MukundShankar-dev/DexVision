"""Checkpoint 5.6 temporal evaluation and immutable visual review evidence."""

from __future__ import annotations

import json
import platform
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from dexvision.evaluation.perception_audit import accepted_inventory, feature_parity_diagnostics
from dexvision.evaluation.perception_metrics import CORE_CLASSES, breakdowns, match_detections, summarize_instances
from dexvision.logging.visual_stream import digest_file, write_json
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_pose import PoseEstimate, validate_delivery
from dexvision.perception.object_tracker import Track
from dexvision.perception.perception_baseline import (
    PerceptionBaseline, load_model, load_recipe, model_digest, read_rgb,
    save_artifact_index, visual_rows,
)


def synthetic_rejection_check() -> dict:
    """Delivery gates independent of the sampled dataset/control cadence."""
    detection = Detection("rigid_cuboid", (0, 0, 3, 3), np.ones((3, 3), bool), 1.)
    track = Track("visual/0", detection, 1., False, False)
    cases = {"stale": (track, 1.10001), "future": (track, .9),
             "ambiguous": (Track("visual/0", detection, 1., True, False), 1.),
             "low_confidence": (Track("visual/0", Detection("rigid_cuboid", detection.box,
                                 detection.mask, .79), 1., False, False), 1.)}
    rejected = {}
    for name, (candidate, now) in cases.items():
        try:
            validate_delivery(candidate, now=now, minimum_confidence=.8, maximum_age_s=.1)
        except ValueError:
            rejected[name] = True
        else:
            rejected[name] = False
    return {"cases": rejected, "rejection_fraction": sum(rejected.values())/len(rejected)}


def score_split(recipe: dict, model: dict, rows: list[dict], split: str,
                *, output_dir: Path | None = None, detector_only: bool = False) -> dict:
    """Truth is read only AFTER independent inference for association/scoring.

    Ordering preserves source timestamp gaps and resets all history per episode.
    A model is never given the task, source ids, GT box/mask/pose or test labels.
    """
    selected = sorted((r for r in rows if r["split"] == split),
                      key=lambda r: (r["source_episode_id"], r["timestamp"]))
    if not selected:
        raise ValueError(f"Missing {split} visual data.")
    baseline = PerceptionBaseline(recipe, model)
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
                    path = output_dir/f"sequence_{episode_index:02d}.avi"
                    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                                            recipe["review_fps"], (640, 480))
                    if not video.isOpened():
                        raise ValueError("MJPEG video encoder unavailable; temporal review cannot be silently omitted")
                    videos.append({"episode": episode, "condition": row["condition"],
                                   "path": path.name, "timestamps": []})
            rgb = read_rgb(root, row)
            started = time.perf_counter()
            if detector_only:
                tracks = baseline.tracker.update(baseline.detector.detect(rgb), row["timestamp"])
                estimates = tuple(PoseEstimate(t, None, None, None, None, None, None, "not_scored") for t in tracks)
            else:
                estimates = baseline.infer(rgb, row["timestamp"])
            latency = time.perf_counter()-started
            latencies.append(latency)
            # Privileged annotations are used ONLY below this inference boundary.
            matches = match_detections(row["annotations"], estimates)
            owner = row["evaluation_only_source_owner"]
            frame = row["source_frame_index"]
            skill_stratum = next((s["skill_name"] for s in owner["segments"]
                                  if s["start_frame"] <= frame < s["end_frame"]), owner["skill"])
            phase_stratum = next((s["phase"] for s in owner["phase_intervals"]
                                  if s["start_frame"] <= frame < s["end_frame"]), "unlabelled")
            false_positives += len(estimates)-len(matches)
            if previous_timestamp is not None:
                gaps.append(row["timestamp"]-previous_timestamp)
            for i, annotation in enumerate(row["annotations"]):
                if annotation["class_id"] not in CORE_CLASSES:
                    continue
                estimate = estimates[matches[i]] if i in matches else None
                truth = np.asarray(annotation["pose_6d"]["translation_m"])
                position_delta, position_error, velocity_error, yaw_error = None, None, None, None
                identifier = annotation["object_id"]
                identity_switch = False
                prior = previous_truth.get(identifier)
                if estimate is not None and estimate.position is not None:
                    position_delta = (np.asarray(estimate.position)-truth).tolist()
                    position_error = float(np.linalg.norm(position_delta))
                    if prior is not None and estimate.linear_velocity is not None:
                        dt = row["timestamp"]-prior[0]
                        if 0 < dt <= recipe["tracker"]["maximum_gap_s"]:
                            velocity_error = float(np.linalg.norm(np.asarray(estimate.linear_velocity)-(truth-prior[1])/dt))
                    if annotation["class_id"] == "rigid_cuboid" and estimate.yaw is not None:
                        rotation = np.asarray(annotation["pose_6d"]["rotation_matrix"])
                        if rotation[2, 2] >= recipe["diagnostics"]["upright_cosine_min"]:
                            target_yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                            period = np.pi/2 if identifier == "block_small" else np.pi
                            yaw_error = float(abs((estimate.yaw-target_yaw+period/2) % period-period/2))
                if estimate is not None and not estimate.track.ambiguous:
                    old = previous_associations.get(identifier)
                    if old is not None and row["timestamp"]-old[0] <= recipe["tracker"]["maximum_gap_s"]:
                        identity_switch = old[1] != estimate.track.track_id
                    previous_associations[identifier] = (row["timestamp"], estimate.track.track_id)
                previous_truth[identifier] = (row["timestamp"], truth)
                metric_rows.append({
                    "episode": episode, "timestamp": row["timestamp"], "source_frame": row["source_frame_index"],
                    "class_id": annotation["class_id"], "object_id": identifier, "condition": row["condition"],
                    "skill_stratum": skill_stratum,
                    "phase_stratum": phase_stratum,
                    "primary_object": identifier == owner["entity_id"],
                    "eligible": bool(annotation["visible"]), "matched": estimate is not None,
                    "certified_pose": False, "position_error_m": position_error,
                    "position_delta_m": position_delta, "velocity_error_mps": velocity_error,
                    "symmetry_yaw_error_rad": yaw_error, "identity_switch": identity_switch,
                    "reacquired": bool(estimate and estimate.track.reacquired),
                    "rejection_reason": estimate.rejection_reason if estimate else "missing_detection",
                })
            if output_dir is not None:
                overlay = review_overlay(rgb, estimates, row, latency, baseline)
                if video is not None:
                    # Hold previous frame across real source gaps; timestamps on
                    # the video make duplicate exclusions and stale cadence visible.
                    if previous_timestamp is not None and last_overlay is not None:
                        repeats = max(0, round((row["timestamp"]-previous_timestamp)*recipe["review_fps"])-1)
                        for _ in range(repeats):
                            video.write(cv2.cvtColor(last_overlay, cv2.COLOR_RGB2BGR))
                    video.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    videos[-1]["timestamps"].append(row["timestamp"])
                for i in matches:
                    category = row["annotations"][i]["class_id"]
                    key = row["condition"], category
                    if key not in panel_candidates:
                        panel_candidates[key] = (overlay.copy(), row, category)
                frame_outputs.append({"episode": episode, "frame": row["source_frame_index"],
                    "timestamp": row["timestamp"], "latency_s": latency,
                    "estimates": [serialize_estimate(e) for e in estimates]})
                last_overlay = overlay
            prior_episode, previous_timestamp = episode, row["timestamp"]
    finally:
        if video is not None:
            video.release()
    rejection = synthetic_rejection_check()
    report = summarize_instances(metric_rows, isolation_rejection_fraction=rejection["rejection_fraction"])
    report.update(split=split, frame_count=len(selected), source_episode_count=episode_index+1,
        false_positive_components=false_positives, breakdowns=breakdowns(metric_rows, rejection["rejection_fraction"]),
        synthetic_rejection=rejection,
        inference_latency_median_s=float(np.median(latencies)), inference_latency_p95_s=float(np.percentile(latencies, 95)),
        deliveries_exceeding_parent_maximum_age=sum(t > recipe["maximum_age_s"] for t in latencies),
        source_gap_median_s=float(np.median(gaps)) if gaps else None,
        source_gap_maximum_s=max(gaps) if gaps else None,
        source_intervals_exceeding_parent_maximum_age=sum(g > recipe["maximum_age_s"] for g in gaps),
        cadence_claim="offline latency; historical export cannot supply 0.1s maximum-age delivery between sampled frames",
        full_orientation="unsupported", angular_velocity="unsupported", fixture_depth="unsupported",
        semantic_identity="unsupported_visual_ids_only", phase_and_contact="unsupported",
        typed_world_state="blocked_missing_full_pose_semantic_ids_and_sensor_fields",
        pose_claim=("learned_body_center_pixel_offsets_and_world_height; calibrated_ray_decoder; full_orientation_unavailable"
                    if "world_height" in model.get("method", "") else
                    "learned_body_center_pixel_offsets_and_camera_depth; calibrated_ray_decoder; full_orientation_unavailable"
                    if "body_center_models" in model else
                    "upright_nominal_size_silhouette_hypotheses_only; no certified control input"))
    primary_rows = [r for r in metric_rows if r["primary_object"]]
    report["primary_object_metrics"] = summarize_instances(primary_rows,
        isolation_rejection_fraction=rejection["rejection_fraction"])
    report["primary_object_breakdowns"] = breakdowns(primary_rows, rejection["rejection_fraction"])
    report["all_primary_gates_pass"] = (
        report["primary_object_metrics"]["all_numerical_gates_pass"] and all(
            c in report["primary_object_breakdowns"]["class_id"] and
            report["primary_object_breakdowns"]["class_id"][c]["numerical_gates"]["recall"]
            for c in CORE_CLASSES))
    class_reports = report["breakdowns"]["class_id"]
    report["numerical_gates"]["all_core_classes_recall"] = all(
        c in class_reports and class_reports[c]["numerical_gates"]["recall"] for c in CORE_CLASSES)
    report["all_numerical_gates_pass"] = all(report["numerical_gates"].values())
    if output_dir is not None:
        write_json(output_dir/"instance_metrics.json", metric_rows)
        write_json(output_dir/"inferred_frames.json", frame_outputs)
        write_review(output_dir, recipe, panel_candidates, videos)
    return report


def serialize_estimate(estimate: PoseEstimate) -> dict:
    value = asdict(estimate)
    value["track"]["detection"].pop("mask")
    value["source"] = "inferred_perception"
    value["field_availability"] = {"position": "uncertified_hypothesis", "orientation": "unsupported",
                                    "linear_velocity": "hypothesis" if estimate.linear_velocity else "unavailable",
                                    "angular_velocity": "unsupported", "semantic_id": "unsupported"}
    return value


def review_overlay(rgb, estimates, row, latency, baseline):
    overlay = rgb.copy()
    for estimate in estimates:
        detection = estimate.track.detection
        contours, _ = cv2.findContours(detection.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
        x0, y0, x1, y1 = detection.box
        cv2.rectangle(overlay, (x0, y0), (x1-1, y1-1), (255, 60, 60), 1)
        label = f"{estimate.track.track_id} {detection.class_id} c={detection.confidence:.2f}"
        cv2.putText(overlay, label, (max(0, x0-20), max(44, y0-5)), cv2.FONT_HERSHEY_SIMPLEX, .30, (255, 255, 255), 1)
        if estimate.position is not None and estimate.yaw is not None:
            c, s = np.cos(estimate.yaw), np.sin(estimate.yaw)
            axes = np.array([[0., 0., 0.], [.02*c, .02*s, 0.], [-.02*s, .02*c, 0.], [0., 0., .02]])
            projected = baseline.pose.project(axes+estimate.position).astype(int)
            for point, color in zip(projected[1:], ((255, 0, 0), (0, 255, 0), (0, 120, 255))):
                cv2.line(overlay, tuple(projected[0]), tuple(point), color, 1)
        elif estimate.position is not None:
            projected = baseline.pose.project(np.asarray([estimate.position])).astype(int)[0]
            cv2.drawMarker(overlay, tuple(projected), (255, 80, 255), cv2.MARKER_CROSS, 8, 1)
        reason = "ambiguous" if estimate.track.ambiguous else (
            "full orientation/id unavailable" if "body_center_models" in getattr(baseline, "model", {})
            else "REJECT uncertified size/tilt/id")
        cv2.putText(overlay, reason, (max(0, x0-20), min(450, y1+10)), cv2.FONT_HERSHEY_SIMPLEX, .27, (255, 255, 255), 1)
    cv2.rectangle(overlay, (0, 0), (639, 37), (0, 0, 0), -1)
    cv2.putText(overlay, f"{row['source_episode_id']} frame={row['source_frame_index']} t={row['timestamp']:.3f}s",
                (5, 13), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
    cv2.putText(overlay, f"{row['condition']} age/latency={latency:.3f}s; axes=hypothesis, full orientation unavailable",
                (5, 29), cv2.FONT_HERSHEY_SIMPLEX, .33, (255, 255, 255), 1)
    return overlay


def write_review(directory, recipe, panels, videos):
    conditions = recipe["conditions"]
    sheet = np.zeros((len(conditions)*480, len(CORE_CLASSES)*640, 3), np.uint8)
    entries = []
    for i, condition in enumerate(conditions):
        for j, category in enumerate(CORE_CLASSES):
            panel = panels.get((condition, category))
            if panel is None:
                image = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(image, f"MISSING {condition}/{category}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 80, 80), 1)
                entries.append({"condition": condition, "class": category, "status": "missing"})
            else:
                image, row, _ = panel
                entries.append({"condition": condition, "class": category, "status": "present",
                                "episode": row["source_episode_id"], "frame": row["source_frame_index"],
                                "timestamp": row["timestamp"], "rgb": row["rgb"]})
            sheet[i*480:(i+1)*480, j*640:(j+1)*640] = image
    sheet_name = "validation_contact_sheet.png" if recipe.get("review_scope") == "validation" else "heldout_contact_sheet.png"
    if not cv2.imwrite(str(directory/sheet_name), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise ValueError("Cannot write manual contact sheet.")
    write_json(directory/"manual_review.json", {"owner_acceptance": "pending_not_claimed",
        "panels": entries, "videos": videos, "unsupported_strata": recipe["coverage_limits"],
        "scope": recipe.get("review_scope", "test"), "contact_sheet": sheet_name,
        "review_status": "diagnostic_only_until_automated_gates_pass"})


def evaluate_perception(config: str | Path, output_dir: str | Path, *, split: str = "test") -> dict:
    recipe = load_recipe(config)
    if split not in recipe.get("allowed_evaluation_splits", ["test"]):
        raise ValueError("This repair recipe permits training/validation only; held-out access is blocked.")
    model = load_model(recipe)
    rows = (visual_rows(recipe, allowed_splits=("train", "validation")) if split == "validation"
            else visual_rows(recipe))
    directory = Path(output_dir)
    if directory.exists():
        raise ValueError(f"Existing evidence cannot be overwritten: {directory}")
    # Reservation is attached to the frozen method, so changing output path
    # cannot silently repeat held-out evaluation.
    model_directory = Path(recipe["model_dir"])
    reservation = model_directory.with_name(model_directory.name+"_heldout_attempt.json")
    if split == "test":
        with reservation.open("x", encoding="utf-8") as handle:
            json.dump({"model_sha256": model_digest(model), "output_dir": directory.as_posix(),
                       "status": "consumed_started"}, handle, indent=2)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/"frozen_model.json", model)
    write_json(directory/"field_source_inventory.json", accepted_inventory(recipe))
    write_json(directory/"feature_diagnostics.json", feature_parity_diagnostics(recipe))
    started = time.perf_counter()
    tracemalloc.start()
    try:
        report = score_split(recipe, model, rows, split, output_dir=directory)
        _, peak_python_memory = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    automated_pass = (report["all_numerical_gates_pass"] and
                      (report["all_primary_gates_pass"] if recipe.get("require_primary_gates") else True))
    report.update(version=recipe["version"], checkpoint="5.6", checkpoint_complete=False,
                  baseline_qualified=False, model_sha256=model_digest(model),
                  automated_baseline_pass=automated_pass, manual_review_eligible=automated_pass,
                  qualification_scope=f"{split}_only_no_fresh_confirmation" if split == "validation" else "heldout_baseline",
                  recipe_sha256=digest_file(Path(config)), input_digests=recipe["input_digests"],
                  implementation_digests=recipe["implementation_digests"],
                  manual_acceptance="pending_not_claimed", wall_time_s=time.perf_counter()-started,
                  model_bytes=(Path(recipe["model_dir"])/"selected.json").stat().st_size,
                  license="Level4 release generated data CC-BY-4.0; custom workcell and source Apache-2.0; dependencies retain own licenses",
                  hardware={"system": platform.system(), "machine": platform.machine(), "processor": platform.processor(),
                            "rtx_5070_ti": "untested_unavailable", "peak_python_memory_bytes": peak_python_memory,
                            "native_opencv_scipy_peak_memory": "not_measured"})
    write_json(directory/"report.json", report)
    save_artifact_index(directory)
    return report
