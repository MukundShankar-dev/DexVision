"""One-time 5.6 confirmation; independent typed inference then truth scoring."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from dexvision.evaluation.evaluate_perception import review_overlay, serialize_estimate, synthetic_rejection_check, write_review
from dexvision.evaluation.perception_audit import accepted_inventory, feature_parity_diagnostics
from dexvision.evaluation.perception_metrics import CORE_CLASSES, breakdowns, match_detections, summarize_instances
from dexvision.logging.perception_confirmation import load_protocol, verify_dataset
from dexvision.logging.visual_stream import digest_file, write_json
from dexvision.perception.object_observations import ObjectObservationError
from dexvision.perception.perception_baseline import load_model, load_recipe, model_digest, read_rgb, save_artifact_index
from dexvision.perception.world_estimate import inferred_world_estimate, world_estimate_from_dict


class TypedBaseline:
    """Offline wrapper: only RGB/time enter the frozen model and shared schema.

    The evaluator receives original hypotheses for unchanged metric semantics;
    typed envelopes are recorded before the evaluator can inspect any labels.
    """

    def __init__(self, baseline):
        self.baseline = baseline
        self.frames = []

    def reset(self):
        self.baseline.reset()

    def infer(self, rgb, timestamp):
        poses = self.baseline.infer(rgb, timestamp)
        world = inferred_world_estimate(poses, timestamp)
        try:
            world.require_control_state(now=timestamp)
        except ObjectObservationError:
            blocked = True
        else:
            blocked = False
        self.frames.append({"world": asdict(world), "accepted_control_state_blocked": blocked})
        return poses


def score_confirmation(recipe: dict, rows: list[dict], wrapper: TypedBaseline, directory: Path) -> dict:
    """Original all-visible IoU/conditional pose gates with explicit omissions.

    Evaluation identity association/primary labels never enter RGB tracking,
    the shared typed estimates, inferred motion, or history.
    """
    metrics, latencies, gaps, outputs, videos, panels = [], [], [], [], [], {}
    episode, video, previous_timestamp, last_overlay = None, None, None, None
    truth_history, associations = {}, {}
    false_positives = 0
    try:
        for row in sorted(rows, key=lambda r: (r["source_episode_id"], r["timestamp"])):
            if row["source_episode_id"] != episode:
                if video is not None:
                    video.release()
                episode = row["source_episode_id"]
                wrapper.reset()
                truth_history, associations = {}, {}
                previous_timestamp, last_overlay = None, None
                path = directory / f"sequence_{len(videos):02d}.avi"
                video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), recipe["review_fps"], (640, 480))
                if not video.isOpened():
                    raise ValueError("Review video encoder unavailable.")
                videos.append({"episode": episode, "condition": row["condition"], "path": path.name, "timestamps": []})
            rgb = read_rgb(Path(recipe["visual_root"]), row)
            start = time.perf_counter()
            estimates = wrapper.infer(rgb, row["timestamp"])
            latencies.append(time.perf_counter()-start)
            # Everything below is evaluation/review only, after typed inference.
            owner = row["evaluation_only_source_owner"]
            matches = match_detections(row["annotations"], estimates)
            false_positives += len(estimates)-len(matches)
            phase = next((p["phase"] for p in owner["phase_intervals"]
                          if p["start_frame"] <= row["source_frame_index"] < p["end_frame"]), "unlabelled")
            for i, label in enumerate(row["annotations"]):
                if label["class_id"] not in CORE_CLASSES:
                    continue
                estimate = estimates[matches[i]] if i in matches else None
                identifier = label["object_id"]
                truth = np.asarray(label["pose_6d"]["translation_m"])
                delta = None if estimate is None or estimate.position is None else np.asarray(estimate.position)-truth
                velocity_error, switch = None, False
                previous = truth_history.get(identifier)
                if estimate is not None and estimate.linear_velocity is not None and previous is not None:
                    dt = row["timestamp"]-previous[0]
                    if 0 < dt <= recipe["tracker"]["maximum_gap_s"]:
                        velocity_error = float(np.linalg.norm(np.asarray(estimate.linear_velocity)-(truth-previous[1])/dt))
                if estimate is not None and not estimate.track.ambiguous:
                    old = associations.get(identifier)
                    switch = bool(old is not None and row["timestamp"]-old[0] <= recipe["tracker"]["maximum_gap_s"]
                                  and old[1] != estimate.track.track_id)
                    associations[identifier] = row["timestamp"], estimate.track.track_id
                truth_history[identifier] = row["timestamp"], truth
                metrics.append({"episode": episode, "timestamp": row["timestamp"], "source_frame": row["source_frame_index"],
                    "class_id": label["class_id"], "object_id": identifier, "condition": row["condition"],
                    "skill_stratum": owner["skill"], "phase_stratum": phase,
                    "primary_object": identifier == owner["entity_id"], "eligible": bool(label["visible"]),
                    "matched": estimate is not None, "certified_pose": False,
                    "position_error_m": float(np.linalg.norm(delta)) if delta is not None else None,
                    "position_delta_m": delta.tolist() if delta is not None else None,
                    "velocity_error_mps": velocity_error, "symmetry_yaw_error_rad": None,
                    "identity_switch": switch, "reacquired": bool(estimate and estimate.track.reacquired),
                    "rejection_reason": estimate.rejection_reason if estimate else "missing_detection"})
            overlay = review_overlay(rgb, estimates, row, latencies[-1], wrapper.baseline)
            if previous_timestamp is not None:
                gap = row["timestamp"]-previous_timestamp
                gaps.append(gap)
                for _ in range(max(0, round(gap*recipe["review_fps"])-1)):
                    video.write(cv2.cvtColor(last_overlay, cv2.COLOR_RGB2BGR))
            video.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            videos[-1]["timestamps"].append(row["timestamp"])
            for i in matches:
                category = row["annotations"][i]["class_id"]
                panels.setdefault((row["condition"], category), (overlay.copy(), row, category))
            outputs.append({"episode": episode, "frame": row["source_frame_index"], "timestamp": row["timestamp"],
                            "estimates": [serialize_estimate(e) for e in estimates]})
            previous_timestamp, last_overlay = row["timestamp"], overlay
    finally:
        if video is not None:
            video.release()
    rejection = synthetic_rejection_check()

    def summary(values):
        result = summarize_instances(values, isolation_rejection_fraction=rejection["rejection_fraction"])
        by_class = breakdowns(values, rejection["rejection_fraction"])
        result["numerical_gates"]["all_core_classes_recall"] = all(
            c in by_class["class_id"] and by_class["class_id"][c]["numerical_gates"]["recall"] for c in CORE_CLASSES)
        result["all_numerical_gates_pass"] = all(result["numerical_gates"].values())
        return result, by_class

    report, groups = summary(metrics)
    primary, primary_groups = summary([r for r in metrics if r["primary_object"]])
    report.update(frame_count=len(rows), source_episode_count=len(videos), false_positive_components=false_positives,
                  primary_object_metrics=primary, primary_object_breakdowns=primary_groups,
                  all_primary_gates_pass=primary["all_numerical_gates_pass"], breakdowns=groups,
                  synthetic_rejection=rejection, inference_latency_median_s=float(np.median(latencies)),
                  inference_latency_p95_s=float(np.percentile(latencies, 95)),
                  source_gap_maximum_s=max(gaps) if gaps else None,
                  source_intervals_exceeding_parent_maximum_age=sum(g > recipe["maximum_age_s"] for g in gaps),
                  split="test")
    write_json(directory / "instance_metrics.json", metrics)
    write_json(directory / "inferred_frames.json", outputs)
    write_review(directory, recipe, panels, videos)
    return report


def evaluate_confirmation(protocol_path: str | Path, dataset_dir: str | Path,
                          output_dir: str | Path) -> dict:
    """Reserve once at protocol level; output aliases cannot enable retries."""
    from dexvision.perception.perception_baseline import PerceptionBaseline

    protocol_path, root, directory = Path(protocol_path), Path(dataset_dir), Path(output_dir)
    protocol = load_protocol(protocol_path)
    manifest = verify_dataset(root, protocol_path)
    parent = load_recipe(protocol["parent_recipe"])
    model = load_model(parent)
    if model_digest(model) != protocol["model_sha256"]:
        raise ValueError("Frozen selected confirmation model changed.")
    if directory.exists():
        raise ValueError("Existing confirmation evidence cannot be overwritten.")
    reservation = protocol_path.with_name("confirmation_attempt.json")
    if reservation.exists():
        raise ValueError("This protocol confirmation is consumed; changing output path cannot retry.")
    recipe = {**parent, "visual_root": root.as_posix(), "review_scope": "fresh_confirmation"}
    owners = json.loads((root / "owners.json").read_text())
    rows = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    for row in rows:
        row["evaluation_only_source_owner"] = owners[row["source_episode_id"]]
    if len(rows) != manifest["frame_count"] or len(owners) != protocol["case_count"]:
        raise ValueError("Incomplete confirmation matrix.")
    wrapper = TypedBaseline(PerceptionBaseline(recipe, model))
    directory.mkdir(parents=True, exist_ok=False)
    with reservation.open("x") as handle:
        json.dump({"protocol_sha256": digest_file(protocol_path), "dataset_sha256": digest_file(root / "manifest.json"),
                   "model_sha256": protocol["model_sha256"], "output_dir": directory.as_posix(),
                   "status": "consumed_started"}, handle, indent=2)
    report = score_confirmation(recipe, rows, wrapper, directory)
    truth_count = 0
    for case in protocol["cases"]:
        path = root / "sources" / case["episode_id"] / "evaluation_only_world_estimates.json"
        for raw in json.loads(path.read_text()):
            truth_world = world_estimate_from_dict(raw)
            if truth_world.source != "simulator_ground_truth":
                raise ValueError("Unmarked evaluation-only truth envelope.")
            truth_count += 1
    for frame in wrapper.frames:
        inferred = world_estimate_from_dict(frame["world"])
        if inferred.source != "inferred_perception":
            raise ValueError("Truth producer leaked into inferred envelope.")
    write_json(directory / "typed_inferred_frames.json", wrapper.frames)
    write_json(directory / "field_source_inventory.json", accepted_inventory(parent))
    write_json(directory / "feature_diagnostics.json", feature_parity_diagnostics(parent))
    report.update(checkpoint="5.6", protocol_sha256=digest_file(protocol_path),
                  dataset_manifest_sha256=digest_file(root / "manifest.json"),
                  model_sha256=protocol["model_sha256"], version=protocol["version"],
                  automated_baseline_pass=report["all_numerical_gates_pass"] and report["all_primary_gates_pass"],
                  checkpoint_complete=False, manual_acceptance="pending_fresh_confirmation_review",
                  typed_estimate_frame_count=len(wrapper.frames),
                  typed_truth_source_frame_count=truth_count,
                  typed_schema_roundtrip_pass=True,
                  full_control_state_available_count=sum(not f["accepted_control_state_blocked"] for f in wrapper.frames),
                  typed_schema="level5/world-estimate-v1",
                  typed_schema_claim="shared availability-aware baseline estimate; accepted v1 control remains blocked",
                  qualification_scope="fresh_reset_confirmation_of_previously_exposed_families",
                  unsupported_skills=["reach", "pick", "place", "push", "press"],
                  source_failed_attempts=sum(not x["source_success"] for x in manifest["attempts"]),
                  source_failure_rule="all episodes/frames retained; no controller qualification claimed")
    write_json(directory / "report.json", report)
    save_artifact_index(directory)
    return report
