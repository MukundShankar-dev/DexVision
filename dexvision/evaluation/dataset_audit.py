"""Read-only Level 4.8 release-candidate audit; never repair source data."""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from dexvision.evaluation.correction_summary import summarize_corrections
from dexvision.evaluation.dataset_coverage import summarize_level4_coverage
from dexvision.evaluation.level4_expert_audit import audit_scripted_episode
from dexvision.evaluation.split_audit import (
    audit_splits, checked_path, content_digest, file_inventory, load_split_config,
    split_manifests, verify_files,
)
from dexvision.logging.corrective_demos import load_level4_6_quarantine
from dexvision.logging.button_amendment import apply_button_replacements
from dexvision.logging.dataset_schema import validate_demo
from dexvision.logging.demo_logger import load_logged_demo
from dexvision.logging.level4_collection import (
    discover_pilot_episodes, load_level4_collection_config,
)
from dexvision.logging.phase_labels import (
    derive_pick_place_segments, phase_disagreement_report, validate_phase_intervals,
)
from dexvision.logging.replay_demo import (
    action_schema_from_metadata, observation_schema_from_metadata, load_replay_demo,
)
from dexvision.logging.session_manifest import load_session_manifest
from dexvision.logging.visual_stream import (
    SPLITS, audit_frame, build_visual_report, digest_file, load_visual_config, primary_entity,
    write_json, configure_render_model, restore_visual_state,
)

AUDIT_VERSION = "level4/dataset-audit-v1"


def apply_puck_replacement(active, *, plan_path: Path, sessions: dict):
    """Apply one hash-bound amendment; missing or altered evidence fails closed.

    The replacement still undergoes the normal fresh replay and split audit.
    The original stays on disk with its original review and exclusion digest.
    """
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    receipt_path = Path(plan["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (plan["version"] != "level4/puck-replacement-plan-v1" or
            receipt["version"] != "level4/puck-replacement-receipt-v1" or
            receipt["plan_sha256"] != digest_file(plan_path) or
            plan["dataset_config_sha256"] != digest_file(Path(plan["dataset_config"]))):
        raise ValueError("Puck replacement plan/receipt/config mismatch")
    originals = [e for e in active if e.episode_id == plan["original_episode_id"]]
    replacements = [e for e in active if e.episode_id == plan["replacement_episode_id"]]
    if len(originals) != 1 or len(replacements) != 1 or originals == replacements:
        raise ValueError("Puck replacement requires distinct, unique accepted episodes")
    original, replacement = originals[0], replacements[0]
    for episode, expected in ((original, plan["original_episode_digest"]),
                              (replacement, receipt["replacement_episode_digest"])):
        session = sessions.get(episode.session_id)
        if (content_digest(file_inventory(episode.path)) != expected or
                episode.goal_condition_id != plan["cell_id"] or
                episode.skill_name != plan["skill"] or episode.source != "scripted" or
                episode.metadata.get("data_stream", "expert_success") != "expert_success" or
                session is None or session.split != plan["split"]):
            raise ValueError("Puck replacement source integrity or ownership mismatch")
    if (plan["split"] != "train" or plan["skill"] != "pick_place_sequence" or
            replacement.session_id != plan["recording_session_id"] or
            replacement.session_id == original.session_id or
            replacement.metadata["random_seed"] != plan["seed"] or
            replacement.metadata["operator_id"] != plan["operator_id"]):
        raise ValueError("Puck replacement does not match the frozen recording recipe")
    exclusion = {"episode_id": original.episode_id, "reason": "superseded_by_replacement",
                 "source": original.source, "review_reasons": [],
                 "episode_digest": plan["original_episode_digest"],
                 "replacement_episode_id": replacement.episode_id}
    hashes = {str(plan_path): digest_file(plan_path), str(receipt_path): digest_file(receipt_path)}
    return [e for e in active if e is not original], exclusion, hashes


def select_audit_episodes(episodes, requirements: dict, quarantined: frozenset):
    """Separate active nominal/corrective evidence from all preserved attempts."""
    prefixes = tuple(requirements["level4_5b_procedural_expansion"]
                     ["diagnostic_quarantine"]["excluded_session_prefixes"])
    active, excluded = [], []
    for episode in episodes:
        if episode.session_id.startswith(prefixes) or episode.episode_id in quarantined:
            reason = "quarantined_diagnostic"
        elif episode.metadata.get("level4_6"):
            active.append(episode)
            continue
        elif episode.expert_accepted and episode.source == "scripted":
            active.append(episode)
            continue
        else:
            reason = "not_accepted_nominal_scripted"
        excluded.append({"episode_id": episode.episode_id, "reason": reason,
                         "source": episode.source,
                         "review_reasons": list(episode.review.rejection_reasons)
                         if episode.review else ["unreviewed"]})
    return active, excluded


def audit_episode(episode, *, root: Path, config_path: Path,
                  workcell_config: Path, requirements: dict) -> tuple[dict, list[str]]:
    """Validate saved schema and independently replay accepted nominal episodes."""
    before = file_inventory(episode.path)
    loaded = load_logged_demo(episode.path)
    metadata = loaded.metadata
    validate_demo(loaded, action_schema=action_schema_from_metadata(metadata),
                  observation_schema=observation_schema_from_metadata(metadata))
    count = len(loaded.timestamps)
    intervals = validate_phase_intervals(metadata["phase_intervals"], frame_count=count,
                                         phases=loaded.online_phases.tolist())
    stream = metadata.get("data_stream", "expert_success")
    issues = []
    quality = {"schema": True}
    if stream == "expert_success":
        audit = audit_scripted_episode(episode.path, config_path=config_path,
                                      workcell_config=workcell_config)
        quality.update(audit.to_dict())
        if not audit.accepted:
            issues.extend(audit.rejection_reasons)
    else:
        # Correction summary independently replays all corrected outcomes and
        # verifies every immutable failure prefix and unsafe abort-only record.
        quality["outcome_audit"] = "correction_summary"
    phase = phase_disagreement_report(loaded.online_phases.tolist(),
                                      loaded.audited_phases.tolist())
    if (stream == "expert_success" and
            phase["disagreement_fraction"] > requirements["quality_thresholds"]
            ["max_phase_annotation_disagreement_fraction"]):
        issues.append("phase_annotation_disagreement")
    quality["phase_annotation"] = phase
    goal = metadata["typed_goal"]
    segments = []
    if stream == "expert_success":
        if episode.skill_name == "pick_place_sequence":
            segments = [s.to_dict() for s in derive_pick_place_segments(intervals,
                                                                       frame_count=count)]
        else:
            segments = [{"skill_name": episode.skill_name, "start_frame": 0,
                         "end_frame": count, "source_phases": sorted(set(loaded.online_phases))}]
    target_interval = ([0, count] if stream == "expert_success" else
                       metadata["intervention_interval"] if stream == "corrective_intervention"
                       else None)
    if target_interval:
        start, end = target_interval
        if np.any(loaded.safety_masks[start:end]):
            issues.append("unsafe_training_target")
    if before != file_inventory(episode.path):
        issues.append("source_changed_during_audit")
    if issues:
        target_interval = None
        segments = []
    return {"episode_id": episode.episode_id, "recording_session_id": episode.session_id,
            "operator_id": metadata["operator_id"], "cell_id": episode.goal_condition_id,
            "skill": episode.skill_name, "stream": stream, "source": episode.source,
            "source_episode_id": metadata.get("source_episode_id"),
            "entity_id": primary_entity(metadata),
            "target_id": goal.get("target_id", goal.get("target_zone")),
            "typed_goal": goal, "outcome": metadata.get("final_outcome", "success"),
            "source_path": episode.path.relative_to(root).as_posix(), "frames": count,
            "phase_intervals": [i.to_dict() for i in intervals], "segments": segments,
            "training_target_interval": target_interval, "audit_passed": not issues, "quality": quality,
            "schema_versions": metadata["schema_versions"],
            "observation_schema": metadata["observation_schema"],
            "action_schema": metadata["action_schema"],
            "action_sha256": before["actions.npy"],
            "initial_state_digest": metadata.get("initial_state_digest"),
            "file_sha256": before, "episode_digest": content_digest(before)}, issues


def audit_annotation_truth(frame: dict, expected: dict) -> list[str]:
    """Compare every claimed pose/id/class against restored simulator truth."""
    issues = []
    for annotation in frame["annotations"]:
        entity = expected.get(annotation["object_id"])
        if entity is None:
            issues.append("unknown annotation identity")
            continue
        for key in ("class_id", "category"):
            if annotation[key] != entity[key]:
                issues.append(f"annotation {key} mismatch")
        pose = annotation["pose_6d"]
        if (not np.allclose(pose["translation_m"], entity["position"], atol=1e-10, rtol=0) or
                not np.allclose(pose["rotation_matrix"], entity["rotation"], atol=1e-10, rtol=0)):
            issues.append("annotation pose differs from saved simulator state")
        visible = annotation["visible_pixels"] > 0
        status = "visible" if visible else "occluded_or_out_of_frame"
        if annotation["visibility"] != status:
            issues.append("annotation visibility status mismatch")
        if annotation["source"] != "simulator_ground_truth" or annotation["confidence"] != 1.0:
            issues.append("annotation truth provenance mismatch")
    return issues


def audit_visual_export(root: Path, records: list[dict], requirements: dict) -> tuple[dict, list]:
    """Decode every RGB/mask and independently check hashes, annotations and splits."""
    import cv2
    import mujoco

    from dexvision.perception.render_annotations import camera_calibration

    files = file_inventory(root)
    config, workcell = load_visual_config(Path("configs/level4_visual_dataset.yaml"))
    if digest_file(root / "export_config.yaml") != digest_file(Path("configs/level4_visual_dataset.yaml")):
        raise ValueError("Frozen visual export config changed")
    if workcell.requirements != requirements:
        raise ValueError("Visual requirements differ from audit requirements")
    sources = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    frames = [json.loads(line) for line in (root / "frames.jsonl").read_text(
        encoding="utf-8").splitlines()]
    saved_report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    approval = json.loads((root / "manual_review_approval.json").read_text(encoding="utf-8"))
    issues = verify_files(root, approval["artifact_sha256"])
    if approval.get("status") != "passed" or approval.get("checkpoint") != "4.7":
        issues.append("Visual manual approval missing")
    if saved_report["config_sha256"] != files["export_config.yaml"]:
        issues.append("Visual export config digest mismatch")
    rows = {row["episode_id"]: row for row in records}
    source_ids = {}
    for source in sources:
        eid = source["episode_id"]
        if eid in source_ids:
            issues.append(f"Duplicate visual source: {eid}")
        source_ids[eid] = source
        row = rows.get(eid)
        if row is None or row["stream"] != "expert_success":
            issues.append(f"Visual source is not active expert: {eid}")
            continue
        for key in ("recording_session_id", "split", "source_path"):
            if source[key] != row[key]:
                issues.append(f"Visual source {key} mismatch: {eid}")
        if source["source_file_sha256"] != row["file_sha256"]:
            issues.append(f"Visual source files changed: {eid}")
    owners, seen, compact = {}, set(), []
    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    width, height = config["camera"]["resolution"]
    if (calibration["image_size"] != [width, height] or
            calibration["version"] != config["camera"]["calibration_version"]):
        issues.append("Visual calibration mismatch")
    held_out = set(requirements["split_policy"]["held_out_object_instances"] +
                   requirements["split_policy"]["held_out_goal_regions"])
    prior_source = None
    for frame in frames:
        eid = frame["source_episode_id"]
        row, source = rows.get(eid), source_ids.get(eid)
        if row is None or source is None:
            issues.append(f"Missing visual source: {eid}")
            continue
        key = (eid, frame["source_frame_index"])
        if key in seen:
            issues.append(f"Duplicate visual frame: {key}")
        seen.add(key)
        for name in ("split", "recording_session_id"):
            if frame[name] != row[name]:
                issues.append(f"Visual {name} leakage: {eid}")
        if (frame["condition"] != source["condition"] or
                frame["condition"] not in config["conditions"]):
            issues.append(f"Visual condition ownership mismatch: {eid}")
        if frame["calibration"] != "calibration.json":
            issues.append(f"Visual calibration reference mismatch: {eid}")
        rgb_path, mask_path = (checked_path(root, frame[k]) for k in ("rgb", "mask"))
        if (files.get(frame["rgb"]) != frame["rgb_sha256"] or
                files.get(frame["mask"]) != frame["mask_sha256"]):
            issues.append(f"Image checksum mismatch: {key}")
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if (rgb is None or mask is None or rgb.shape != (height, width, 3) or
                mask.shape != (height, width) or mask.dtype != np.uint16):
            issues.append(f"Invalid image format: {key}")
            continue
        import hashlib
        pixels = hashlib.sha256(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
        if pixels != frame["rgb_pixel_sha256"]:
            issues.append(f"RGB pixel checksum mismatch: {key}")
        if prior_source != eid:
            loaded = load_replay_demo(Path(row["absolute_source_path"]), model_override=workcell.model_path)
            model = mujoco.MjModel.from_xml_path(str(workcell.model_path))
            entities, _ = configure_render_model(model, loaded.episode.metadata, workcell, config, frame["split"])
            data = mujoco.MjData(model)
            prior_source = eid
        timestamps = loaded.episode.state_timestamps
        try:
            audit_frame({**frame, "rgb_pixel_sha256": pixels}, mask,
                        source_timestamps=timestamps, image_owners=owners)
        except ValueError as exc:
            issues.append(f"Visual {key}: {exc}")
        index = frame["source_frame_index"]
        actions = loaded.episode.action_timestamps
        if not 0 <= index < len(actions) or frame["action_timestamp"] != actions[index]:
            issues.append(f"Visual action alignment mismatch: {key}")
        if 0 <= index < len(timestamps):
            restore_visual_state(model, data, loaded, index)
            truth = {entity.object_id: {"class_id": entity.class_id, "category": entity.category,
                     "position": data.xpos[entity.body_id],
                     "rotation": data.xmat[entity.body_id].reshape(3, 3)} for entity in entities}
            issues.extend(f"Visual {key}: {issue}" for issue in audit_annotation_truth(frame, truth))
            actual_camera = camera_calibration(model, data, config["camera"]["id"], width, height)
            actual_camera["version"] = config["camera"]["calibration_version"]
            if actual_camera != calibration:
                issues.append(f"Visual camera calibration differs from simulator: {key}")
        if index % config["sampling_stride"]:
            issues.append(f"Visual sampling stride mismatch: {key}")
        if not 0 <= frame["condition_parameters"].get("target_removed_fraction", 0) <= 0.35:
            issues.append(f"Visual occlusion exceeds frozen bound: {key}")
        ids = [a["object_id"] for a in frame["annotations"]]
        expected = (set(requirements["workcell"]["objects"]) |
                    set(workcell.targets) | {"start_button"}) - set(source["hidden_visual_entities"])
        if len(ids) != len(set(ids)) or set(ids) != expected:
            issues.append(f"Incomplete visual entity annotations: {key}")
        for annotation in frame["annotations"]:
            visible = annotation["visible_pixels"] > 0
            if annotation["visible"] != visible:
                issues.append(f"Visibility flag mismatch: {key}")
            if frame["split"] != "test" and annotation["object_id"] in held_out:
                issues.append(f"Held-out visual annotation leakage: {key}")
        compact.append({k: frame[k] for k in ("source_episode_id", "source_frame_index",
                       "split", "condition", "rgb", "mask", "rgb_sha256", "mask_sha256")})
    for split in SPLITS:
        saved = json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
        if saved != [frame["rgb"] for frame in frames if frame["split"] == split]:
            issues.append(f"Visual split manifest mismatch: {split}")
    report = build_visual_report(config, workcell, sources, frames)
    if not report["automated_passed"]:
        issues.extend(f"Visual missing cell: {cell}" for cell in report["missing_cells"])
    if files != file_inventory(root):
        issues.append("Visual export changed during audit")
    return {"passed": not issues, "issues": sorted(set(issues)),
            "frame_count": len(frames), "source_count": len(sources), "cells": report["cells"],
            "file_sha256": files, "export_digest": content_digest(files),
            "excluded_duplicate_frames": saved_report["rejected_duplicate_frame_count"],
            "asset_provenance": saved_report["asset_provenance"]}, compact


def summarize_records(records: list[dict]) -> dict:
    """Publish episodes, skill segments and phase intervals without double counting."""
    result = {}
    for kind in ("episodes", "segments", "phase_intervals"):
        grouped = {key: Counter() for key in ("skill", "phase", "session", "object", "target", "outcome")}
        for row in records:
            units = ([{}] if kind == "episodes" else row[kind])
            for unit in units:
                values = {"skill": unit.get("skill_name", row["skill"]),
                          "session": row["recording_session_id"], "object": row["entity_id"],
                          "target": row["target_id"] or "not_applicable", "outcome": row["outcome"]}
                for key, value in values.items():
                    grouped[key][value] += 1
                phases = ({i["phase"] for i in row["phase_intervals"]} if kind == "episodes"
                          else unit.get("source_phases", [unit.get("phase")]))
                grouped["phase"].update(phases)
        result[kind] = {key: dict(sorted(value.items())) for key, value in grouped.items()}
    result["phase_count_note"] = "Episode/segment phase memberships overlap; phase intervals are separate units."
    return result


def _audit_worker(episode, **kwargs):
    """Keep malformed episode failures in the report, even in spawned workers."""
    try:
        return audit_episode(episode, **kwargs)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        return None, [str(exc)]


def audit_level4_dataset(*, config_path: str | Path, splits_path: str | Path,
                         dataset_dir: str | Path, output_dir: str | Path,
                         progress=print) -> dict:
    """Write a new audit directory; failed data produce an amendment, never repairs."""
    config_path, splits_path = Path(config_path), Path(splits_path)
    root, output = Path(dataset_dir), Path(output_dir)
    requirements, _ = load_level4_collection_config(config_path)
    split_config = load_split_config(splits_path, requirements)
    visual_root = Path(split_config["visual_dataset_dir"])
    for protected in (root, visual_root, Path("datasets")):
        if output.resolve().is_relative_to(protected.resolve()):
            raise ValueError("Audit output must be outside source datasets and visual export")
    if output.exists():
        raise ValueError(f"Audit output already exists; choose a new version: {output}")
    implementation_hashes = {p.as_posix(): digest_file(p) for p in sorted(Path("dexvision").rglob("*.py"))}
    sessions = {s.recording_session_id: s for s in
                load_session_manifest(root / "session_manifest.json").sessions}
    episodes = discover_pilot_episodes(root)
    active, excluded = select_audit_episodes(episodes, requirements, load_level4_6_quarantine(root))
    amendment_hashes = {}
    replacement = None
    if split_config.get("replacement_plan"):
        active, replacement, amendment_hashes = apply_puck_replacement(
            active, plan_path=Path(split_config["replacement_plan"]), sessions=sessions)
        excluded.append(replacement)
    button_replacements = []
    if split_config.get("button_replacement_plan"):
        active, button_replacements, hashes = apply_button_replacements(
            active, plan_path=Path(split_config["button_replacement_plan"]), sessions=sessions)
        excluded.extend(button_replacements)
        amendment_hashes.update(hashes)
    replacement_lineage = {r["replacement_episode_id"]: r["episode_id"]
                           for r in [*button_replacements, *([replacement] if replacement else [])]}
    progress(f"Auditing {len(active)} active episodes; preserving {len(excluded)} excluded attempts")
    records, issues = [], []
    worker = partial(_audit_worker, root=root, config_path=config_path,
                     workcell_config=Path(split_config["workcell_config"]), requirements=requirements)
    # Separate spawned processes keep MuJoCo state isolated and work on Windows/macOS.
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        for number, (episode, result) in enumerate(zip(active, pool.map(worker, active), strict=True), 1):
            row, failures = result
            if row is not None:
                session = sessions.get(episode.session_id)
                row["split"] = session.split if session else "missing"
                row["absolute_source_path"] = str(episode.path.resolve())
                if episode.episode_id in replacement_lineage:
                    row["replaces_episode_id"] = replacement_lineage[episode.episode_id]
                records.append(row)
            issues.extend(f"{episode.episode_id}: {failure}" for failure in failures)
            if number % 20 == 0 or number == len(active):
                progress(f"Validated/replayed {number}/{len(active)}; issues={len(issues)}")
    coverage = summarize_level4_coverage(config_path=config_path, dataset_dir=root)
    nominal = coverage["level4_5b_procedural_expansion"]
    if not nominal["automated_requirements_passed"]:
        issues.append("Nominal coverage/independence/scaling audit failed")
    progress("Replaying all corrections and verifying immutable failure prefixes")
    corrections = summarize_corrections(config_path=config_path, dataset_dir=root)
    if not corrections["automated_requirements_passed"]:
        issues.append("Failure/correction audit failed")
    splits = audit_splits(records, sessions, requirements)
    issues.extend(splits["issues"])
    progress("Decoding and auditing every saved visual frame")
    visual, frames = audit_visual_export(visual_root, records, requirements)
    issues.extend(visual["issues"])
    for row in records:
        row.pop("absolute_source_path")
    coverage_cells = []
    for cell in requirements["coverage_cells"]:
        counts = Counter(r["split"] for r in records if r["cell_id"] == cell["id"] and r["audit_passed"])
        saved_counts = Counter(r["split"] for r in records if r["cell_id"] == cell["id"])
        minimum = (10 if cell["data_group"] == "failure_correction" else
                   requirements["level4_5b_procedural_expansion"]["accepted_episodes_per_cell"])
        row = {"cell_id": cell["id"], "owner": cell["split_owner"],
               "by_split": {split: {"observed": counts[split], "saved": saved_counts[split],
                            "minimum": minimum if split == cell["split_owner"] else 0}
                            for split in SPLITS}}
        row["passed"] = counts[cell["split_owner"]] >= minimum and all(
            counts[split] == 0 for split in SPLITS if split != cell["split_owner"])
        if not row["passed"]:
            issues.append(f"Frozen coverage shortage/leakage: {cell['id']}")
        coverage_cells.append(row)
    config_hashes = {str(p): digest_file(p) for p in
                     (config_path, splits_path, Path(split_config["workcell_config"]))}
    config_hashes.update(amendment_hashes)
    config_digest = content_digest(config_hashes)
    if verify_files(Path.cwd(), implementation_hashes):
        issues.append("Audit implementation changed during execution")
    if verify_files(Path.cwd(), amendment_hashes):
        issues.append("Replacement amendment changed during execution")
    manifests = split_manifests(records, frames, config_digest, audit_passed=not issues)
    report = {"version": AUDIT_VERSION, "passed": not issues, "issues": sorted(set(issues)),
              "active_episode_count": len(records), "excluded_episode_count": len(excluded),
              "fresh_episode_audit_pass_count": sum(r["audit_passed"] for r in records),
              "fresh_episode_audit_fail_ids": [r["episode_id"] for r in records if not r["audit_passed"]],
              "stream_counts": dict(Counter(r["stream"] for r in records)),
              "counts": summarize_records(records), "splits": splits,
              "coverage_cells": coverage_cells,
              "nominal_coverage": nominal, "failure_correction_coverage": corrections,
              "visual": visual, "config_sha256": config_hashes, "config_digest": config_digest,
              "dataset_digest": content_digest({r["episode_id"]: r["episode_digest"] for r in records}),
              "exclusions": excluded,
              "replacement_amendment": replacement,
              "button_replacement_amendments": button_replacements,
              "rejection_counts": dict(Counter(reason for r in excluded for reason in r["review_reasons"])),
              "excluded_reason_counts": dict(Counter(r["reason"] for r in excluded)),
              "session_manifest_sha256": digest_file(root / "session_manifest.json"),
              "provenance_file_sha256": {p.name: digest_file(p) for p in sorted(root.glob("*.json"))},
              "implementation_sha256": implementation_hashes,
              "limitations": [
                  "Deterministic simulator experts; no human, cross-operator, real-world or cross-camera claim.",
                  "Single camera compact nominal visual subset; failures/corrections have no rendered stream.",
                  "Raw background simulator entities include held-out ids and must be excluded from learner inputs.",
                  "Pick/place uses qualified isolated fixtures and assisted orientation; no arbitrary grasping claim.",
                  "Shared training-pool objects/goals intentionally recur across sessions; only declared ids/regions are held out.",
                  "Custom workcell asset license remains unspecified; packaging requires owner resolution.",
                  "No normalization fitted, training, release packaging or Level 5 qualification performed."],
              "coverage_exclusions": requirements["coverage_exclusions"]}
    output.mkdir(parents=True, exist_ok=False)
    for split, manifest in manifests.items():
        write_json(output / f"{split}.json", manifest)
    write_json(output / "report.json", report)
    if issues:
        write_json(output / "collection_amendment_v1.json", {
            "version": "level4/audit-collection-amendment-v1", "status": "required_not_collected",
            "parent_dataset_digest": report["dataset_digest"], "issues": sorted(set(issues)),
            "failed_episode_ids": report["fresh_episode_audit_fail_ids"],
            "shortages": [c for c in coverage_cells if not c["passed"]],
            "cross_split_action_groups": splits["cross_split_action_groups"],
            "required_action": "Freeze a new append-only collection amendment for affected cells. "
            "Preserve failed and duplicate trajectories as diagnostic evidence, collect genuinely "
            "distinct replacements under whole-session ownership, and rerun this audit. "
            "Do not change task thresholds or use test outcomes for tuning.",
            "rule": "Preserve all accepted episodes. Any collection/schema/split changes need a new frozen plan version; never repair in place."})
    write_json(output / "file_sha256.json", file_inventory(output))
    return report
