"""Append-only derived visual exports with source-state and split provenance."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from dexvision.logging.level4_collection import discover_pilot_episodes
from dexvision.logging.replay_demo import load_replay_demo
from dexvision.logging.session_manifest import load_session_manifest
from dexvision.perception.render_annotations import (
    RenderEntity, camera_calibration, camera_rotation, render_frame,
)
from dexvision.sim.workcell import load_workcell_config


SPLITS = ("train", "validation", "test")


def digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_visual_config(path: str | Path) -> tuple[dict, object]:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    workcell = load_workcell_config(path.parent / config["workcell_config"])
    frozen = workcell.requirements["visual_claim"]
    if config["version"] != "level4/visual-dataset-v1":
        raise ValueError("Unsupported visual dataset version.")
    camera = config["camera"]
    if camera["id"] != frozen["camera_id"] or camera["resolution"] != frozen["resolution"]:
        raise ValueError("Visual camera must match the frozen id and resolution.")
    stride = config["sampling_stride"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("sampling_stride must be a positive integer.")
    if set(config["conditions"]) != set(frozen["conditions"]):
        raise ValueError("All four frozen visual conditions are required.")
    for condition, rule in frozen["conditions"].items():
        for split in SPLITS:
            quota = config["selection"]["minimum_episodes_by_condition_split"][split]
            if not isinstance(quota, int) or quota < max(
                rule["minimum_source_episodes_by_split"][split],
                len(config["selection"]["primary_categories"]),
            ):
                raise ValueError("Visual episode quotas cannot weaken frozen minima/coverage.")
    illumination = config["conditions"]["mild_illumination"]
    if not 0.85 <= illumination["intensity_multiplier"] <= 1.15:
        raise ValueError("Illumination outside frozen bounds.")
    if illumination["color_temperature_shift_k"] != 0:
        raise ValueError("v1 freezes neutral color temperature; nonzero shifts unsupported.")
    occlusion = config["conditions"]["partial_occlusion"]
    if not 0 < occlusion["maximum_target_mask_occlusion_fraction"] <= 0.35:
        raise ValueError("Occlusion outside frozen bounds.")
    if not 0 < occlusion["target_box_width_fraction"] <= 0.2:
        raise ValueError("Occluder width must be bounded.")
    distractors = config["conditions"]["bounded_distractors"]
    if not 1 <= len(distractors["positions_m"]) <= 2:
        raise ValueError("Require one or two bounded distractors.")
    size = np.asarray(distractors["half_size_m"], float)
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError("Distractor sizes must be finite positive 3D half sizes.")
    for position in distractors["positions_m"]:
        p = np.asarray(position, float)
        if p.shape != (3,) or not np.all(np.isfinite(p)) or not (
            0.18 <= p[0]-size[0] and p[0]+size[0] <= 0.21 and
            0.14 <= abs(p[1])-size[1] and abs(p[1])+size[1] <= 0.17
        ):
            raise ValueError("Distractors must stay in the frozen peripheral non-task strip.")
    return config, workcell


def primary_entity(metadata: dict) -> str:
    goal = metadata["typed_goal"]
    return next(str(goal[key]) for key in ("object_id", "entity_id", "button_id") if key in goal)


def select_visual_sources(episodes, sessions: dict, config: dict, requirements: dict) -> list:
    """Select whole source episodes deterministically, without inspecting images."""
    cells = {c["id"]: c for c in requirements["coverage_cells"]}
    excluded = tuple(requirements["level4_5b_procedural_expansion"]
                     ["diagnostic_quarantine"]["excluded_session_prefixes"])
    families = {k: v["family"] for k, v in requirements["workcell"]["objects"].items()}
    buckets = defaultdict(list)
    seen = set()
    for episode in sorted(episodes, key=lambda item: item.episode_id):
        if episode.episode_id in seen:
            raise ValueError(f"Duplicate source episode: {episode.episode_id}")
        seen.add(episode.episode_id)
        if not episode.expert_accepted or episode.source != "scripted":
            continue
        if episode.session_id.startswith(excluded):
            continue
        cell = cells.get(episode.goal_condition_id)
        if not cell or cell["data_group"] == "failure_correction":
            continue
        split = sessions.get(episode.session_id)
        if split != cell["split_owner"]:
            raise ValueError(f"Session/cell split mismatch: {episode.episode_id}")
        entity = primary_entity(episode.metadata)
        category = families.get(entity, entity)
        if split != "test" and entity in requirements["split_policy"]["held_out_object_instances"]:
            raise ValueError("Held-out source object in non-test split.")
        buckets[split, category].append(episode)
    selected = []
    used_sessions = set()
    categories = config["selection"]["primary_categories"]
    for condition in config["conditions"]:
        for split in SPLITS:
            count = config["selection"]["minimum_episodes_by_condition_split"][split]
            for index in range(count):
                category = categories[index % len(categories)]
                candidates = buckets[split, category]
                while candidates and candidates[0].session_id in used_sessions:
                    candidates.pop(0)
                if not candidates:
                    raise ValueError(f"Missing source cell: {condition}/{split}/{category}")
                episode = candidates.pop(0)
                used_sessions.add(episode.session_id)
                selected.append((condition, split, category, episode))
    # Render every training condition before validation, then test.
    # Held-out pixels can never decide which training frames are retained.
    return sorted(selected, key=lambda item: SPLITS.index(item[1]))


def named_joint_values(model, layout, row: np.ndarray, *, velocity: bool) -> np.ndarray:
    """Resolve all MuJoCo joint coordinates by saved names, never packed offsets."""
    import mujoco

    if layout.source_array != "robot_states":
        raise ValueError("Joint layouts must reference robot_states.")
    indices = (list(range(*layout.column_range)) if layout.column_range is not None
               else list(layout.column_indices))
    if len(indices) != len(layout.names) or len(set(layout.names)) != len(indices):
        raise ValueError("Joint coordinate names are missing or duplicated.")
    values = dict(zip(layout.names, row[indices], strict=True))
    expected = []
    for joint_id in range(model.njnt):
        joint = model.joint(joint_id)
        if int(joint.type[0]) == int(mujoco.mjtJoint.mjJNT_FREE):
            suffix = ("vx", "vy", "vz", "wx", "wy", "wz") if velocity else (
                "x", "y", "z", "qw", "qx", "qy", "qz")
        elif int(joint.type[0]) == int(mujoco.mjtJoint.mjJNT_BALL):
            suffix = ("wx", "wy", "wz") if velocity else ("qw", "qx", "qy", "qz")
        else:
            suffix = ()
        expected.extend([f"{joint.name}/{s}" for s in suffix] if suffix else [joint.name])
    if set(expected) != set(values):
        raise ValueError("Saved named joint layout disagrees with the render model.")
    result = np.asarray([values[name] for name in expected])
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite saved joint state.")
    return result


def configure_render_model(model, metadata: dict, workcell, config: dict, split: str):
    """Restore saved static poses/scale, then apply explicit visual-only filtering."""
    import mujoco

    initial = metadata["task_config"]["initial_state"]
    for name in (*workcell.targets, *workcell.fixtures):
        model.body(name).pos[:] = initial["entity_positions_m"][name]
    variation = metadata["task_config"].get("procedural_variation", {})
    entity = primary_entity(metadata)
    if entity in workcell.object_ids:
        spec = next(s for s in workcell.objects if s.object_id == entity)
        model.geom(spec.geom).size[:] *= float(variation.get("object_scale_multiplier", 1.))
    if metadata.get("skill_name") in {"pick_object", "pick_place_sequence"}:
        for name in ("fixture_wall_geom", "start_button_geom"):
            model.geom(name).rgba[3] = 0
    camera = config["camera"]
    camera_model = model.camera(camera["id"])
    camera_model.pos[:] = camera["position_m"]
    rotation = camera_rotation(camera["position_m"], camera["lookat_m"])
    mujoco.mju_mat2Quat(camera_model.quat, rotation.reshape(9))
    camera_model.fovy[:] = camera["vertical_fov_deg"]
    # The scene contains annotation cues and held-out background instances.
    # Hide their render geoms only. The saved qpos and raw episodes stay intact.
    hidden = set()
    if split != "test":
        policy = workcell.requirements["split_policy"]
        hidden.update(policy["held_out_object_instances"])
        hidden.update(policy["held_out_goal_regions"])
    for body_id in range(model.nbody):
        name = model.body(body_id).name
        if name in hidden or name.startswith("pilot_"):
            model.geom_rgba[model.geom_bodyid == body_id, 3] = 0
    entities = []
    specs = [(s.object_id, s.class_id, s.family, s.body) for s in workcell.objects]
    specs += [(name, "button", "start_button", raw["body"])
              for name, raw in workcell.fixtures.items()]
    specs += [(name, "placement_target", "target", raw["body"])
              for name, raw in workcell.targets.items()]
    for name, class_id, category, body in specs:
        if name in hidden:
            continue
        body_id = model.body(body).id
        geoms = tuple(int(g) for g in np.flatnonzero(model.geom_bodyid == body_id))
        entities.append(RenderEntity(name, class_id, category, body_id, geoms))
    return tuple(entities), sorted(hidden)


def restore_visual_state(model, data, loaded, index: int) -> float:
    """Render the exact recorded resulting state, without integrating actions."""
    import mujoco

    episode = loaded.episode
    layouts = loaded.observation_schema.layouts
    row = episode.robot_states[index]
    data.qpos[:] = named_joint_values(model, layouts["robot_qpos"], row, velocity=False)
    data.qvel[:] = named_joint_values(model, layouts["robot_qvel"], row, velocity=True)
    timestamp = float(episode.state_timestamps[index])
    if not np.isfinite(timestamp) or abs(timestamp-float(episode.timestamps[index])) > 1e-9:
        raise ValueError("State/image timestamp alignment failed.")
    data.time = timestamp
    mujoco.mj_forward(model, data)
    return timestamp


def audit_frame(frame: dict, masks: np.ndarray, *, source_timestamps: np.ndarray,
                image_owners: dict) -> None:
    index = frame["source_frame_index"]
    if not 0 <= index < len(source_timestamps) or frame["timestamp"] != source_timestamps[index]:
        raise ValueError("Frame has invalid source timestamp/index alignment.")
    previous = image_owners.setdefault(frame["rgb_pixel_sha256"], frame["split"])
    if previous != frame["split"]:
        raise ValueError("Duplicate image across splits.")
    from dexvision.perception.render_annotations import mask_box

    valid_values = {0}
    for annotation in frame["annotations"]:
        value = annotation["mask_value"]
        if value in valid_values:
            raise ValueError("Duplicate instance mask value.")
        valid_values.add(value)
        mask = masks == value
        if (annotation["box_xyxy"] != mask_box(mask) or
                annotation["visible_pixels"] != int(mask.sum())):
            raise ValueError("Mask/box/visibility disagreement.")
        pose = annotation["pose_6d"]
        rotation = np.asarray(pose["rotation_matrix"])
        if (rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3)) or
                not np.isclose(np.linalg.det(rotation), 1.) or
                not np.all(np.isfinite(pose["translation_m"]))):
            raise ValueError("Invalid metric pose.")
    if not set(np.unique(masks)).issubset(valid_values):
        raise ValueError("Mask contains an unknown instance.")


def _write_png(path: Path, pixels: np.ndarray, *, rgb: bool = False) -> None:
    import cv2

    if rgb:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), pixels):
        raise ValueError(f"Could not write image: {path}")


def export_visual_dataset(*, config_path: str | Path, dataset_dir: str | Path,
                          output_dir: str | Path, progress=print) -> dict:
    """Export the frozen compact matrix atomically to a new directory only."""
    import mujoco

    config_path, dataset_dir, output_dir = map(Path, (config_path, dataset_dir, output_dir))
    if output_dir.exists():
        raise ValueError(f"Output already exists; use a new version: {output_dir}")
    if output_dir.resolve().is_relative_to(dataset_dir.resolve()):
        raise ValueError("Visual output must be outside the source episode tree.")
    config, workcell = load_visual_config(config_path)
    code_root = Path(__file__).resolve().parents[2]
    code_hashes = {name: digest_file(code_root / name) for name in (
        "dexvision/logging/visual_stream.py", "dexvision/perception/render_annotations.py",
        "dexvision/apps/export_visual_dataset.py",
    )}
    sessions = {s.recording_session_id: s.split for s in
                load_session_manifest(dataset_dir / "session_manifest.json").sessions}
    selected = select_visual_sources(discover_pilot_episodes(dataset_dir), sessions,
                                     config, workcell.requirements)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    frame_estimate = sum((int(ep.metadata["num_steps"])+config["sampling_stride"]-1)
                         // config["sampling_stride"] for _, _, _, ep in selected)
    width, height = config["camera"]["resolution"]
    progress(f"Frozen plan: {len(selected)} source episodes, {frame_estimate} frames; "
             f"uncompressed RGB+masks upper estimate {frame_estimate*width*height*5:,} bytes.")
    with tempfile.TemporaryDirectory(prefix=".visual-staging-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        sources, frames, sheets, rejected_frames = [], [], [], []
        image_owners = {}
        calibration_reference = None
        for number, (condition, split, category, source) in enumerate(selected, 1):
            loaded = load_replay_demo(source.path, model_override=workcell.model_path)
            source_hashes = {p.name: digest_file(p) for p in sorted(source.path.iterdir())
                             if p.is_file()}
            model = mujoco.MjModel.from_xml_path(str(workcell.model_path))
            entities, hidden = configure_render_model(model, source.metadata, workcell, config, split)
            if condition == "mild_illumination":
                multiplier = config["conditions"][condition]["intensity_multiplier"]
                model.light_diffuse[:] *= multiplier
                model.light_ambient[:] *= multiplier
                model.light_specular[:] *= multiplier
            data = mujoco.MjData(model)
            destination = staging / split / source.episode_id
            destination.mkdir(parents=True)
            source_info = {
                "episode_id": source.episode_id, "recording_session_id": source.session_id,
                "source_path": source.path.relative_to(dataset_dir).as_posix(),
                "split": split, "condition": condition, "primary_category": category,
                "goal_entity": primary_entity(source.metadata),
                "source_file_sha256": source_hashes, "hidden_visual_entities": hidden,
                "frame_count": len(loaded.episode.timestamps),
            }
            sources.append(source_info)
            sheet_candidate = None
            with mujoco.Renderer(model, height=height, width=width) as renderer:
                for index in range(0, len(loaded.episode.timestamps), config["sampling_stride"]):
                    timestamp = restore_visual_state(model, data, loaded, index)
                    calibration = camera_calibration(model, data, config["camera"]["id"], width, height)
                    calibration["version"] = config["camera"]["calibration_version"]
                    if calibration_reference is None:
                        calibration_reference = calibration
                    if calibration != calibration_reference:
                        raise ValueError("Fixed camera calibration changed across frames.")
                    rgb, masks, annotations, parameters = render_frame(
                        renderer, model, data, camera=config["camera"]["id"], entities=entities,
                        condition=condition, parameters=config["conditions"][condition],
                        goal_entity=primary_entity(source.metadata))
                    relative = destination.relative_to(staging) / f"{index:06d}"
                    frame = {
                        "source_episode_id": source.episode_id, "source_frame_index": index,
                        "recording_session_id": source.session_id, "timestamp": timestamp,
                        "action_timestamp": float(loaded.episode.action_timestamps[index]),
                        "split": split, "condition": condition, "condition_parameters": parameters,
                        "rgb": relative.as_posix()+"_rgb.png", "mask": relative.as_posix()+"_mask.png",
                        "rgb_pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                        "calibration": "calibration.json", "annotations": annotations,
                    }
                    prior_owner = image_owners.get(frame["rgb_pixel_sha256"])
                    if prior_owner is not None and prior_owner != split:
                        rejected_frames.append({
                            "source_episode_id": source.episode_id, "source_frame_index": index,
                            "timestamp": timestamp, "split": split, "condition": condition,
                            "reason": "exact_rgb_duplicate_across_splits",
                            "retained_split": prior_owner,
                            "rgb_pixel_sha256": frame["rgb_pixel_sha256"],
                        })
                        continue
                    audit_frame(frame, masks, source_timestamps=loaded.episode.state_timestamps,
                                image_owners=image_owners)
                    _write_png(staging / frame["rgb"], rgb, rgb=True)
                    _write_png(staging / frame["mask"], masks)
                    frame["rgb_sha256"] = digest_file(staging / frame["rgb"])
                    frame["mask_sha256"] = digest_file(staging / frame["mask"])
                    frames.append(frame)
                    goal = next(a for a in annotations if a["object_id"] == source_info["goal_entity"])
                    if sheet_candidate is None or goal["visible_pixels"] > sheet_candidate[0]:
                        sheet_candidate = (goal["visible_pixels"], frame, rgb.copy(), masks.copy())
            if source_hashes != {p.name: digest_file(p) for p in sorted(source.path.iterdir())
                                 if p.is_file()}:
                raise ValueError("Source episode changed during export.")
            if sheet_candidate is None:
                raise ValueError(f"Source episode has no unique exported frames: {source.episode_id}")
            sheets.append((source_info, sheet_candidate))
            progress(f"Rendered {number}/{len(selected)}: {condition}/{split}/{source.episode_id}")
        report = build_visual_report(config, workcell, sources, frames)
        report["candidate_frame_count"] = frame_estimate
        report["rejected_duplicate_frame_count"] = len(rejected_frames)
        write_json(staging / "excluded_frames.json", rejected_frames)
        report["uncompressed_rgb_mask_bytes"] = len(frames)*width*height*5
        report["payload_bytes"] = sum(p.stat().st_size for p in staging.rglob("*.png"))
        report["implementation_sha256"] = code_hashes
        report["config_sha256"] = digest_file(config_path)
        report["requirements_sha256"] = digest_file(workcell.requirements_path)
        report["mujoco_version"] = mujoco.__version__
        report["asset_provenance"] = asset_provenance(workcell)
        write_json(staging / "calibration.json", calibration_reference)
        write_json(staging / "sources.json", sources)
        with (staging / "frames.jsonl").open("w", encoding="utf-8") as handle:
            for frame in frames:
                handle.write(json.dumps(frame, sort_keys=True, allow_nan=False)+"\n")
        for split in SPLITS:
            write_json(staging / f"{split}.json", [f["rgb"] for f in frames if f["split"] == split])
        create_contact_sheets(staging, sheets)
        shutil.copyfile(config_path, staging / "export_config.yaml")
        report["total_bytes_before_report"] = sum(p.stat().st_size for p in staging.rglob("*")
                                                 if p.is_file())
        write_json(staging / "report.json", report)
        # No existing output, source, archive, or release is overwritten.
        if output_dir.exists():
            raise ValueError("Output appeared during export; refusing replacement.")
        staging.rename(output_dir)
    return report


def build_visual_report(config, workcell, sources, frames) -> dict:
    cells, missing = [], []
    for condition, rule in workcell.requirements["visual_claim"]["conditions"].items():
        for split in SPLITS:
            exported_ids = {f["source_episode_id"] for f in frames}
            source_ids = {s["episode_id"] for s in sources
                          if s["condition"] == condition and s["split"] == split
                          and s["episode_id"] in exported_ids}
            subset = [f for f in frames if f["source_episode_id"] in source_ids]
            visible = {a["category"] for f in subset for a in f["annotations"] if a["visible"]}
            visible_ids = {a["object_id"] for f in subset for a in f["annotations"] if a["visible"]}
            expected = set(rule["entity_coverage"]) - {"all_targets"}
            targets = set(workcell.targets) if "all_targets" in rule["entity_coverage"] else set()
            if split != "test":
                targets -= set(workcell.requirements["split_policy"]["held_out_goal_regions"])
            occluded = any(f["condition_parameters"].get("target_removed_fraction", 0) > 0
                           for f in subset)
            passed = (len(source_ids) >= rule["minimum_source_episodes_by_split"][split] and
                      expected <= visible and targets <= visible_ids and
                      (condition != "partial_occlusion" or occluded))
            cell = {"condition": condition, "split": split, "source_episodes": len(source_ids),
                    "minimum": rule["minimum_source_episodes_by_split"][split],
                    "frames": len(subset), "visible_categories": sorted(visible),
                    "missing_categories": sorted(expected-visible),
                    "missing_targets": sorted(targets-visible_ids), "passed": passed}
            cells.append(cell)
            if not passed:
                missing.append(f"{condition}/{split}")
    return {"version": config["version"], "automated_passed": not missing,
            "manual_verification": "pending_user_contact_sheet_review", "checkpoint_complete": False,
            "source_episode_count": len(sources), "frame_count": len(frames),
            "cells": cells, "missing_cells": missing,
            "split_frame_counts": dict(Counter(f["split"] for f in frames)),
            "claim_boundary": "One fixed simulated camera; no cross-camera or real-world transfer.",
            "annotation_rule": config["annotation"], "storage_plan": config["storage"],
            "coverage_scope": "all_targets means all split-permitted targets; union covers all five",
            "selection_scope": "compact accepted nominal subset; failures/corrections remain separate"}


def asset_provenance(workcell) -> list:
    root = workcell.model_path.parent
    hand = root / "menagerie" / "shadow_hand"
    return [
        {"asset": "DexVision procedural workcell", "source": "repository assets/mujoco/workcell_scene.xml",
         "license": "unspecified; owner must choose a license before release",
         "sha256": digest_file(workcell.model_path)},
        {"asset": "Shadow Hand E3M5", "source": "Google DeepMind MuJoCo Menagerie / Shadow Robot Company",
         "license": "Apache-2.0", "license_sha256": digest_file(hand / "LICENSE"),
         "files": {p.relative_to(root).as_posix(): digest_file(p)
                   for p in sorted(hand.rglob("*")) if p.suffix in {".xml", ".obj", ".stl", ".png"}}},
    ]


def create_contact_sheets(output: Path, sheets: list) -> None:
    """Write native-resolution labeled mask overlays, stratified by split/category."""
    import cv2

    colors = [(245, 90, 70), (80, 220, 100), (100, 150, 255), (240, 205, 60)]
    review = []
    for condition in ("nominal", "mild_illumination", "partial_occlusion", "bounded_distractors"):
        panels = []
        seen = set()
        for source, candidate in sheets:
            key = source["split"], source["primary_category"]
            if source["condition"] != condition or key in seen:
                continue
            seen.add(key)
            _, frame, rgb, masks = candidate
            panel = rgb.copy()
            for index, annotation in enumerate(frame["annotations"]):
                mask = masks == annotation["mask_value"]
                color = colors[index % len(colors)]
                panel[mask] = (panel[mask]*0.65 + np.asarray(color)*0.35).astype(np.uint8)
                if annotation["box_xyxy"]:
                    x0, y0, x1, y1 = annotation["box_xyxy"]
                    cv2.rectangle(panel, (x0, y0), (x1-1, y1-1), color, 1)
                    cv2.putText(panel, annotation["object_id"], (x0, max(12, y0-3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
            header = np.full((52, panel.shape[1], 3), 24, np.uint8)
            for row, label in enumerate((f'{condition} | {source["split"]} | {source["primary_category"]}',
                                         f'{source["episode_id"]} frame {frame["source_frame_index"]}')):
                cv2.putText(header, label, (8, 19+row*23), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (245, 245, 245), 1, cv2.LINE_AA)
            panels.append(np.vstack((header, panel)))
            review.append({"sheet": f"contact_sheet_{condition}.png", "rgb": frame["rgb"],
                           "source_episode_id": source["episode_id"],
                           "source_frame_index": frame["source_frame_index"],
                           "split": source["split"], "category": source["primary_category"]})
        # Three split rows, four category columns; retain full image resolution.
        sheet = np.vstack([np.hstack(panels[i:i+4]) for i in range(0, len(panels), 4)])
        _write_png(output / f"contact_sheet_{condition}.png", sheet, rgb=True)
    write_json(output / "manual_review.json", {"status": "pending", "samples": review,
               "pass_criteria": ["Masks and boxes align", "IDs match objects",
                                 "No held-out object or test scene in train",
                                 "All three families and every split reviewed"]})
