"""Separate frozen development/confirmation datasets for the 5.6 repair.

Versioned copy of the V2 collector: old source and consumed protocols stay immutable."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from dexvision.logging.visual_stream import (
    _write_png, configure_render_model, digest_file, load_visual_config, write_json,
)
from dexvision.perception.perception_baseline import load_model, load_recipe, model_digest, save_artifact_index


def load_protocol(path: str | Path) -> dict:
    path = Path(path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    checksum = path.with_suffix(".sha256").read_text().split()[0]
    if digest_file(path) != checksum or protocol["version"] not in {
            "level5/perception-development-v1", "level5/perception-confirmation-v3"}:
        raise ValueError("Confirmation protocol checksum/version mismatch.")
    for name, expected in protocol["pins"].items():
        if digest_file(Path(name)) != expected:
            raise ValueError(f"Frozen confirmation input/implementation changed: {name}")
    return protocol


def freeze_protocol(config: str | Path, output_dir: str | Path) -> dict:
    """Freeze method, reset cases and limits before any new render/inference."""
    config = Path(config)
    expected = config.with_suffix(".sha256").read_text().split()[0]
    if digest_file(config) != expected:
        raise ValueError("Confirmation config checksum mismatch.")
    spec = yaml.safe_load(config.read_text())
    if spec["version"] not in {"level5/perception-development-v1", "level5/perception-confirmation-v3"}:
        raise ValueError("Unsupported confirmation version.")
    tag = spec["version"].rsplit("-", 1)[-1]
    recipe = load_recipe(spec["parent_recipe"])
    model = load_model(recipe)
    train = json.loads(Path(spec["train_split"]).read_text())["episodes"]
    prior_seeds = set()
    # Seed provenance only. Do not open/decode consumed test images/annotations.
    for split in ("train", "validation", "test"):
        rows = json.loads((Path(spec["train_split"]).parent / f"{split}.json").read_text())["episodes"]
        for row in rows:
            for metadata in (Path(spec["source_root"]) / row["source_path"] / "metadata.json",):
                if metadata.is_file():
                    prior_seeds.add(json.loads(metadata.read_text())["random_seed"])
    pins = {config.as_posix(): digest_file(config)}
    paths = [spec["parent_recipe"], spec["train_split"], spec["visual_config"],
             spec["workcell_config"], spec["dataset_config"], spec["camera_calibration"],
             spec["prior_frame_index"], "configs/level1_teleop.yaml", *spec["implementation_files"],
             *spec["prior_frame_indices"], *recipe["input_digests"], *recipe["implementation_digests"]]
    pins.update({name: digest_file(Path(name)) for name in paths})
    if spec.get("parent_failed_protocol"):
        name = spec["parent_failed_protocol"]
        pins[name] = digest_file(Path(name))
    cases = []
    for index, cell in enumerate(spec["cells"]):
        row = next(r for r in train if r["cell_id"] == cell)
        metadata_path = Path(spec["source_root"]) / row["source_path"] / "metadata.json"
        if digest_file(metadata_path) != row["file_sha256"]["metadata.json"]:
            raise ValueError("Expert parameter metadata differs from immutable TRAIN source.")
        metadata = json.loads(metadata_path.read_text())
        # Only configured neutral finger parameters enter generation, never saved actions/states.
        neutral = dict(zip(metadata["finger_target_names"], metadata["initial_applied_action"][7:], strict=True))
        for split, allocation in spec["splits"].items():
            if split not in ("train", "validation", "test"):
                raise ValueError("Unknown split allocation.")
            for condition_index, condition in enumerate(spec["conditions"]):
                for repetition in range(allocation["resets_per_cell_condition"]):
                    seed = allocation["seed_start"] + index*1000 + condition_index*100 + repetition
                    if seed in prior_seeds or any(c["seed"] == seed for c in cases):
                        raise ValueError("Reset seed collides with historical/new source.")
                    cases.append({"episode_id": f"perception_{tag}_{split}_{seed}",
                                  "session_id": f"sim_{tag}_{split}_{seed}", "seed": seed,
                                  "split": split, "cell_id": cell, "skill": row["skill"],
                                  "entity_id": row["entity_id"], "condition": condition,
                                  "boundary_case": repetition % 2 == 1, "neutral_fingers": neutral,
                                  "finger_target_names": metadata["finger_target_names"],
                                  "parameter_source_sha256": digest_file(metadata_path)})
    for prior in spec["preserved_protocols"]:
        previous = json.loads(Path(prior).read_text())
        prior_seeds.update(c["seed"] for c in previous["cases"])
        pins[prior] = digest_file(Path(prior))
    if any(c["seed"] in prior_seeds for c in cases):
        raise ValueError("New reset overlaps a consumed/preserved protocol.")
    if spec["role"] == "development" and set(spec["splits"]) != {"train", "validation"}:
        raise ValueError("Development must contain train/validation only.")
    if spec["role"] == "confirmation" and set(spec["splits"]) != {"test"}:
        raise ValueError("Confirmation must contain only fresh test sessions.")
    protocol = {**spec, "pins": pins, "cases": cases, "model_sha256": model_digest(model),
                "parent_recipe_sha256": recipe["_recipe_sha256"], "config_sha256": digest_file(config),
                "case_count": len(cases), "version": spec["version"],
                "selection_uses_confirmation": False, "dataset_role": spec["role"], "fresh_pixels_not_yet_generated": True,
                "prior_seed_count": len(prior_seeds)}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "protocol.json"
    write_json(path, protocol)
    path.with_suffix(".sha256").write_text(f"{digest_file(path)}  protocol.json\n")
    for name in spec["implementation_files"]:
        destination = directory / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
    print(f"Frozen {len(cases)} fresh simulator sessions before rendering/evaluation", flush=True)
    return protocol


def _expert(task, case):
    from dexvision.retargeting.curl_retargeter import CurlRetargeter, load_curl_retargeter_config
    from dexvision.sim.level4_expert import (
        DeterministicButtonPressConfig, DeterministicButtonPressExpert,
        DeterministicGraspLiftConfig, DeterministicPickPlaceExpert, DeterministicPlaceConfig,
    )
    from dexvision.sim.scripted_expert_settings import resolve_scripted_expert_settings

    neutral = {name: case["neutral_fingers"][name] for name in case["finger_target_names"]}
    if case["skill"] == "press_button":
        config = DeterministicButtonPressConfig.from_mapping(resolve_scripted_expert_settings(task, "scripted_button"))
        return DeterministicButtonPressExpert(finger_targets=neutral, config=config), config.sim_steps_per_action
    if case["skill"] != "pick_place_sequence":
        raise ValueError("Frozen v1 collection supports pick/place and fixture sequences only.")
    retargeter = CurlRetargeter.from_mapping(load_curl_retargeter_config("configs/level1_teleop.yaml"))
    closed = dict(neutral)
    for finger in retargeter.config.fingers:
        for target in finger.targets:
            closed[target.name] = target.map_control(1.)
    grasp = DeterministicGraspLiftConfig.from_mapping(resolve_scripted_expert_settings(task, "scripted_grasp"))
    place = DeterministicPlaceConfig.from_mapping(resolve_scripted_expert_settings(task, "scripted_place"))
    return DeterministicPickPlaceExpert(open_finger_targets=neutral, closed_finger_targets=closed,
                                       grasp_config=grasp, place_config=place,
                                       validate_on_reset=False), place.sim_steps_per_action


def generate_source(protocol: dict, case: dict, directory: Path) -> dict:
    """Run a new deterministic expert with ordinary physics; retain failures.

    No saved trajectory replay, learned controller, copied-state candidate
    search, object pose writes after reset, or outcome-based resampling occurs.
    """
    from dexvision.logging.level4_collection import WorkcellPilotTask, sample_level4_procedural_variation
    from dexvision.perception.world_estimate import evaluation_world_estimate
    from dexvision.evaluation.pick_policy import physical_evidence_world

    collection = yaml.safe_load(Path(protocol["dataset_config"]).read_text())
    variation = sample_level4_procedural_variation(collection["level4_5b_procedural_expansion"],
                                                  skill_name=case["skill"], seed=case["seed"],
                                                  boundary_case=case["boundary_case"])
    directory.mkdir(parents=True, exist_ok=False)
    with WorkcellPilotTask(workcell_config=protocol["workcell_config"], dataset_config=protocol["dataset_config"],
                           skill_name=case["skill"], goal_condition_id=case["cell_id"], seed=case["seed"],
                           procedural_variation=variation) as task:
        expert, steps = _expert(task, case)
        expert.reset(task, task.initial_world_state)
        metadata = {"task_config": task.metadata_task_config(), "skill_name": case["skill"],
                    "typed_goal": task.goal, "case": case, "orientation_hold": False,
                    "controller": "deterministic_scripted_expert", "no_policy_input": True}
        qpos, qvel, times, actions, phases, truth_worlds = [], [], [], [], [], []
        state = task.current_state
        reason, done = "source_horizon", False
        for _ in range(protocol["maximum_source_actions"]):
            qpos.append(task.env.data.qpos.copy())
            qvel.append(task.env.data.qvel.copy())
            times.append(float(task.env.data.time))
            truth_worlds.append(asdict(evaluation_world_estimate(physical_evidence_world(task))))
            requested, phase, done, error = expert.step(state.world_state)
            phases.append(phase)
            if error:
                reason = error
                break
            if done:
                reason = "scripted_complete"
                break
            action = requested.as_array()
            if not np.isfinite(action).all():
                reason = "nonfinite_scripted_action"
                break
            actions.append(action)
            task.env.set_mocap_pose(str(task.workcell.config.scene["hand_base_target"]),
                                   position=requested.base_position, orientation_quat=requested.base_orientation_wxyz)
            task.env.set_joint_targets(requested.finger_targets)
            state = task.step(n_steps=steps)
        np.savez_compressed(directory / "states.npz", qpos=qpos, qvel=qvel, timestamps=times,
                            actions=np.asarray(actions).reshape(-1, 27), phases=phases)
        metadata.update(source_reason=reason, source_success=bool(state.success),
                        source_done=done, frame_count=len(times), control_steps=steps)
        write_json(directory / "metadata.json", metadata)
        write_json(directory / "evaluation_only_world_estimates.json", truth_worlds)
    return metadata


def collect_dataset(protocol_path: str | Path, output_dir: str | Path) -> dict:
    """Generate/render every frozen reset once; labels stay evaluation-only."""
    import mujoco
    from dexvision.perception.render_annotations import camera_calibration, render_frame

    protocol = load_protocol(protocol_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    config, workcell = load_visual_config(protocol["visual_config"])
    reference = json.loads(Path(protocol["camera_calibration"]).read_text())
    # Digest-only collision audit; consumed images and annotations are never decoded.
    prior_hashes = set(re.findall(r'"rgb_pixel_sha256"\s*:\s*"([0-9a-f]{64})"',
                                 Path(protocol["prior_frame_index"]).read_text()))
    for prior in protocol["prior_frame_indices"]:
        prior_hashes.update(re.findall(r'"rgb_pixel_sha256"\s*:\s*"([0-9a-f]{64})"', Path(prior).read_text()))
    frames, owners, attempts = [], {}, []
    for index, case in enumerate(protocol["cases"]):
        child = root / "sources" / case["episode_id"]
        metadata = generate_source(protocol, case, child)
        with np.load(child / "states.npz", allow_pickle=False) as states:
            model = mujoco.MjModel.from_xml_path(str(workcell.model_path))
            entities, hidden = configure_render_model(model, metadata, workcell, config, "validation")
            if case["condition"] == "mild_illumination":
                for light in (model.light_diffuse, model.light_ambient, model.light_specular):
                    light[:] *= config["conditions"][case["condition"]]["intensity_multiplier"]
            data = mujoco.MjData(model)
            owner = {"skill": case["skill"], "segments": [], "phase_intervals": [],
                     "entity_id": case["entity_id"], "episode_id": case["episode_id"]}
            owners[case["episode_id"]] = owner
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                for frame_index in range(0, len(states["timestamps"]), protocol["sampling_stride"]):
                    data.qpos[:] = states["qpos"][frame_index]
                    data.qvel[:] = states["qvel"][frame_index]
                    data.time = float(states["timestamps"][frame_index])
                    mujoco.mj_forward(model, data)
                    calibration = camera_calibration(model, data, config["camera"]["id"], 640, 480)
                    calibration["version"] = "level4/visual-camera-v1"
                    if calibration != reference:
                        raise ValueError("Confirmation camera changed from frozen calibration.")
                    rgb, mask, annotations, params = render_frame(renderer, model, data,
                        camera=config["camera"]["id"], entities=entities, condition=case["condition"],
                        parameters=config["conditions"][case["condition"]], goal_entity=case["entity_id"])
                    rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                    # Never remove/re-render a colliding image to improve independence or metrics.
                    relative = Path("images") / case["episode_id"] / f"{frame_index:06d}"
                    (root / relative).parent.mkdir(parents=True, exist_ok=True)
                    rgb_path, mask_path = root / (str(relative)+"_rgb.png"), root / (str(relative)+"_mask.png")
                    _write_png(rgb_path, rgb, rgb=True)
                    _write_png(mask_path, mask)
                    phase = str(states["phases"][frame_index])
                    owner["phase_intervals"].append({"start_frame": frame_index, "end_frame": frame_index+1, "phase": phase})
                    frames.append({"source_episode_id": case["episode_id"], "source_frame_index": frame_index,
                        "recording_session_id": case["session_id"], "timestamp": float(data.time),
                        "split": case["split"], "condition": case["condition"], "condition_parameters": params,
                        "rgb": rgb_path.relative_to(root).as_posix(), "mask": mask_path.relative_to(root).as_posix(),
                        "rgb_sha256": digest_file(rgb_path), "mask_sha256": digest_file(mask_path),
                        "rgb_pixel_sha256": rgb_hash, "annotations": annotations,
                        "historical_pixel_collision": rgb_hash in prior_hashes})
        attempts.append({**case, "source_reason": metadata["source_reason"],
                         "source_success": metadata["source_success"], "source_frames": metadata["frame_count"],
                         "hidden_visual_entities": hidden})
        print(f"Collected {index+1}/{len(protocol['cases'])}: {case['episode_id']} {metadata['source_reason']}", flush=True)
    (root / "frames.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False)+"\n" for r in frames))
    write_json(root / "calibration.json", reference)
    write_json(root / "owners.json", owners)
    manifest = {"version": protocol["version"], "protocol_sha256": digest_file(Path(protocol_path)),
                "frame_count": len(frames), "episode_count": len(attempts), "attempts": attempts,
                "historical_pixel_collisions": sum(r["historical_pixel_collision"] for r in frames),
                "no_filtered_attempts": len(attempts) == protocol["case_count"],
                "source_kind": "fresh_simulator_sessions_ordinary_physics_scripted_experts",
                "disclosure": protocol["disclosure"], "training_eligible": protocol["role"] == "development",
                "splits": protocol["splits"], "dataset_role": protocol["role"]}
    write_json(root / "manifest.json", manifest)
    seen_pixels = {}
    cross_split = 0
    for row in frames:
        old = seen_pixels.setdefault(row["rgb_pixel_sha256"], row["split"])
        cross_split += old != row["split"]
    manifest["cross_split_pixel_collisions"] = cross_split
    write_json(root / "manifest.json", manifest)
    save_artifact_index(root)
    return manifest


def verify_dataset(root: Path, protocol_path: Path) -> dict:
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or digest_file(path) != expected:
            raise ValueError("Confirmation dataset checksum/path mismatch.")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest_file(protocol_path):
        raise ValueError("Confirmation dataset belongs to another protocol.")
    if (manifest["historical_pixel_collisions"] or manifest["cross_split_pixel_collisions"] or
            not manifest["no_filtered_attempts"]):
        raise ValueError("Confirmation independence/completeness audit failed.")
    return manifest


def package_dataset(protocol_path: str | Path, dataset_dir: str | Path, release: str | Path) -> dict:
    """Publish a new local immutable archive; never overwrite any release."""
    protocol_path, root, archive = Path(protocol_path), Path(dataset_dir), Path(release)
    load_protocol(protocol_path)
    manifest = verify_dataset(root, protocol_path)
    sidecar = archive.with_name(archive.name+".sha256")
    release_manifest = archive.with_name(archive.name.removesuffix(".tar.gz")+"_manifest.json")
    if any(p.exists() for p in (archive, sidecar, release_manifest)):
        raise ValueError("Release exists; use a new version.")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as tar:
        for file in sorted(p for p in root.rglob("*") if p.is_file()):
            tar.add(file, arcname=(Path("datasets") / archive.name.removesuffix(".tar.gz") / file.relative_to(root)).as_posix())
    digest = digest_file(archive)
    sidecar.write_text(f"{digest}  {archive.as_posix()}\n")
    release_info = {**manifest, "archive": archive.as_posix(), "archive_sha256": digest,
                    "archive_bytes": archive.stat().st_size, "protocol": protocol_path.as_posix(),
                    "payload_files": {p.relative_to(root).as_posix(): digest_file(p)
                                      for p in sorted(root.rglob("*")) if p.is_file()},
                    "license": "CC-BY-4.0 generated dataset; source assets retain original licenses"}
    write_json(release_manifest, release_info)
    return release_info
