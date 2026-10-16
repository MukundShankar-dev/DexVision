"""Pixel-center RGB/mask alignment without changing physics or eligible labels."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml

from dexvision.logging.perception_development import load_protocol, verify_dataset, generate_source
from dexvision.logging.visual_stream import (
    _write_png, configure_render_model, digest_file, load_visual_config, write_json,
)
from dexvision.perception.perception_baseline import save_artifact_index


def freeze_development(config: str | Path, output_dir: str | Path) -> dict:
    path = Path(config)
    if digest_file(path) != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("Raster amendment checksum mismatch.")
    spec = yaml.safe_load(path.read_text())
    if spec["role"] != "development" or spec["rasterization"] != {"offscreen_samples": 0}:
        raise ValueError("Only explicit pixel-center development amendment is supported.")
    parent_path = Path(spec["source_protocol"])
    parent = load_protocol(parent_path)
    root = Path(spec["source_dataset"])
    manifest = verify_dataset(root, parent_path)
    if parent["dataset_role"] != "development" or set(manifest["splits"]) != {"train", "validation"}:
        raise ValueError("Only unconsumed TRAIN/validation sources can be re-rendered.")
    pins = {**parent["pins"], str(path): digest_file(path), str(parent_path): digest_file(parent_path)}
    for file in [root/"SHA256SUMS", root/"manifest.json", *map(Path, spec["implementation_files"])]:
        pins[str(file)] = digest_file(file)
    protocol = {**parent, **spec, "pins": pins, "source_protocol_sha256": digest_file(parent_path),
                "cases": parent["cases"], "case_count": len(parent["cases"]),
                "dataset_role": "development", "fresh_pixels_not_yet_generated": True,
                "new_independent_source_sessions": False, "source_attempts_reused_without_resampling": True}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    destination = output/"protocol.json"
    write_json(destination, protocol)
    destination.with_suffix(".sha256").write_text(f"{digest_file(destination)}  protocol.json\n")
    for file in spec["implementation_files"]:
        target = output/"source_snapshot"/file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(file).read_bytes())
    return protocol


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
        if protocol.get("source_dataset"):
            # Re-render verified TRAIN/validation states only; never replay controls.
            import shutil
            source = Path(protocol["source_dataset"]) / "sources" / case["episode_id"]
            shutil.copytree(source, child)
            metadata = json.loads((child / "metadata.json").read_text())
        else:
            metadata = generate_source(protocol, case, child)
        with np.load(child / "states.npz", allow_pickle=False) as states:
            model = mujoco.MjModel.from_xml_path(str(workcell.model_path))
            model.vis.quality.offsamples = protocol["rasterization"]["offscreen_samples"]
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
                "source_kind": ("verified_development_source_rerender" if protocol.get("source_dataset") else
                                "fresh_simulator_sessions_ordinary_physics_scripted_experts"),
                "rasterization": protocol["rasterization"],
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

