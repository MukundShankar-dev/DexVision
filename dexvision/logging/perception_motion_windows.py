"""Frozen phase-window render amendment of allowed TRAIN/validation sources.

No new physics sessions are claimed. Every declared motion-phase control frame is a
query, with three causal context frames at the original five-step cadence.
Selection never uses visibility, source outcomes or consumed test data.
"""

from __future__ import annotations
import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import yaml
from dexvision.logging.perception_development import load_protocol, verify_dataset
from dexvision.logging.visual_stream import (
    digest_file,
    write_json,
    load_visual_config,
    configure_render_model,
    _write_png,
)
from dexvision.perception.perception_baseline import save_artifact_index


def freeze_phase_windows(config, output_dir):
    path = Path(config)
    if digest_file(path) != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("Phase plan checksum mismatch.")
    spec = yaml.safe_load(path.read_text())
    cases = []
    coverage = []
    pins = {str(path): digest_file(path)}
    for protocol_name, root_name in spec["parents"]:
        parent = load_protocol(protocol_name)
        if parent["dataset_role"] != "development":
            raise ValueError("Consumed/test sources cannot enter phase windows.")
        root = Path(root_name)
        manifest = verify_dataset(root, Path(protocol_name))
        if not manifest["training_eligible"]:
            raise ValueError("Only TRAIN/validation source parents permitted.")
        for source in parent["cases"]:
            if source["entity_id"] != "block_small":
                continue
            episode = source["episode_id"]
            child = root / "sources" / episode
            with np.load(child / "states.npz", allow_pickle=False) as states:
                queries = np.flatnonzero(np.isin(states["phases"], spec["phases"])).tolist()
            coverage.append(
                {
                    "parent_root": root_name,
                    "episode": episode,
                    "session": source["session_id"],
                    "split": source["split"],
                    "query_count": len(queries),
                    "source_outcome_not_filtered": True,
                }
            )
            for query in queries:
                window = f"phase_v7_{episode}_{query:06d}"
                context = sorted(
                    {
                        max(0, query - spec["context_stride"] * k)
                        for k in range(spec["context_frames"] + 1)
                    }
                )
                cases.append(
                    {
                        **source,
                        "episode_id": window,
                        "original_episode": episode,
                        "parent_root": root_name,
                        "query_frame": query,
                        "context_frames": context,
                    }
                )
            for name in (
                child / "states.npz",
                child / "metadata.json",
                child / "evaluation_only_world_estimates.json",
            ):
                pins[str(name)] = digest_file(name)
        for name in (Path(protocol_name), root / "manifest.json", root / "SHA256SUMS"):
            pins[str(name)] = digest_file(name)
    for name in (
        spec["implementation_files"]
        + spec["consumed_indices"]
        + [
            spec["visual_config"],
            spec["camera_calibration"],
        ]
    ):
        pins[name] = digest_file(Path(name))
    protocol = {
        **spec,
        "version": "level5/perception-development-v1",
        "dataset_role": "development",
        "role": "development",
        "pins": pins,
        "cases": cases,
        "case_count": len(cases),
        "parent_coverage": coverage,
        "physical_source_sessions": len(coverage),
        "no_new_sessions": True,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    p = output / "protocol.json"
    write_json(p, protocol)
    p.with_suffix(".sha256").write_text(f"{digest_file(p)}  protocol.json\n")
    print(
        "Frozen phase queries",
        len(cases),
        "from",
        len(coverage),
        "existing source sessions",
        flush=True,
    )
    return protocol


def collect_phase_windows(protocol_path, output_dir):
    import mujoco
    from dexvision.perception.render_annotations import camera_calibration, render_frame

    protocol = load_protocol(protocol_path)
    if protocol["derivation"] != "phase-query-windows-v1":
        raise ValueError("Exact phase derivation required.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    config, workcell = load_visual_config(protocol["visual_config"])
    reference = json.loads(Path(protocol["camera_calibration"]).read_text())
    grouped = defaultdict(list)
    for case in protocol["cases"]:
        grouped[case["parent_root"], case["original_episode"]].append(case)
    prior = set()
    for name in protocol["consumed_indices"]:
        prior.update(
            re.findall(
                r'"rgb_pixel_sha256"\s*:\s*"([0-9a-f]{64})"', Path(name).read_text()
            )
        )
    historical = 0
    frames = []
    owners = {}
    seen = {}
    cross = 0
    for index, ((parent, episode), cases) in enumerate(grouped.items()):
        child = Path(parent) / "sources" / episode
        metadata = json.loads((child / "metadata.json").read_text())
        first = cases[0]
        model = mujoco.MjModel.from_xml_path(str(workcell.model_path))
        model.vis.quality.offsamples = 0
        entities, _ = configure_render_model(
            model, metadata, workcell, config, "validation"
        )
        if first["condition"] == "mild_illumination":
            for light in (
                model.light_diffuse,
                model.light_ambient,
                model.light_specular,
            ):
                light[:] *= config["conditions"][first["condition"]][
                    "intensity_multiplier"
                ]
        data = mujoco.MjData(model)
        rendered = {}
        with (
            np.load(child / "states.npz", allow_pickle=False) as states,
            mujoco.Renderer(model, height=480, width=640) as renderer,
        ):
            for f in sorted({f for c in cases for f in c["context_frames"]}):
                data.qpos[:] = states["qpos"][f]
                data.qvel[:] = states["qvel"][f]
                data.time = float(states["timestamps"][f])
                mujoco.mj_forward(model, data)
                cal = camera_calibration(model, data, config["camera"]["id"], 640, 480)
                cal["version"] = "level4/visual-camera-v1"
                if cal != reference:
                    raise ValueError("Phase camera differs.")
                rgb, mask, annotations, params = render_frame(
                    renderer,
                    model,
                    data,
                    camera=config["camera"]["id"],
                    entities=entities,
                    condition=first["condition"],
                    parameters=config["conditions"][first["condition"]],
                    goal_entity=first["entity_id"],
                )
                relative = Path("images") / episode / f"{f:06d}"
                rgb_path = root / (str(relative) + "_rgb.png")
                mask_path = root / (str(relative) + "_mask.png")
                rgb_path.parent.mkdir(parents=True, exist_ok=True)
                _write_png(rgb_path, rgb, rgb=True)
                _write_png(mask_path, mask)
                pixel = hashlib.sha256(rgb.tobytes()).hexdigest()
                cross += seen.setdefault(pixel, first["split"]) != first["split"]
                historical += pixel in prior
                rendered[f] = {
                    "source_frame_index": f,
                    "timestamp": float(data.time),
                    "split": first["split"],
                    "condition": first["condition"],
                    "condition_parameters": params,
                    "rgb": rgb_path.relative_to(root).as_posix(),
                    "mask": mask_path.relative_to(root).as_posix(),
                    "rgb_sha256": digest_file(rgb_path),
                    "mask_sha256": digest_file(mask_path),
                    "rgb_pixel_sha256": pixel,
                    "annotations": annotations,
                }
            for case in cases:
                window = case["episode_id"]
                owners[window] = {
                    "entity_id": case["entity_id"],
                    "episode_id": window,
                    "segments": [],
                    "evaluation_only_parent_episode": episode,
                    "evaluation_only_source_root": parent,
                    "phase_intervals": [
                        {
                            "phase": str(states["phases"][f]),
                            "start_frame": f,
                            "end_frame": f + 1,
                        }
                        for f in case["context_frames"]
                    ],
                }
                for f in case["context_frames"]:
                    frames.append(
                        {
                            **rendered[f],
                            "source_episode_id": window,
                            "recording_session_id": case["session_id"],
                            "phase_window_query": f == case["query_frame"],
                        }
                    )
        if (index + 1) % 20 == 0:
            print("Rendered physical sources", index + 1, "/", len(grouped), flush=True)
    (root / "calibration.json").write_bytes(
        Path(protocol["camera_calibration"]).read_bytes()
    )
    write_json(root / "owners.json", owners)
    with (root / "frames.jsonl").open("x") as f:
        for row in frames:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "version": protocol["version"],
        "protocol_sha256": digest_file(Path(protocol_path)),
        "dataset_role": "development",
        "training_eligible": True,
        "splits": {"train": {}, "validation": {}},
        "historical_pixel_collisions": historical,
        "cross_split_pixel_collisions": cross,
        "no_filtered_attempts": True,
        "frame_count": len(frames),
        "query_count": len(protocol["cases"]),
        "episode_count": len(protocol["cases"]),
        "physical_source_sessions": protocol["physical_source_sessions"],
        "source_kind": "derived_phase_windows_from_frozen_development_sources",
        "no_new_sessions": True,
        "intentional_parent_pixel_reuse": True,
        "parents": protocol["parents"],
        "parent_coverage": protocol["parent_coverage"],
        "scope": "all declared motion-phase query frames, three causal prior frames, no visibility/outcome filter; original whole-session split retained; raw parents unchanged",
    }
    write_json(root / "manifest.json", manifest)
    save_artifact_index(root)
    return manifest
