"""Derived TRAIN/validation union; raw frozen parents remain immutable."""

from __future__ import annotations
import json
from pathlib import Path
import shutil
from dexvision.logging.perception_development import load_protocol, verify_dataset
from dexvision.logging.visual_stream import digest_file, write_json
from dexvision.perception.perception_baseline import save_artifact_index


def union_development(parents: list[tuple[str, str]], output_dir: str | Path) -> dict:
    rows = []
    owners = {}
    cases = []
    pins = {}
    seen_ids = set()
    seen_sessions = set()
    seen_seeds = set()
    pixels = {}
    calibration = None
    for protocol_name, root_name in parents:
        protocol_path, root = Path(protocol_name), Path(root_name)
        protocol = load_protocol(protocol_path)
        if protocol["dataset_role"] != "development":
            raise ValueError("Only frozen TRAIN/validation parents may enter a union.")
        manifest = verify_dataset(root, protocol_path)
        if not manifest["training_eligible"] or set(manifest["splits"]) != {
            "train",
            "validation",
        }:
            raise ValueError("Only frozen TRAIN/validation parents may enter a union.")
        current_cal = (root / "calibration.json").read_bytes()
        if calibration is not None and current_cal != calibration:
            raise ValueError("Union camera calibration differs.")
        calibration = current_cal
        current_owners = json.loads((root / "owners.json").read_text())
        for c in protocol["cases"]:
            if (
                c["episode_id"] in seen_ids
                or c["session_id"] in seen_sessions
                or c["seed"] in seen_seeds
            ):
                raise ValueError("Parent source/session/seed overlap.")
            seen_ids.add(c["episode_id"])
            seen_sessions.add(c["session_id"])
            seen_seeds.add(c["seed"])
            cases.append(c)
            owners[c["episode_id"]] = {
                **current_owners[c["episode_id"]],
                "evaluation_only_source_root": root.as_posix(),
            }
        case_map = {c["episode_id"]: c for c in protocol["cases"]}
        seen_parent = set()
        for line in (root / "frames.jsonl").open():
            row = json.loads(line)
            case = case_map.get(row["source_episode_id"])
            if case is None or (row["recording_session_id"], row["split"]) != (
                case["session_id"],
                case["split"],
            ):
                raise ValueError("Parent row ownership differs from frozen cases.")
            seen_parent.add(row["source_episode_id"])
            for field in ("rgb", "mask"):
                relative = Path(row[field])
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not (root / relative).resolve().is_relative_to(root.resolve())
                ):
                    raise ValueError("Unsafe parent image path.")
            if row["split"] not in ("train", "validation"):
                raise ValueError("Test row prohibited in union.")
            if pixels.setdefault(row["rgb_pixel_sha256"], row["split"]) != row["split"]:
                raise ValueError("Cross-split pixel collision.")
            rows.append((row, root))
        if seen_parent != set(case_map):
            raise ValueError("Missing parent source attempts.")
        for name in [
            protocol_path,
            root / "manifest.json",
            root / "SHA256SUMS",
            root / "frames.jsonl",
            root / "owners.json",
        ]:
            pins[name.as_posix()] = digest_file(name)
    if not rows or not owners:
        raise ValueError("Empty training union.")
    pins["dexvision/logging/perception_union.py"] = digest_file(Path(__file__))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = output / "dataset"
    data.mkdir()
    pdir = output / "protocol"
    pdir.mkdir()
    protocol = {
        "version": "level5/perception-development-v1",
        "checkpoint": "5.6",
        "dataset_role": "development",
        "role": "development",
        "case_count": len(cases),
        "cases": cases,
        "pins": pins,
        "parents": parents,
        "source_kind": "derived_union_of_frozen_development_releases",
        "no_new_sessions": True,
        "splits": {
            s: {"episode_count": sum(c["split"] == s for c in cases)}
            for s in ("train", "validation")
        },
    }
    path = pdir / "protocol.json"
    write_json(path, protocol)
    path.with_suffix(".sha256").write_text(f"{digest_file(path)}  protocol.json\n")
    for row, root in rows:
        for field in ("rgb", "mask"):
            source = root / row[field]
            destination = data / row[field]
            if not source.resolve().is_relative_to(
                root.resolve()
            ) or not destination.resolve().is_relative_to(data.resolve()):
                raise ValueError("Unsafe union path.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    (data / "calibration.json").write_bytes(calibration)
    write_json(data / "owners.json", owners)
    with (data / "frames.jsonl").open("x") as f:
        for row, _ in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "version": protocol["version"],
        "protocol_sha256": digest_file(path),
        "dataset_role": "development",
        "training_eligible": True,
        "splits": protocol["splits"],
        "episode_count": len(cases),
        "frame_count": len(rows),
        "historical_pixel_collisions": 0,
        "cross_split_pixel_collisions": 0,
        "no_filtered_attempts": True,
        "source_kind": protocol["source_kind"],
        "parents": parents,
        "disclosure": "Exact copied RGB/masks/labels; all parent attempts retain original split. No new physics and no consumed test parent. Raw sources remain in immutable parent releases.",
    }
    write_json(data / "manifest.json", manifest)
    save_artifact_index(data)
    verify_dataset(data, path)
    write_json(
        output / "receipt.json",
        {
            "complete": True,
            "protocol_sha256": digest_file(path),
            "manifest_sha256": digest_file(data / "manifest.json"),
            "frame_count": len(rows),
            "episode_count": len(cases),
            "sources_reused_not_new": True,
        },
    )
    return manifest
