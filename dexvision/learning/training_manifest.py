"""Immutable inputs, stream ownership and portable Level 5 training manifests."""
from __future__ import annotations

import io
import json
import platform
import subprocess
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from dexvision.logging.dataset_release import (
    content_digest, digest_file, publication_blockers, read_json, safe_path,
)

SPLITS = ("train", "validation", "test")
STREAMS = ("expert_success", "ordinary_failure", "corrective_intervention", "visual", "legacy")


@dataclass(frozen=True)
class TrainingInputs:
    root: Path
    skill: dict
    plan: dict
    release: dict
    schemas: dict
    splits: dict
    requirements: dict
    digests: dict
    config_snapshot: dict


def load_training_inputs(config: str | Path, *, root: Path | None = None) -> TrainingInputs:
    """Verify the 5.0 freeze before selecting data; never infer membership from disk."""
    root = (root or Path(__file__).resolve().parents[2]).resolve()
    config = Path(config)
    config = config if config.is_absolute() else root / config
    skill = yaml.safe_load(config.read_text(encoding="utf-8"))
    plan = yaml.safe_load(safe_path(root, skill["learning_config"]).read_text())
    pins = plan["inputs"]
    digests = {}
    for line in safe_path(root, "configs/level5/SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest_file(safe_path(root, name)) != expected:
            raise ValueError(f"Frozen protocol checksum mismatch: {name}")
        digests[name] = expected
    relative_config = config.resolve().relative_to(root).as_posix()
    if relative_config not in digests or skill["skill"] not in plan["required_skills"]:
        raise ValueError("Skill config is not part of the frozen protocol")
    for key, pin in pins.items():
        if key in {"archive", "dataset_digest"}:
            continue
        entries = pin.values() if key == "splits" else (pin,)
        for entry in entries:
            if digest_file(safe_path(root, entry["path"])) != entry["sha256"]:
                raise ValueError(f"Frozen input checksum mismatch: {entry['path']}")
            digests[entry["path"]] = entry["sha256"]
    release = read_json(safe_path(root, pins["release"]["path"]))
    release_dir = safe_path(root, pins["release"]["path"]).parent
    checksums = dict(line.split("  ", 1)[::-1]
                     for line in (release_dir / "SHA256SUMS").read_text().splitlines())
    for name, expected in checksums.items():
        if digest_file(safe_path(release_dir, name)) != expected:
            raise ValueError(f"Release metadata checksum mismatch: {name}")
    if publication_blockers(release_dir, release, checksums):
        raise ValueError("Level 4 release is not publication ready")
    if (release["dataset_digest"] != pins["dataset_digest"]
            or release["archive"]["sha256"] != pins["archive"]["sha256"]):
        raise ValueError("Release disagrees with frozen learning inputs")
    for name, entry in release["files"].items():
        if name.startswith("assets/mujoco/"):
            if digest_file(safe_path(root, name)) != entry["sha256"]:
                raise ValueError(f"Simulator asset differs from the immutable release: {name}")
            digests[name] = entry["sha256"]
    splits = {s: read_json(safe_path(root, pins["splits"][s]["path"])) for s in SPLITS}
    schemas = read_json(safe_path(root, pins["schemas"]["path"]))
    requirements = yaml.safe_load(safe_path(root, pins["dataset_config"]["path"]).read_text())
    validate_split_ownership(splits, requirements, schemas)
    digests.update(dataset_sha256=pins["dataset_digest"], archive_sha256=pins["archive"]["sha256"],
                   schema_sha256=pins["schemas"]["sha256"],
                   split_sha256={s: pins["splits"][s]["sha256"] for s in SPLITS})
    snapshot = {"skill": skill, "learning": plan,
                "evaluation": yaml.safe_load(safe_path(root, skill["evaluation_config"]).read_text())}
    digests["config_sha256"] = content_digest(snapshot)
    return TrainingInputs(root, skill, plan, release, schemas, splits, requirements,
                          digests, snapshot)


def validate_split_ownership(splits: dict, requirements: dict, schemas: dict) -> None:
    """Reject episode/session/cell/lineage, reserved-id and image ownership leakage."""
    rows, sessions, cells, images, sources, actions = {}, {}, {}, {}, {}, {}
    policy = requirements["split_policy"]
    frozen_cells = {c["id"]: c for c in requirements["coverage_cells"]}
    for split in SPLITS:
        doc = splits[split]
        if (doc["split"] != split or doc["status"] != "frozen" or not doc["dataset_audit_passed"]
                or doc["manifest_digest"] != content_digest(
                    {k: v for k, v in doc.items() if k != "manifest_digest"})):
            raise ValueError(f"Invalid frozen split manifest: {split}")
        for row in doc["episodes"]:
            eid = row["episode_id"]
            if eid in rows or row["split"] != split or not row["audit_passed"]:
                raise ValueError(f"Duplicate, unqualified or misassigned episode: {eid}")
            rows[eid] = row
            for name, owners in (("recording_session_id", sessions), ("cell_id", cells)):
                value = row[name]
                if not value or owners.setdefault(value, split) != split:
                    raise ValueError(f"Cross-split {name}: {eid}")
            if frozen_cells[row["cell_id"]]["split_owner"] != split:
                raise ValueError(f"Frozen condition ownership mismatch: {eid}")
            if split != "test" and (
                row["entity_id"] in policy["held_out_object_instances"]
                or row["entity_id"] in policy["held_out_goal_regions"]
                or row["target_id"] in policy["held_out_goal_regions"]
            ):
                raise ValueError(f"Held-out object or goal leakage: {eid}")
            layout = {k: row[k] for k in ("action_schema", "observation_schema", "schema_versions")}
            digest = content_digest(layout)
            if (schemas["episode_layout_digest"].get(eid) != digest
                    or schemas["layouts_by_digest"].get(digest) != layout):
                raise ValueError(f"Executable schema mismatch: {eid}")
            if row["stream"] == "expert_success":
                if row["source"] != "scripted":
                    raise ValueError("Only scripted expert successes are baseline targets")
                for key in ("action_sha256", "initial_state_digest"):
                    value = row.get(key)
                    if value and actions.setdefault((key, value), split) != split:
                        raise ValueError(f"Duplicate {key} across splits")
            bound_entity = next((row["typed_goal"][k] for k in ("entity_id", "object_id", "button_id")
                                 if k in row["typed_goal"]), None)
            bound_target = row["typed_goal"].get("target_id", row["typed_goal"].get("target_zone"))
            if bound_entity != row["entity_id"] or bound_target != row["target_id"]:
                raise ValueError(f"Typed goal/entity binding mismatch: {eid}")
        if split != "train" and doc["normalization_inputs"]:
            raise ValueError("Held-out normalization inputs are forbidden")
        for item in doc["images"]:
            if item["split"] != split:
                raise ValueError("Image split mismatch")
            source = item["source_episode_id"]
            if sources.setdefault(source, split) != split:
                raise ValueError("Image source crosses splits")
            for key in ("rgb", "mask", "rgb_sha256"):
                if images.setdefault((key, item[key]), split) != split:
                    raise ValueError("Duplicate image across splits")
    for eid, row in rows.items():
        source = row.get("source_episode_id")
        if source and (source not in rows or rows[source]["split"] != row["split"]):
            raise ValueError(f"Missing or cross-split source lineage: {eid}")
    # Superseded visual sources keep their original whole-session ownership.
    # Their active replacement must not silently turn them into policy targets.
    for eid, split in sources.items():
        if eid in rows and rows[eid]["split"] != split:
            raise ValueError("Image/episode ownership mismatch")
    seen = set()
    for entry in splits["train"]["normalization_inputs"]:
        eid = entry["episode_id"]
        row = rows[eid]
        if (eid in seen or row["split"] != "train" or row["stream"] != "expert_success"
                or entry["episode_digest"] != row["episode_digest"]
                or entry["frame_interval"] != row["training_target_interval"]):
            raise ValueError(f"Invalid normalization owner: {eid}")
        seen.add(eid)


def skill_interval(skill: dict, row: dict) -> tuple[int, int] | None:
    """Use exactly the 5.0 source skill and inclusive-exclusive interval rule."""
    if row["skill"] != skill["source_skill"] or row["stream"] != "expert_success":
        return None
    segments = [s for s in row["segments"] if s["skill_name"] == skill["skill"]]
    if len(segments) != 1:
        raise ValueError(f"Ambiguous or missing skill segment: {row['episode_id']}")
    start, end = segments[0]["start_frame"], segments[0]["end_frame"]
    phase = skill["training"]["first_contact_phase"]
    if phase:
        start = next(p["start_frame"] for p in row["phase_intervals"] if p["phase"] == phase)
    lower, upper = row["training_target_interval"]
    start, end = max(start, lower), min(end, upper)
    if not 0 <= start < end <= row["frames"]:
        raise ValueError("Invalid eligible skill interval")
    return start, end


def select_streams(inputs: TrainingInputs, streams: tuple[str, ...] = ("expert_success",)) -> dict:
    """Explicitly inventory independent streams; diagnostics never become BC targets."""
    if not streams or len(set(streams)) != len(streams) or set(streams) - set(STREAMS):
        raise ValueError(f"Choose distinct streams from {STREAMS}")
    result = {name: {s: [] for s in SPLITS} for name in streams if name != "legacy"}
    for split, doc in inputs.splits.items():
        for row in doc["episodes"]:
            if row["stream"] in result:
                if row["stream"] != "expert_success" or skill_interval(inputs.skill, row):
                    result[row["stream"]][split].append(row)
        if "visual" in result:
            result["visual"][split] = doc["images"]
    if "legacy" in streams:
        pin = inputs.release["legacy_release"]
        path = safe_path(inputs.root, pin["manifest"])
        if digest_file(path) != pin["sha256"]:
            raise ValueError("Legacy release manifest checksum mismatch")
        legacy = read_json(path)
        result["legacy"] = {"manifest": legacy, "manifest_sha256": pin["sha256"],
                            "comparison_only": True, "recording_session_ids_available": False,
                            "baseline_target_count": 0}
    return result


class ReleaseSource:
    """Read only checksum-bound release bytes, from an archive or restored root."""

    def __init__(self, inputs: TrainingInputs, release_root: Path | None = None):
        self.inputs = inputs
        self.root = Path(release_root).resolve() if release_root is not None else None

    def read_many(self, names: list[str]) -> dict[str, bytes]:
        wanted = set(names)
        if len(wanted) != len(names) or wanted - self.inputs.release["files"].keys():
            raise ValueError("Duplicate or unlisted release file requested")
        result = {}
        if self.root is not None:
            for name in sorted(wanted):
                result[name] = safe_path(self.root, name).read_bytes()
        else:
            pin = self.inputs.plan["inputs"]["archive"]
            archive = safe_path(self.inputs.root, pin["path"])
            if not archive.is_file() or digest_file(archive) != pin["sha256"]:
                raise ValueError("Release archive missing or changed; retrieve it with git lfs pull")
            seen = set()
            with tarfile.open(archive, "r|gz") as tar:
                for member in tar:
                    safe_path(archive.parent, member.name)
                    if not member.isfile() or member.issparse() or member.name in seen:
                        raise ValueError("Unsafe or duplicate archive member")
                    seen.add(member.name)
                    if member.name in wanted:
                        with tar.extractfile(member) as stream:
                            result[member.name] = stream.read()
        if result.keys() != wanted:
            raise ValueError("Missing requested release payload")
        import hashlib

        for name, data in result.items():
            expected = self.inputs.release["files"][name]
            if len(data) != expected["size_bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ValueError(f"Release payload checksum mismatch: {name}")
        return result

    def read_episode_records(self, rows: list[dict]) -> dict[str, dict]:
        """Expose original arrays and metadata for explicitly selected streams."""
        names = [f"data/demos/level4/{r['source_path']}/{name}"
                 for r in rows for name in r["file_sha256"]]
        blobs = self.read_many(names)
        records = {}
        for row in rows:
            prefix = f"data/demos/level4/{row['source_path']}/"
            record = {}
            for name in row["file_sha256"]:
                data = blobs[prefix + name]
                if name.endswith(".npy"):
                    record[name[:-4]] = np.load(io.BytesIO(data), allow_pickle=False)
                elif name == "metadata.json":
                    record["metadata"] = json.loads(data)
            records[row["episode_id"]] = record
        return records

    def read_visual(self, entries: list[dict]) -> list[dict]:
        """Return frozen RGB/mask bytes paired with original frame ownership."""
        names = [f"data/visual/level4/{r[k]}" for r in entries for k in ("rgb", "mask")]
        blobs = self.read_many(names)
        return [{**row, "rgb_bytes": blobs[f"data/visual/level4/{row['rgb']}"],
                 "mask_bytes": blobs[f"data/visual/level4/{row['mask']}"]} for row in entries]

    def read_legacy_records(self, *, episode_paths: list[str]) -> dict[str, dict]:
        """Read explicitly selected Level 2 archive episodes for separate comparisons.

        Paths are archive-relative directories under data/demos/raw. No Level 4
        split, session, normalization, or supervision assignment is invented.
        """
        legacy = select_streams(self.inputs, ("legacy",))["legacy"]["manifest"]
        archive = safe_path(self.inputs.root, legacy["archive"])
        if digest_file(archive) != legacy["archive_sha256"]:
            raise ValueError("Legacy archive checksum mismatch; run git lfs pull")
        if not episode_paths or len(set(episode_paths)) != len(episode_paths):
            raise ValueError("Choose unique legacy episode paths")
        records = {}
        for path in episode_paths:
            safe_path(self.inputs.root, path)
            if not path.startswith("data/demos/raw/"):
                raise ValueError("Legacy comparison must explicitly select raw episode paths")
            records[path] = {}
        seen = set()
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if member.isdir():
                    continue
                safe_path(archive.parent, member.name)
                if not member.isfile() or member.issparse() or member.name in seen:
                    raise ValueError("Unsafe legacy archive member")
                seen.add(member.name)
                parent, _, name = member.name.rpartition("/")
                if parent in records and (name == "metadata.json" or name.endswith(".npy")):
                    with tar.extractfile(member) as stream:
                        data = stream.read()
                    records[parent]["metadata" if name == "metadata.json" else name[:-4]] = (
                        json.loads(data) if name == "metadata.json"
                        else np.load(io.BytesIO(data), allow_pickle=False))
        if any("metadata" not in r or "actions" not in r for r in records.values()):
            raise ValueError("Missing legacy episode metadata or actions")
        return records


def environment_manifest(device: str) -> dict:
    import os

    import mujoco
    import torch

    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "numpy": np.__version__, "torch": str(torch.__version__), "mujoco": mujoco.__version__,
            "device": device, "cuda": torch.version.cuda,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def build_training_manifest(inputs: TrainingInputs, *, seed: int, device: str,
                            model: str, streams: tuple[str, ...] = ("expert_success",)) -> dict:
    selected = select_streams(inputs, streams)
    rows = selected.get("expert_success", {s: [] for s in SPLITS})
    assignments = {s: [{"episode_id": r["episode_id"], "session_id": r["recording_session_id"],
                        "cell_id": r["cell_id"], "episode_digest": r["episode_digest"],
                        "frame_interval": list(skill_interval(inputs.skill, r))}
                       for r in rows[s]] for s in SPLITS}
    command = subprocess.run(["git", "rev-parse", "HEAD"], cwd=inputs.root,
                             text=True, capture_output=True, check=False)
    implementation = {p.relative_to(inputs.root).as_posix(): digest_file(p)
                      for p in sorted((inputs.root / "dexvision").rglob("*.py"))}
    result = {"version": "level5/training-manifest-v1", "skill": inputs.skill["skill"],
              "seed": seed, "device": device, "model": model, "digests": inputs.digests,
              "assignments": assignments, "streams": list(streams),
              "stream_counts": {name: {s: len(values[s]) for s in SPLITS}
                                for name, values in selected.items() if name != "legacy"},
              "legacy": selected.get("legacy"),
              "sampling": inputs.plan["training"]["sampling"],
              "validation_matrix_sha256": content_digest(inputs.skill["evaluation"]["validation"]),
              "validation_rollout_count": inputs.skill["evaluation"]["validation"]["rollouts_per_track_per_training_seed"],
              "cell_counts": {s: dict(Counter(r["cell_id"] for r in rows[s])) for s in SPLITS},
              "source_commit": command.stdout.strip() if command.returncode == 0 else None,
              "implementation_sha256": implementation, "environment": environment_manifest(device),
              "selection_status": "awaiting_training_and_validation_rollouts",
              "test_data_used_for_training_or_selection": False}
    result["manifest_digest"] = content_digest(result)
    return result
