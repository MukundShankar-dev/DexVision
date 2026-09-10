"""Frozen whole-session splits and content integrity for Level 4.8."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from dexvision.logging.visual_stream import SPLITS, digest_file

SPLIT_VERSION = "level4/frozen-splits-v1"


def content_digest(value: object) -> str:
    """Hash canonical JSON independently of output directory or formatting."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_inventory(root: Path) -> dict[str, str]:
    """Inventory all regular files; reject symlinks rather than escape the root."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink is not an immutable dataset file: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest_file(path)
    return result


def checked_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
        raise ValueError(f"Unsafe manifest path: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes root: {relative}")
    return path


def verify_files(root: Path, hashes: dict[str, str]) -> list[str]:
    issues = []
    for relative, expected in hashes.items():
        path = checked_path(root, relative)
        if not path.is_file() or digest_file(path) != expected:
            issues.append(f"Missing or changed file: {relative}")
    return issues


def load_split_config(path: Path, requirements: dict) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "version": SPLIT_VERSION,
        "dataset_config_sha256": digest_file(Path(config["dataset_config"])),
        "ownership_unit": "complete_recording_session",
        "normalization_source": "train_expert_success_only",
        "held_out_object_instances": requirements["split_policy"]["held_out_object_instances"],
        "held_out_goal_regions": requirements["split_policy"]["held_out_goal_regions"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen split config mismatch: {key}")
    bound = yaml.safe_load(Path(config["dataset_config"]).read_text(encoding="utf-8"))
    if requirements != bound:
        raise ValueError("Audit requirements differ from frozen split requirements")
    return config


def audit_splits(records: list[dict], sessions: dict, requirements: dict) -> dict:
    """Check task-relevant ids, cell ownership, lineage, and whole trajectories.

    Shared training-pool identities/goals are permitted by the frozen protocol.
    Full background simulator state is archival truth, never a normalization input.
    """
    issues = []
    cells = {cell["id"]: cell for cell in requirements["coverage_cells"]}
    seen = {}
    action_owners = {}
    action_groups = defaultdict(list)
    initial_owners = {}
    used_sessions = {split: set() for split in SPLITS}
    for row in records:
        eid, split = row["episode_id"], row["split"]
        if eid in seen:
            issues.append(f"Duplicate episode: {eid}")
        seen[eid] = row
        session = sessions.get(row["recording_session_id"])
        if split not in SPLITS or session is None or session.split != split:
            issues.append(f"Session ownership mismatch: {eid}")
            continue
        used_sessions[split].add(session.recording_session_id)
        if row["operator_id"] != session.operator_id:
            issues.append(f"Session operator mismatch: {eid}")
        cell = cells.get(row["cell_id"])
        if cell is None or cell["split_owner"] != split:
            issues.append(f"Cell ownership mismatch: {eid}")
            continue
        if split != "test":
            if row["entity_id"] in requirements["split_policy"]["held_out_object_instances"]:
                issues.append(f"Held-out object leakage: {eid}")
            if row["target_id"] in requirements["split_policy"]["held_out_goal_regions"]:
                issues.append(f"Held-out goal leakage: {eid}")
        if row["stream"] == "expert_success":
            action_groups[row["action_sha256"]].append({"episode_id": eid, "split": split})
            for key, owners in (("action_sha256", action_owners),
                                ("initial_state_digest", initial_owners)):
                digest = row.get(key)
                if digest and owners.setdefault(digest, split) != split:
                    issues.append(f"Cross-split {key} duplicate: {eid}")
    for row in records:
        source = row.get("source_episode_id")
        if source and (source not in seen or seen[source]["split"] != row["split"]):
            issues.append(f"Missing or cross-split source lineage: {row['episode_id']}")
    for split, minimum in (("train", 2), ("validation", 1), ("test", 1)):
        if len(used_sessions[split]) < minimum:
            issues.append(f"Insufficient genuine sessions: {split}")
    return {"passed": not issues, "issues": sorted(set(issues)),
            "cross_split_action_groups": [{"action_sha256": digest, "episodes": group}
                for digest, group in sorted(action_groups.items())
                if len({row["split"] for row in group}) > 1],
            "episode_counts": dict(Counter(r["split"] for r in records)),
            "session_counts": {k: len(v) for k, v in used_sessions.items()},
            "claim_boundary": "Whole-session isolation and task-relevant held-out ids; "
                              "shared training-pool goals are intentional. Raw background "
                              "state is excluded from learner inputs."}


def split_manifests(records: list[dict], frames: list[dict], config_digest: str,
                    *, audit_passed: bool = True) -> dict:
    """Freeze every episode and image assignment, including explicit target intervals."""
    result = {}
    for split in SPLITS:
        rows = sorted((r for r in records if r["split"] == split),
                      key=lambda r: r["episode_id"])
        normalization = [{"episode_id": r["episode_id"], "episode_digest": r["episode_digest"],
                          "frame_interval": [0, r["frames"]],
                          "entity_id": r["entity_id"], "typed_goal": r["typed_goal"],
                          "schema_versions": r["schema_versions"]}
                         for r in rows if split == "train" and r["stream"] == "expert_success"
                         and r["audit_passed"]]
        payload = {"version": SPLIT_VERSION, "split": split, "config_digest": config_digest,
                   "dataset_audit_passed": audit_passed,
                   "status": "frozen" if audit_passed else "diagnostic_only_do_not_train_or_release",
                   "episodes": rows,
                   "images": [f for f in frames if f["split"] == split],
                   "normalization_inputs": normalization,
                   "normalization_rule": "Fit only the listed train expert frames using named "
                   "robot fields, task-relevant entity and typed goal, and complete applied action. "
                   "Exclude background objects, retrospective annotations, failures, corrections, "
                   "validation and test. No statistics are fitted in Level 4.8."}
        payload["manifest_digest"] = content_digest(payload)
        result[split] = payload
    return result
