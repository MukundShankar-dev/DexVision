"""Frozen, append-only button replacements for the Level 4.8 leakage gate."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml

from dexvision.evaluation.split_audit import content_digest, file_inventory
from dexvision.logging.level4_collection import sample_level4_procedural_variation
from dexvision.logging.visual_stream import digest_file


def load_button_plan(path: Path) -> dict:
    """Require the exact frozen recipe and eight distinct same-cell assignments."""
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    if plan["version"] != "level4/button-replacement-plan-v1":
        raise ValueError("Unsupported button replacement plan")
    for key in ("dataset_config", "workcell_config", "parent_report"):
        if digest_file(Path(plan[key])) != plan[key + "_sha256"]:
            raise ValueError(f"Button amendment changed input: {key}")
    requirements = yaml.safe_load(Path(plan["dataset_config"]).read_text(encoding="utf-8"))
    parent = json.loads(Path(plan["parent_report"]).read_text(encoding="utf-8"))
    expected = {e["episode_id"] for g in parent["splits"]["cross_split_action_groups"]
                for e in g["episodes"] if e["split"] != "train"}
    rows = plan["assignments"]
    for key in ("original_episode_id", "replacement_episode_id", "recording_session_id", "seed"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate button assignment: {key}")
    if len(rows) != 8 or {r["original_episode_id"] for r in rows} != expected:
        raise ValueError("Button plan must replace exactly the eight non-training duplicates")
    if {r["original_episode_id"] for r in rows} & {r["replacement_episode_id"] for r in rows}:
        raise ValueError("Button replacement ids must be new")
    cells = {c["id"]: c for c in requirements["coverage_cells"]}
    for row in rows:
        cell = cells[row["cell_id"]]
        if (row["split"] not in ("validation", "test") or cell["data_group"] != "button"
                or cell["split_owner"] != row["split"]):
            raise ValueError("Button replacement cell/split mismatch")
        variation = sample_level4_procedural_variation(
            requirements["level4_5b_procedural_expansion"], skill_name="press_button",
            seed=row["seed"])
        if row["variation"] != variation:
            raise ValueError("Button variation differs from the frozen existing sampler")
        for key in ("replacement_episode_id", "recording_session_id"):
            value = row[key]
            if not value or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in value):
                raise ValueError(f"Unsafe button assignment {key}")
    return plan


def recording_variation(args, plan_path: Path) -> dict:
    """Resolve a button recipe before the recorder appends a session or episode."""
    plan = load_button_plan(plan_path)
    row = next((r for r in plan["assignments"] if r["replacement_episode_id"] == args.episode_id), None)
    if row is None:
        raise ValueError("Episode is absent from the frozen button amendment")
    expected = {"task_seed": row["seed"], "session_id": row["recording_session_id"],
                "session_split": row["split"], "goal_condition_id": row["cell_id"],
                "operator_id": plan["operator_id"], "skill_name": "press_button",
                "source": "scripted", "enforce_frozen_cell_owner": True,
                "resume_existing_session": False, "level4_procedural_repetition": None}
    if any(getattr(args, key) != value for key, value in expected.items()):
        raise ValueError("Recording arguments differ from the frozen button amendment")
    for key in ("dataset_config", "workcell_config"):
        actual = args.level4_dataset_config if key == "dataset_config" else args.workcell_config
        if Path(actual).resolve() != Path(plan[key]).resolve():
            raise ValueError("Recording config differs from the button amendment")
    return row["variation"]


def apply_button_replacements(active, *, plan_path: Path, sessions: dict):
    """Verify immutable sources/receipts; fresh replay and leakage gates still follow."""
    plan = load_button_plan(plan_path)
    receipt_path = Path(plan["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt["version"] != "level4/button-replacement-receipt-v1"
            or receipt["plan_sha256"] != digest_file(plan_path)
            or set(receipt["episodes"]) != {r["replacement_episode_id"] for r in plan["assignments"]}):
        raise ValueError("Button replacement receipt mismatch")
    exclusions = []
    for row in plan["assignments"]:
        pair = []
        for key in ("original_episode_id", "replacement_episode_id"):
            matches = [e for e in active if e.episode_id == row[key]]
            if len(matches) != 1:
                raise ValueError("Button amendment requires unique accepted source/replacement")
            episode = matches[0]
            session = sessions.get(episode.session_id)
            expected_digest = (row["original_episode_digest"] if key == "original_episode_id"
                               else receipt["episodes"][row[key]]["episode_digest"])
            if (content_digest(file_inventory(episode.path)) != expected_digest
                    or episode.skill_name != "press_button" or episode.source != "scripted"
                    or episode.metadata.get("data_stream", "expert_success") != "expert_success"
                    or episode.goal_condition_id != row["cell_id"]
                    or session is None or session.split != row["split"]):
                raise ValueError("Button amendment source integrity/ownership mismatch")
            pair.append(episode)
        original, replacement = pair
        if (replacement.session_id != row["recording_session_id"]
                or replacement.session_id == original.session_id
                or replacement.metadata["operator_id"] != plan["operator_id"]
                or replacement.metadata["random_seed"] != row["seed"]
                or replacement.metadata["task_config"].get("procedural_variation") != row["variation"]):
            raise ValueError("Button replacement differs from frozen recording recipe")
        exclusions.append({"episode_id": original.episode_id, "reason": "superseded_by_replacement",
                           "source": original.source, "review_reasons": [],
                           "episode_digest": row["original_episode_digest"],
                           "replacement_episode_id": replacement.episode_id})
    excluded_ids = {r["episode_id"] for r in exclusions}
    return ([e for e in active if e.episode_id not in excluded_ids], exclusions,
            {str(plan_path): digest_file(plan_path), str(receipt_path): digest_file(receipt_path)})


def collect_button_replacements(plan_path: Path, dataset_dir: Path) -> dict:
    """Record every frozen assignment in a fresh process; stop on any failure.

    Existing data is never retried, relabeled or overwritten. A failed run keeps
    its evidence and requires a new plan/namespace before further collection.
    """
    from dexvision.evaluation.level4_expert_audit import audit_scripted_episode
    from dexvision.logging.level4_collection import (
        PilotReview, discover_pilot_episodes, save_pilot_review,
    )
    from dexvision.logging.session_manifest import load_session_manifest
    import mujoco
    import numpy as np

    plan = load_button_plan(plan_path)
    requirements = yaml.safe_load(Path(plan["dataset_config"]).read_text(encoding="utf-8"))
    independence = requirements["level4_5b_procedural_expansion"]["independence_audit"]
    output = Path(plan["receipt_path"]).parent
    if output.exists():
        raise ValueError(f"Button evidence directory already exists: {output}")
    episodes = discover_pilot_episodes(dataset_dir)
    sessions = load_session_manifest(dataset_dir / "session_manifest.json")
    seeds = {e.metadata.get("random_seed") for e in episodes}
    ids = {e.episode_id for e in episodes}
    for row in plan["assignments"]:
        originals = [e for e in episodes if e.episode_id == row["original_episode_id"]]
        if (len(originals) != 1 or not originals[0].expert_accepted
                or content_digest(file_inventory(originals[0].path)) != row["original_episode_digest"]):
            raise ValueError("Button original changed or is not accepted")
        if (row["seed"] in seeds or row["replacement_episode_id"] in ids
                or (dataset_dir / row["recording_session_id"]).exists()
                or any(s.recording_session_id == row["recording_session_id"] for s in sessions.sessions)):
            raise ValueError("Button amendment seed/id/session already used; freeze a new plan")
    output.mkdir(parents=True, exist_ok=False)
    # Preserve an independent baseline including quarantines and all historical files.
    baseline = file_inventory(dataset_dir)
    (output / "source_inventory_before.json").write_text(json.dumps(baseline, sort_keys=True))
    (output / "session_manifest_before.json").write_bytes((dataset_dir / "session_manifest.json").read_bytes())
    snapshot = output / "recording_snapshot"
    for root in (Path("dexvision"), Path("configs")):
        for source in sorted(root.rglob("*")):
            if source.suffix in (".py", ".yaml"):
                target = snapshot / source
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    receipt = {"version": "level4/button-replacement-receipt-v1",
               "plan_sha256": digest_file(plan_path), "episodes": {},
               "environment": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                               "numpy": np.__version__, "platform": platform.platform()},
               "simulator_asset_sha256": file_inventory(Path("assets/mujoco")),
               "recording_snapshot_sha256": file_inventory(snapshot)}
    def descriptor(path):
        actions = np.load(path / "applied_actions.npy", allow_pickle=False).astype(np.float64)
        indices = np.linspace(0, len(actions) - 1,
                              independence["trajectory_descriptor_samples"]).round().astype(int)
        return actions[indices].reshape(-1)

    previous_descriptors = [(e.goal_condition_id, descriptor(e.path)) for e in episodes
                            if e.expert_accepted and e.skill_name == "press_button"]
    for row in plan["assignments"]:
        argv = [sys.executable, "-m", "dexvision.apps.record_demo", "--task", "level4_workcell",
                "--skill", "press_button", "--source", "scripted", "--goal-condition-id", row["cell_id"],
                "--task-seed", str(row["seed"]), "--session-id", row["recording_session_id"],
                "--operator-id", plan["operator_id"], "--session-split", row["split"],
                "--level4-dataset-config", plan["dataset_config"], "--level4-dataset-dir", str(dataset_dir),
                "--workcell-config", plan["workcell_config"], "--episode-id", row["replacement_episode_id"],
                "--level4-button-replacement-plan", str(plan_path),
                "--enforce-frozen-cell-owner", "--print-interval", "1000"]
        subprocess.run(argv, check=True)
        episode_path = dataset_dir / row["recording_session_id"] / "episode_000001"
        audit = audit_scripted_episode(episode_path, config_path=plan["dataset_config"],
                                       workcell_config=plan["workcell_config"])
        (output / f"{row['replacement_episode_id']}_audit.json").write_text(
            json.dumps(audit.to_dict(), indent=2) + "\n")
        if not audit.accepted:
            raise ValueError(f"Button replacement failed qualification: {audit.rejection_reasons}")
        # Check action independence before accepting; background/reset changes alone are insufficient.
        action_digest = digest_file(episode_path / "actions.npy")
        if any(digest == action_digest for name, digest in baseline.items() if name.endswith("/actions.npy")):
            raise ValueError("Button replacement duplicates a preserved action trajectory")
        if any(r["action_sha256"] == action_digest for r in receipt["episodes"].values()):
            raise ValueError("Button replacements duplicate each other")
        current = descriptor(episode_path)
        minimum_distance = min(float(np.linalg.norm(current - previous))
                               for cell, previous in previous_descriptors if cell == row["cell_id"])
        if minimum_distance < independence["minimum_trajectory_descriptor_l2_distance"]:
            raise ValueError("Button replacement is only cosmetic trajectory variation")
        previous_descriptors.append((row["cell_id"], current))
        save_pilot_review(episode_path, PilotReview(
            episode_id=audit.episode_id, schema_validation=audit.schema_validation,
            timestamp_alignment=audit.timestamp_alignment, headless_replay=audit.headless_replay,
            terminal_metric_recomputation=audit.terminal_metric_recomputation,
            recomputed_success=audit.recomputed_success, operator_label_agreement=audit.operator_label_agreement,
            quality_thresholds=True, coverage_assignment=audit.coverage_assignment,
            split_session_leakage_audit=True, expert_accepted=True, rejection_reasons=()))
        receipt["episodes"][audit.episode_id] = {
            "episode_digest": content_digest(file_inventory(episode_path)), "action_sha256": action_digest,
            "minimum_same_cell_descriptor_distance": minimum_distance,
            "qualification": audit.to_dict()}
    for name, digest in baseline.items():
        if name != "session_manifest.json" and digest_file(dataset_dir / name) != digest:
            raise ValueError(f"Preserved source changed: {name}")
    after = load_session_manifest(dataset_dir / "session_manifest.json")
    if after.sessions[:len(sessions.sessions)] != sessions.sessions or len(after.sessions) != len(sessions.sessions) + 8:
        raise ValueError("Session manifest did not preserve existing entries and append exactly eight")
    with Path(plan["receipt_path"]).open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt
