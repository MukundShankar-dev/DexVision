"""Read-only event audit for perception-interrupted 5.7 trajectories.

No renderer, policy inference or recorded-trajectory integration. Original
attempts remain immutable; this independently bound supplement prevents missing
event fields from silently becoming zero-rate qualification evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from dexvision.evaluation.pick_policy import PickEvidence
from dexvision.evaluation.place_policy import PlaceEvidence
from dexvision.learning.training_manifest import digest_file
from dexvision.learning.train_skill import _json, _verify_run
from dexvision.logging.level4_collection import WorkcellPilotTask

EVENT_FIELDS = {
    "pick": ("slip_drop",),
    "place": ("premature_release", "nominally_placed", "settled"),
}


def event_rates(rows, skill):
    """All scheduled attempts count; unknown events explicitly fail coverage."""
    fields = EVENT_FIELDS[skill]
    complete = all(all(row.get(name) is not None for name in fields) for row in rows)
    gates = {"event_coverage": complete}
    if skill == "pick":
        rate = sum(row.get("slip_drop") is True for row in rows) / len(rows)
        gates["drop_rate"] = complete and rate <= 0.1
        return {"attempts": len(rows), "drop_rate": rate, "gates": gates}
    premature = sum(row.get("premature_release") is True for row in rows) / len(rows)
    nominal = sum(row.get("nominally_placed") is True for row in rows)
    settled = sum(
        row.get("nominally_placed") is True and row.get("settled") is True
        for row in rows
    )
    fraction = settled / nominal if nominal else None
    gates.update(
        premature_release=complete and premature <= 0.1,
        settled_placement=complete and fraction is not None and fraction >= 0.9,
    )
    return {
        "attempts": len(rows),
        "premature_release_rate": premature,
        "nominally_placed": nominal,
        "settled_among_nominally_placed": fraction,
        "gates": gates,
    }


def reconstruct_events(skill, record, arrays, terminal):
    """Rebuild post-action contacts/motion with mj_forward only, never mj_step."""
    row = record["row"]
    with WorkcellPilotTask(
        workcell_config="configs/workcell.yaml",
        dataset_config="configs/level4_dataset.yaml",
        skill_name="pick_place_sequence",
        goal_condition_id=row["cell_id"],
        seed=row["seed"],
        procedural_variation=row["variation"],
    ) as task:
        if skill == "pick":
            evidence = PickEvidence(
                row["entity_id"],
                task.initial_world_state.require_entity(row["entity_id"]).position[2],
                terminal,
            )
        else:
            task.env.data.qpos[:] = arrays["qpos"][0]
            task.env.data.qvel[:] = arrays["qvel"][0]
            task.env._mujoco.mj_forward(task.env.model, task.env.data)
            initial = (
                task.workcell.get_world_state()
                .require_entity(row["entity_id"])
                .position
            )
            evidence = PlaceEvidence(
                row["entity_id"], row["typed_goal"]["target_id"], terminal, initial
            )
        drop = False
        for index in range(1, len(arrays["qpos"])):
            task.env.data.qpos[:] = arrays["qpos"][index]
            task.env.data.qvel[:] = arrays["qvel"][index]
            task.env.data.time = float(arrays["state_timestamps"][index])
            task.env._mujoco.mj_forward(task.env.model, task.env.data)
            world = task.workcell.get_world_state()
            if skill == "pick":
                metrics = evidence.update(world)
                drop |= metrics["slip_drop"]
            else:
                evidence.update(world, SimpleNamespace(values={}, success=False))
        if skill == "pick":
            return {"slip_drop": bool(drop)}
        return {
            "premature_release": bool(evidence.premature_release),
            "nominally_placed": bool(evidence.nominally_placed),
            "settled": bool(
                evidence.settle_dwell >= terminal["consecutive_samples"]
                and not evidence.bounced
            ),
        }


def audit_instrumented_events(evaluation_dir, output_dir, config):
    """Fill missing evaluation-only event evidence without changing a rollout."""
    root, output, config = Path(evaluation_dir), Path(output_dir), Path(config)
    if output.exists():
        raise ValueError(
            "Preserve existing event audits; choose a new output directory"
        )
    if (
        config.with_suffix(".sha256").read_text().strip()
        != f"{digest_file(config)}  {config.name}"
    ):
        raise ValueError("5.7 event audit recipe checksum mismatch")
    recipe = yaml.safe_load(config.read_text())
    for name, sha in recipe["input_digests"].items():
        if digest_file(Path(name)) != sha:
            raise ValueError(f"Frozen event analysis input changed: {name}")
    _verify_run(root)
    report = json.loads((root / "report.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    from dexvision.evaluation.instrumented_rollouts import load_runtime

    output.mkdir(parents=True, exist_ok=False)
    summaries, records = {}, []
    for skill in EVENT_FIELDS:
        runtime = load_runtime(skill)
        cases = json.loads(
            (
                Path(protocol["cases_root"]) / report["split"] / skill / "cases.json"
            ).read_text()
        )["records"]
        groups = {}
        for seed in report["reports"][skill]:
            for index, record in enumerate(cases):
                for directory in sorted(
                    (root / skill / seed / f"{index:03d}").iterdir()
                ):
                    result_path = directory / "result.json"
                    result = json.loads(result_path.read_text())
                    fields = EVENT_FIELDS[skill]
                    if all(result.get(name) is not None for name in fields):
                        events = {name: bool(result[name]) for name in fields}
                        method = "original_physical_report"
                    elif result["executed_actions"] == 0:
                        events = dict.fromkeys(fields, False)
                        method = "no_policy_actions_no_policy_event"
                    else:
                        with np.load(
                            directory / "trajectory.npz", allow_pickle=False
                        ) as data:
                            arrays = {name: data[name].copy() for name in data.files}
                        if len(arrays["qpos"]) != result["executed_actions"] + 1:
                            raise ValueError(
                                "Interrupted event trace lacks post-action states"
                            )
                        events = reconstruct_events(
                            skill, record, arrays, runtime.spec["terminal"]
                        )
                        method = "independent_saved_state_forward_reconstruction"
                    arm = "state" if directory.name == "state" else "perception"
                    key = f"{seed}/{arm}"
                    groups.setdefault(key, []).append(events)
                    record_out = {
                        "attempt": directory.relative_to(root).as_posix(),
                        "events": events,
                        "method": method,
                        "result_sha256": digest_file(result_path),
                        "trajectory_sha256": digest_file(directory / "trajectory.npz"),
                    }
                    records.append(record_out)
                    print(
                        f"5.7 event audit {record_out['attempt']}: {method}", flush=True
                    )
        summaries[skill] = {
            key: event_rates(rows, skill) for key, rows in groups.items()
        }
    value = {
        "checkpoint": "5.7",
        "split": report["split"],
        "audit_recipe_sha256": digest_file(config),
        "report_sha256": digest_file(root / "report.json"),
        "summaries": summaries,
        "records": records,
        "checkpoint_complete": False,
        "manual_acceptance": "pending_owner_confirmation",
        "default_registry_admission": False,
        "scope": "Missing event fields reconstructed from already recorded physical states; no RGB/control rerun or metric-threshold changes",
    }
    _json(output / "report.json", value)
    from dexvision.perception.perception_baseline import save_artifact_index

    save_artifact_index(output)
    return {key: value[key] for key in value if key != "records"}
