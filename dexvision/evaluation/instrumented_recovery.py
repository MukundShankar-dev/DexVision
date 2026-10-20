"""Continue unstarted 5.7 requests after a pre-entry reporting failure.

Completed requests are immutable and never rerun. The original observation,
policy and rollout functions remain unchanged. A failed paired scripted setup
rejects all RGB requests before creating an observer or invoking a policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from dexvision.evaluation.instrumented_rollouts import (
    SKILLS,
    checkpoint_paths,
    execute_attempt,
    load_instrumented_protocol,
    load_runtime,
    summarize_paired,
)
from dexvision.learning.training_manifest import digest_file
from dexvision.learning.train_skill import _json, _verify_run


def failed_place_entry(state):
    return (
        state.get("setup", {}).get("success") is False
        and state["terminal_result"] == "rejected_before_execution"
        and state["executed_actions"] == 0
    )


def reject_failed_place_setup(
    state_path, output, *, condition, allow_empty=False, infrastructure_failure=False
):
    """Reuse measured paired setup failure, never retry it or fabricate control."""
    state_path, output = Path(state_path), Path(output)
    state = json.loads(state_path.read_text())
    if not failed_place_entry(state):
        raise ValueError("Only a measured failed pre-entry setup can skip observation")
    if output.exists():
        if not allow_empty or any(output.iterdir()):
            raise ValueError("Preserve completed/interrupted attempt contents")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with np.load(state_path.parent / "trajectory.npz", allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    if len(arrays["qpos"]) != 1 or len(arrays.get("applied_actions", ())):
        raise ValueError("Failed setup must have no learned control actions")
    arrays["applied_actions"] = np.empty((0, 27))
    np.savez_compressed(output / "trajectory.npz", **arrays)
    result = {
        **state,
        "track": "perception",
        "condition": condition,
        "terminal_result": "rejected_before_execution",
        "executed_actions": 0,
        "inference_latencies_s": [],
        "observation_source": "not_started_preentry_rejected",
        "failure_attribution": "physics_setup",
        "estimated_preview": False,
        "paired_setup_evidence_sha256": digest_file(state_path),
        "physical_setup_reused_for_preentry_rejection": True,
        "infrastructure_failure_before_control": infrastructure_failure,
        "default_registry_admission": False,
    }
    result["execution_id"] = state.get("execution_id", "paired_setup").replace(
        "/state/nominal", f"/perception/{condition}"
    )
    _json(output / "result.json", result)
    return result


def recover_instrumented(config):
    config = Path(config)
    if (
        config.with_suffix(".sha256").read_text().strip()
        != f"{digest_file(config)}  {config.name}"
    ):
        raise ValueError("5.7 recovery recipe checksum mismatch")
    recovery = yaml.safe_load(config.read_text())
    for name, sha in recovery["input_digests"].items():
        if digest_file(Path(name)) != sha:
            raise ValueError(f"Interrupted evidence or recovery input changed: {name}")
    protocol = load_instrumented_protocol(recovery["evaluation_config"])
    _verify_run(Path(protocol["cases_root"]))
    root = Path(recovery["evaluation_dir"])
    reservation = json.loads(
        (Path(protocol["run_root"]) / "test_attempt.json").read_text()
    )
    if (
        Path(reservation["output_dir"]).resolve() != root.resolve()
        or (root / "report.json").exists()
    ):
        raise ValueError("Recovery requires the original incomplete reserved test")
    index_lines = Path(recovery["interrupted_index"]).read_text().splitlines()
    if {line.split("  ", 1)[1] for line in index_lines} != {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    }:
        raise ValueError("Interrupted evidence index is incomplete")
    for line in index_lines:
        sha, name = line.split("  ", 1)
        if digest_file(root / name) != sha:
            raise ValueError(f"Completed test evidence changed: {name}")
    incomplete = {
        p.relative_to(root).as_posix()
        for p in root.glob("*/*/*/*")
        if p.is_dir() and not (p / "result.json").exists()
    }
    if incomplete != set(recovery["empty_interrupted_requests"]):
        raise ValueError("Unexpected interrupted request; no control retry permitted")
    claim = Path(protocol["run_root"]) / "recovery_attempt.json"
    with claim.open("x") as handle:
        json.dump(
            {"recovery_recipe_sha256": digest_file(config), "root": str(root)}, handle
        )
    _json(root / "recovery_protocol.json", recovery)
    interrupted = set(recovery["empty_interrupted_requests"])
    reports = {}
    skipped = created = setup_rejections = 0
    case_root = Path(protocol["cases_root"]) / "test"
    for skill in SKILLS:
        records = json.loads((case_root / skill / "cases.json").read_text())["records"]
        runtime = None
        reports[skill] = {}
        for seed in protocol["training_seeds"]:
            parent = root / skill / str(seed)
            missing = any(
                not (parent / f"{i:03d}" / arm / "result.json").exists()
                for i in range(len(records))
                for arm in ["state", *protocol["conditions"]]
            )
            policies = None
            if missing:
                runtime = runtime or load_runtime(skill)
                path = checkpoint_paths(skill)[seed]
                policies = {
                    arm: runtime.policy(path) for arm in ("state", "perception")
                }
                for policy in policies.values():
                    policy._evaluation_seed = seed
            pairs = []
            for index, record in enumerate(records):
                directory = parent / f"{index:03d}"
                state_path = directory / "state/result.json"
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    skipped += 1
                else:
                    with np.load(
                        case_root / skill / record["prefix"], allow_pickle=False
                    ) as data:
                        prefix = data["actions"]
                    state = execute_attempt(
                        protocol,
                        runtime,
                        skill,
                        record,
                        prefix,
                        policies["state"],
                        directory / "state",
                        track="state",
                        condition="nominal",
                    )
                    created += 1
                for condition in protocol["conditions"]:
                    output = directory / condition
                    result_path = output / "result.json"
                    if result_path.exists():
                        current = json.loads(result_path.read_text())
                        skipped += 1
                    elif skill == "place" and failed_place_entry(state):
                        relative = output.relative_to(root).as_posix()
                        current = reject_failed_place_setup(
                            state_path,
                            output,
                            condition=condition,
                            allow_empty=relative in interrupted,
                            infrastructure_failure=relative in interrupted,
                        )
                        created += 1
                        setup_rejections += 1
                    else:
                        if output.exists():
                            raise ValueError("An interrupted request may not be rerun")
                        with np.load(
                            case_root / skill / record["prefix"], allow_pickle=False
                        ) as data:
                            prefix = data["actions"]
                        current = execute_attempt(
                            protocol,
                            runtime,
                            skill,
                            record,
                            prefix,
                            policies["perception"],
                            output,
                            track="perception",
                            condition=condition,
                            video=seed == 5001
                            and len(list(parent.glob(f"*/{condition}/review.avi")))
                            < protocol["review_cases_per_skill"],
                        )
                        created += 1
                    if state.get("entry_state_sha256") != current.get(
                        "entry_state_sha256"
                    ):
                        raise ValueError("Paired entry drift during recovery")
                    pairs.append({"state": state, "perception": current})
                print(
                    f"5.7 preserved/resumed {skill}/{seed}: {index + 1}/{len(records)}",
                    flush=True,
                )
                if (
                    sum(
                        p.stat().st_size
                        for p in Path(protocol["run_root"]).rglob("*")
                        if p.is_file()
                    )
                    > protocol["maximum_saved_bytes"]
                ):
                    raise ValueError("5.7 evidence storage ceiling reached")
            runtime = runtime or load_runtime(skill)
            reports[skill][str(seed)] = summarize_paired(pairs, runtime)
    report = {
        "checkpoint": "5.7",
        "protocol_sha256": protocol["protocol_sha256"],
        "split": "test",
        "reports": reports,
        "checkpoint_complete": False,
        "manual_acceptance": "pending_owner_review",
        "method": "instrumented_simulator_fiducials_and_explicit_sensors",
        "default_registry_admission": False,
        "state_baseline_reuse": "One independent state execution per reset/model shared by four RGB comparisons; measured failed place setups reject RGB requests without repeating the invalid setup.",
        "recovery": {
            "recipe_sha256": digest_file(config),
            "preserved_completed_attempts": skipped,
            "new_request_dispositions": created,
            "paired_setup_rejections": setup_rejections,
            "rerun_completed_or_interrupted_control": 0,
        },
        "storage_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
    }
    from dexvision.evaluation.instrumented_review import index_review
    from dexvision.perception.perception_baseline import save_artifact_index

    report["review_clips"] = len(index_review(root))
    _json(root / "report.json", report)
    save_artifact_index(root)
    return {key: value for key, value in report.items() if key != "reports"}
