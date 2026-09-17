"""Frozen Level 5.2 reach evaluation, validation selection and saved-state review.

Only reach/state is implemented. Policies receive causal named features, never
source actions. Each matrix member is executed independently in all conditions;
render-only conditions have no physical or state-observation effect on this track.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dexvision.evaluation.benchmark_retargeters import mean_action_jerk
from dexvision.learning.skill_datasets import (
    _entity_vector, _restore_initial, feature_schema, numeric_goal, rotation6d,
)
from dexvision.learning.skill_models import SkillMLP, decode_delta, model_schema
from dexvision.learning.train_skill import (
    _checksum_index, _json, _verify_run, load_checkpoint, select_checkpoint,
)
from dexvision.learning.training_manifest import (
    ReleaseSource, TrainingInputs, content_digest, digest_file, load_training_inputs,
)
from dexvision.logging.level4_collection import WorkcellPilotTask

SAFETY_KEYS = ("invalid_actions", "workspace_violations", "joint_limit_violations",
               "object_workspace_violations")


def create_reach_task(inputs: TrainingInputs, row: dict, metadata: dict) -> WorkcellPilotTask:
    """Recreate the released reset without reading any demonstration action."""
    for key in ("episode_id", "recording_session_id", "typed_goal"):
        if metadata[key] != row[key]:
            raise ValueError(f"Reset metadata disagrees with split: {key}")
    task = WorkcellPilotTask(
        workcell_config=inputs.root / inputs.plan["inputs"]["workcell_config"]["path"],
        dataset_config=inputs.root / inputs.plan["inputs"]["dataset_config"]["path"],
        skill_name="reach_object", goal_condition_id=row["cell_id"], seed=metadata["random_seed"],
        procedural_variation=metadata["task_config"].get("procedural_variation"))
    try:
        _restore_initial(task, metadata)
        task._configure_pilot_cue()
        task._task.reset_dwell()
        return task
    except Exception:
        task.close()
        raise


def _joint_values(env, names, *, velocity=False):
    suffixes = ("vx", "vy", "vz", "wx", "wy", "wz") if velocity else ("x", "y", "z", "qw", "qx", "qy", "qz")
    values = env.data.qvel if velocity else env.data.qpos
    result = []
    for name in names:
        joint_name, _, suffix = name.partition("/")
        joint = env.model.joint(joint_name)
        address = int((joint.dofadr if velocity else joint.qposadr)[0])
        result.append(values[address + (suffixes.index(suffix) if suffix else 0)])
    return np.asarray(result)


def reach_features(task, world, row: dict, inputs: TrainingInputs, prior_commanded,
                   prior_applied, previous_safety, phase: str) -> np.ndarray:
    """[192] unnormalized pre-action features, identical to the training layout."""
    groups = inputs.plan["observations"]["feature_groups"]
    by_group = {g["group"]: g for g in groups}
    pose = _joint_values(task.env, inputs.plan["observations"]["robot_fields"]["robot_qpos"])
    parts = {"robot_position": pose[:3], "robot_rotation": rotation6d(pose[3:]),
             "robot_velocity": _joint_values(task.env,
                 [n.removeprefix("robot.velocity.") for n in by_group["robot_velocity"]["names"]], velocity=True),
             "prior_commanded_actions": np.asarray(prior_commanded),
             "prior_applied_actions": np.asarray(prior_applied),
             "previous_safety_masks": np.asarray(previous_safety),
             "online_phase": np.array([phase == p for p in inputs.plan["observations"]["phase_vocabulary"]]),
             "entity": _entity_vector(world, row["entity_id"], inputs.plan)}
    for name in ("finger_joint_positions", "finger_joint_velocities"):
        parts[name] = _joint_values(task.env, [n.removeprefix(name + ".") for n in by_group[name]["names"]],
                                    velocity=name.endswith("velocities"))
    if any(parts[g["group"]].shape != tuple(g["shape"]) for g in groups):
        raise ValueError("Live feature group disagrees with frozen layout")
    vector = np.r_[np.concatenate([parts[g["group"]] for g in groups]), numeric_goal(inputs.skill, row, world)]
    if not np.isfinite(vector).all():
        raise ValueError("Non-finite live observation")
    return vector


class ReachPolicy:
    """Verified standalone MLP; inference has no expert or saved-action input."""

    def __init__(self, checkpoint: dict, inputs: TrainingInputs):
        self.schema = checkpoint["schema"]
        manifest = checkpoint["manifest"]
        if (manifest["skill"] != "reach_object" or manifest["digests"] != inputs.digests
                or manifest["seed"] not in inputs.plan["training"]["seeds"]
                or self.schema != model_schema(inputs.plan, inputs.skill, manifest["model"])
                or checkpoint["settings"] != inputs.plan["training"]):
            raise ValueError("Checkpoint inputs, schema or training protocol are incompatible")
        if content_digest({k: v for k, v in manifest.items() if k != "manifest_digest"}) != manifest["manifest_digest"]:
            raise ValueError("Training manifest digest mismatch")
        norm = checkpoint["normalization"]
        if content_digest({k: v for k, v in norm.items() if k != "normalization_digest"}) != norm["normalization_digest"]:
            raise ValueError("Normalization digest mismatch")
        names, _ = feature_schema(inputs.plan, inputs.skill)
        owners = {r["episode_id"] for r in manifest["assignments"]["train"]}
        if norm["ordered_feature_names"] != list(names) or set(norm["training_episode_ids"]) != owners:
            raise ValueError("Normalization names or training ownership mismatch")
        self.mean, self.std = np.asarray(norm["mean"]), np.asarray(norm["std"])
        if (self.mean.shape != (len(names),) or self.std.shape != self.mean.shape
                or not np.isfinite(np.r_[self.mean, self.std]).all() or np.any(self.std <= 0)):
            raise ValueError("Invalid normalization statistics")
        self.model = SkillMLP(self.schema)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.plan = inputs.plan

    def __call__(self, features, prior, phase):
        normalized = ((features - self.mean) / self.std).astype(np.float32)
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(normalized[None]))[0].numpy()
        if self.schema["model"] == "baseline":
            try:
                action = decode_delta(prediction, prior, phase, self.plan)
            except ValueError:
                action = np.full(27, np.nan)
        else:
            action = prediction.astype(float)
            relevance = self.plan["action"]["phase_relevance"][phase]
            mask = np.array([relevance[f["group"]] for f in self.plan["action"]["named_layout"]])
            action[~mask] = prior[~mask]
        return action, prediction


def safe_action(raw, prior, plan: dict):
    """Check raw limits, preserve violations, clip and produce a finite fallback.

    Counts are violating values, matching Level 3 reporting. An unsafe action
    terminates its rollout even when its application has been bounded.
    """
    raw, prior = np.asarray(raw, float), np.asarray(prior, float)
    counts = dict.fromkeys(SAFETY_KEYS, 0)
    mask = np.zeros(27, bool)
    if raw.shape != (27,) or not np.isfinite(raw).all():
        counts["invalid_actions"] = max(1, int(np.count_nonzero(~np.isfinite(raw))))
        return prior.copy(), np.ones(27, bool), counts, "invalid_action"
    fields = plan["action"]["named_layout"]
    lower, upper = (np.array([f[k] for f in fields]) for k in ("lower", "upper"))
    mask = (raw < lower) | (raw > upper)
    counts["workspace_violations"] = int(mask[:3].sum())
    counts["joint_limit_violations"] = int(mask[7:].sum())
    applied = np.clip(raw, lower, upper)
    length = np.linalg.norm(applied[3:7])
    if length < 1e-12:
        counts["invalid_actions"] += 1
        return prior.copy(), np.ones(27, bool), counts, "invalid_action"
    applied[3:7] /= length
    if applied[3:7] @ prior[3:7] < 0:
        applied[3:7] *= -1
    reason = "workspace_violation" if counts["workspace_violations"] else (
        "joint_limit_violation" if counts["joint_limit_violations"] else None)
    return applied, mask, counts, reason


def apply_action(task, applied, fields):
    task.env.set_mocap_pose(str(task.workcell.config.scene["hand_base_target"]),
                           position=applied[:3], orientation_quat=applied[3:7])
    task.env.set_joint_targets({f["name"].split("/", 1)[1]: float(applied[f["index"]]) for f in fields[7:]})


def check_preconditions(world, row, skill):
    checks = {"robot.safe_neutral": bool(world.robot.safe_neutral), "goal.entity_id": False,
              "entity.state_age_s": False}
    try:
        entity = world.require_entity(row["typed_goal"]["entity_id"], maximum_age_s=0.1)
        checks["goal.entity_id"] = entity.object_id in skill["contract"]["goal_fields"]["entity_id"]["allowed_ids"]
        checks["entity.state_age_s"] = True
    except ValueError:
        pass
    return checks


def run_reach_rollout(inputs, row, metadata, policy, *, condition: str, output_dir: Path) -> dict:
    """Execute one attempt and retain initial/final states even on rejection."""
    output_dir.mkdir(parents=True, exist_ok=False)
    fields = inputs.plan["action"]["named_layout"]
    ranges = np.array([f["upper"] - f["lower"] for f in fields])
    arrays = defaultdict(list)
    world_records, metric_records = [], []
    counters = dict.fromkeys(SAFETY_KEYS, 0)
    reason, terminal = "timeout", "timed_out"
    metrics = {}
    with create_reach_task(inputs, row, metadata) as task:
        env = task.env
        prior = np.asarray(metadata["initial_applied_action"], float)
        commanded = np.asarray(metadata["initial_commanded_action"], float)
        previous_safety = np.zeros(27)
        phase = metadata["initial_online_phase"]
        initial_world = task.workcell.get_world_state()
        checks = check_preconditions(initial_world, row, inputs.skill)
        start = float(env.data.time)
        world_records.append(asdict(initial_world))
        arrays["qpos"].append(env.data.qpos.copy())
        arrays["qvel"].append(env.data.qvel.copy())
        arrays["state_timestamps"].append(start)
        if not all(checks.values()):
            reason, terminal = "rejected_precondition", "rejected_before_execution"
        else:
            for _ in range(inputs.skill["contract"]["timeout_steps"]):
                world = task.workcell.get_world_state()
                features = reach_features(task, world, row, inputs, commanded, prior, previous_safety, phase)
                raw, prediction = policy(features, prior, phase)
                applied, mask, count, unsafe = safe_action(raw, prior, inputs.plan)
                for key in SAFETY_KEYS:
                    counters[key] += count[key]
                for name, value in (("observations", features), ("raw_actions", raw), ("policy_predictions", prediction),
                                    ("commanded_actions", raw), ("applied_actions", applied),
                                    ("prior_commanded_actions", commanded), ("prior_applied_actions", prior),
                                    ("safety_masks", mask), ("phases", phase), ("action_timestamps", float(env.data.time))):
                    arrays[name].append(np.asarray(value).copy())
                apply_action(task, applied, fields)
                state = task.step(n_steps=inputs.plan["action"]["simulation_steps_per_action"])
                metrics = dict(state.task_values)
                metric_records.append(metrics)
                world_records.append(asdict(state.world_state))
                arrays["qpos"].append(env.data.qpos.copy())
                arrays["qvel"].append(env.data.qvel.copy())
                arrays["state_timestamps"].append(float(env.data.time))
                # Wrong contact is an explicit failure even if pose/dwell passes.
                entities = {e.object_id for e in state.world_state.entities}
                wrong = any(any(n.startswith("rh_") for n in pair) and
                            any(n in entities and n != row["entity_id"] for n in pair)
                            for pair in state.world_state.contacts)
                board = inputs.requirements["workcell"]["board_workspace"]
                off_board = any(np.any(np.asarray(state.world_state.require_entity(name).position[:2]) <
                                      np.asarray(board["min_xy_m"]) + board["safe_edge_margin_m"]) or
                                np.any(np.asarray(state.world_state.require_entity(name).position[:2]) >
                                      np.asarray(board["max_xy_m"]) - board["safe_edge_margin_m"])
                                for name in task.workcell.config.object_ids)
                if off_board:
                    counters["object_workspace_violations"] += 1
                if state.failure_reason == "workspace_violation" and not count["workspace_violations"]:
                    counters["workspace_violations"] += 1
                arrays["safety_reasons"].append(unsafe or state.failure_reason or ("object_workspace_violation" if off_board else "none"))
                prior, commanded, previous_safety, phase = applied, raw, mask, task._phase
                if unsafe or state.failure_reason or off_board or wrong:
                    reason = unsafe or state.failure_reason or ("object_workspace_violation" if off_board else "wrong_contact")
                    terminal = "failed"
                    break
                if state.success:
                    reason, terminal = "pose_and_dwell_passed", "succeeded"
                    break
        end = float(env.data.time)
    lengths = {"observations": 192, "raw_actions": 27, "commanded_actions": 27, "applied_actions": 27,
               "prior_commanded_actions": 27, "prior_applied_actions": 27, "safety_masks": 27}
    saved = {name: np.asarray(value) for name, value in arrays.items()}
    for name, width in lengths.items():
        saved[name] = np.asarray(arrays[name]).reshape(-1, width)
    np.savez_compressed(output_dir / "trajectory.npz", **saved)
    _json(output_dir / "world_states.json", {"states": world_records, "metrics": metric_records})
    errors = [metrics.get(t["metric"]) for t in inputs.skill["goal_error_terms"]]
    error = float(np.mean([v / t["scale"] for v, t in zip(errors, inputs.skill["goal_error_terms"], strict=True)])) if all(v is not None for v in errors) else None
    entity = initial_world.require_entity(row["entity_id"])
    category = entity.class_id.removeprefix("rigid_")
    variation = metadata["task_config"].get("procedural_variation") or {}
    result = {"episode_id": row["episode_id"], "episode_digest": row["episode_digest"],
              "reset_seed": metadata["random_seed"], "start_frame": 0, "cell_id": row["cell_id"],
              "session": row["recording_session_id"], "object_instance": row["entity_id"],
              "object_family": category if category in {"cuboid", "cylinder", "flat_puck"} else "not_applicable",
              "target_id": row["entity_id"], "target_type": category,
              "boundary_class": variation.get("case_class", "unspecified_in_source"),
              "render_condition": condition, "render_condition_effect": "none_on_state_or_physics; no_RGB_claim",
              "typed_goal": row["typed_goal"], "request_id": output_dir.name, "execution_id": output_dir.name,
              "start_timestamp": start, "end_timestamp": end, "preconditions": checks,
              "terminal_result": terminal, "terminal_reason": reason, "final_metrics": metrics,
              "retry_count": 0, "safety_counters": counters, "normalized_terminal_error": error,
              "mean_action_jerk": mean_action_jerk(saved["applied_actions"] / ranges) if len(saved["applied_actions"]) else 0.0,
              "trajectory_path": "trajectory.npz", "world_states_path": "world_states.json"}
    _json(output_dir / "result.json", result)
    _checksum_index(output_dir)
    return result


def frozen_scenarios(inputs, split):
    if split not in {"validation", "test"}:
        raise ValueError("Use the frozen validation or test matrix")
    matrix = inputs.skill["evaluation"][split]
    rows = {r["episode_id"]: r for r in inputs.splits[split]["episodes"]}
    scenarios = []
    for cell in matrix["cells"]:
        for eid, seed, frame in zip(cell["episode_ids"], cell["reset_seeds"], cell["start_frames"], strict=True):
            row = rows[eid]
            if row["cell_id"] != cell["cell_id"] or row["skill"] != "reach_object" or frame != 0:
                raise ValueError("Incompatible reach matrix member; no alternative start search")
            for condition in inputs.config_snapshot["evaluation"]["rollouts"]["perception_conditions"]:
                scenarios.append((row, seed, condition))
    if (len(scenarios) != matrix["rollouts_per_track_per_training_seed"] or
            len({r[0]["episode_id"] for r in scenarios}) != matrix["unique_reset_count"]):
        raise ValueError("Incomplete frozen matrix")
    return scenarios


def summarize_rollouts(results, inputs, split):
    expected = {(r["episode_id"], c) for r, _, c in frozen_scenarios(inputs, split)}
    actual = {(r["episode_id"], r["render_condition"]) for r in results}
    if actual != expected or len(results) != len(expected):
        raise ValueError("Missing, duplicate or extra scheduled attempts")
    breakdowns = {}
    for field in inputs.config_snapshot["evaluation"]["reporting"]["required_breakdowns"]:
        if field in {"seed", "track"}:
            continue  # Fixed per report, never pooled across seeds.
        groups = defaultdict(list)
        for r in results:
            groups[r[field]].append(r)
        breakdowns[field] = {k: {"count": len(v), "success_rate": sum(r["terminal_result"] == "succeeded" for r in v) / len(v),
                                 "terminal_reasons": dict(Counter(r["terminal_reason"] for r in v)),
                                 "safety_counters": {key: sum(r["safety_counters"][key] for r in v) for key in SAFETY_KEYS}}
                             for k, v in sorted(groups.items())}
    success = sum(r["terminal_result"] == "succeeded" for r in results) / len(results)
    worst = min(v["success_rate"] for field in ("object_family", "target_type")
                for key, v in breakdowns[field].items() if key != "not_applicable")
    counts = {k: sum(r["safety_counters"][k] for r in results) for k in SAFETY_KEYS}
    jerk = float(np.mean([r["mean_action_jerk"] for r in results]))
    gates = {"aggregate_success": success >= inputs.skill["gates"]["state_success_min"],
             "worst_family_success": worst >= inputs.skill["gates"]["state_worst_family_success_min"],
             "mean_action_jerk": jerk <= inputs.skill["gates"]["mean_action_jerk_max"],
             **{k: v == 0 for k, v in counts.items()},
             "explicit_terminal_results": all(r["terminal_result"] in inputs.config_snapshot["evaluation"]["terminal_results"] for r in results)}
    error = None if any(r["normalized_terminal_error"] is None for r in results) else float(np.mean([r["normalized_terminal_error"] for r in results]))
    return {"rollout_count": len(results), "unique_reset_count": len({r["episode_id"] for r in results}),
            "success_rate": success, "worst_family_success_rate": worst,
            "worst_cell_success_rate": min(v["success_rate"] for v in breakdowns["cell_id"].values()),
            "normalized_terminal_error": error,
            "terminal_error_unavailable_reason": "rejected_precondition" if error is None else None,
            "safety_counters": counts, "safety_violation_count": sum(counts.values()),
            "mean_action_jerk": jerk, "gates": gates, "state_numerical_gates_passed": all(gates.values()),
            "status": "failed" if sum(counts.values()) or not gates["explicit_terminal_results"] else "experimental",
            "state_qualified": False, "manual_acceptance": "pending" if all(gates.values()) else "not_required_numerical_gates_failed",
            "terminal_reasons": dict(Counter(r["terminal_reason"] for r in results)), "breakdowns": breakdowns}


def evaluate_checkpoint(inputs, checkpoint_path: Path, *, split: str, output_dir: Path,
                        release_root: Path | None = None) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    policy = ReachPolicy(checkpoint, inputs)
    manifest = checkpoint["manifest"]
    scenarios = frozen_scenarios(inputs, split)
    run_dir = checkpoint_path.parent if checkpoint_path.parent.name != "validation_candidates" else checkpoint_path.parent.parent
    if split == "test":
        _verify_run(run_dir)
        selection = json.loads((run_dir / "selection.json").read_text())
        if checkpoint_path.name != "selected.pt" or selection["status"] != "selected_from_validation":
            raise ValueError("Test requires the validation-selected checkpoint")
        source = run_dir / "validation_candidates" / f"epoch_{selection['epoch']:04d}.pt"
        if digest_file(source) != selection["source_checkpoint_sha256"] or checkpoint["training_state"]["completed_epochs"] != selection["epoch"]:
            raise ValueError("Selected checkpoint does not match validation evidence")
        chosen = load_checkpoint(source)
        if any(not torch.equal(v, chosen["model"][k]) for k, v in checkpoint["model"].items()):
            raise ValueError("Selected weights differ from selected candidate")
        if (run_dir / "test_state_attempt.json").exists():
            raise ValueError("This run already reserved its single test evaluation; retained attempts cannot be repeated")
    unique = {r["episode_id"]: r for r, _, _ in scenarios}
    # Reach starts at frame zero. Read metadata only, including during test.
    names = [f"data/demos/level4/{r['source_path']}/metadata.json" for r in unique.values()]
    blobs = ReleaseSource(inputs, release_root).read_many(names)
    metadata = {r["episode_id"]: json.loads(blobs[f"data/demos/level4/{r['source_path']}/metadata.json"]) for r in unique.values()}
    for row, seed, _ in scenarios:
        if metadata[row["episode_id"]]["random_seed"] != seed:
            raise ValueError("Frozen reset seed does not match archived metadata")
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = {"version": "level5/reach-evaluation-v1", "skill": "reach_object", "track": "state", "split": split,
                  "seed": manifest["seed"], "epoch": checkpoint["training_state"]["completed_epochs"],
                  "model": manifest["model"], "checkpoint_sha256": digest_file(checkpoint_path),
                  "manifest_digest": manifest["manifest_digest"], "input_digests": inputs.digests,
                  "normalization_sha256": checkpoint["normalization"]["normalization_digest"],
                  "validation_matrix_sha256": manifest["validation_matrix_sha256"],
                  "matrix_sha256": content_digest(inputs.skill["evaluation"][split]),
                  "evaluation_implementation_sha256": digest_file(Path(__file__)),
                  "execution_implementation_sha256": {p.relative_to(inputs.root).as_posix(): digest_file(p)
                                                      for p in sorted((inputs.root / "dexvision").rglob("*.py"))},
                  "checkpoint_path": str(checkpoint_path.resolve()), "output_dir": str(output_dir.resolve()),
                  "selection_uses_test": False}
    _json(output_dir / "provenance.json", provenance)
    if split == "test":
        with (run_dir / "test_state_attempt.json").open("x") as stream:
            json.dump(provenance, stream, indent=2)
    results = []
    for index, (row, _, condition) in enumerate(scenarios):
        directory = output_dir / f"{index:03d}_{row['episode_id']}_{condition}"
        result = run_reach_rollout(inputs, row, metadata[row["episode_id"]], policy, condition=condition, output_dir=directory)
        result["result_path"] = f"{directory.name}/result.json"
        results.append(result)
    report = {**provenance, **summarize_rollouts(results, inputs, split), "results": results,
              "training_history": checkpoint["training_state"]["history"]}
    _json(output_dir / "report.json", report)
    _checksum_index(output_dir)
    if split == "test":
        _checksum_index(run_dir)
    return report


def select_reach_candidates(inputs, run_dir: Path, *, output_dir: Path, release_root=None):
    _verify_run(run_dir)
    checkpoint = load_checkpoint(run_dir / "last.pt")
    if not checkpoint["training_state"]["finished"] or (run_dir / "selected.pt").exists():
        raise ValueError("Candidate evaluation requires finished, unselected training")
    output_dir.mkdir(parents=True, exist_ok=False)
    reports = []
    for epoch, candidate in sorted(checkpoint["training_state"]["candidates"].items(), key=lambda item: int(item[0])):
        print(f"Validation {run_dir.name}: epoch {epoch}", flush=True)
        reports.append(evaluate_checkpoint(inputs, run_dir / candidate["path"], split="validation",
                       output_dir=output_dir / f"epoch_{int(epoch):04d}", release_root=release_root))
    # Subdirectories inside the run are new artifacts; bind them before selection.
    _checksum_index(run_dir)
    result = select_checkpoint(run_dir, reports)
    _checksum_index(output_dir)
    return result


def review_saved_rollouts(inputs, checkpoint_path: Path):
    """Viewer replays saved states, never reruns or selects on the test matrix."""
    import mujoco.viewer

    run_dir = checkpoint_path.parent
    _verify_run(run_dir)
    attempt = json.loads((run_dir / "test_state_attempt.json").read_text())
    directory = Path(attempt["output_dir"])
    _verify_run(directory)
    report = json.loads((directory / "report.json").read_text())
    if report["checkpoint_sha256"] != digest_file(checkpoint_path) or not report["state_numerical_gates_passed"]:
        raise ValueError("Manual review requires this checkpoint's passing numerical report")
    chosen = {}
    for result in report["results"]:
        chosen.setdefault((result["object_family"], result["boundary_class"]), result)
    rows = {r["episode_id"]: r for r in inputs.splits["test"]["episodes"]}
    for result in chosen.values():
        row = rows[result["episode_id"]]
        name = f"data/demos/level4/{row['source_path']}/metadata.json"
        metadata = json.loads(ReleaseSource(inputs).read_many([name])[name])
        trajectory = np.load(directory / Path(result["result_path"]).parent / "trajectory.npz", allow_pickle=False)
        with create_reach_task(inputs, row, metadata) as task:
            print(f"Review {row['entity_id']} {result['boundary_class']}: {result['terminal_result']}", flush=True)
            with mujoco.viewer.launch_passive(task.env.model, task.env.data) as viewer:
                for i, (qpos, qvel) in enumerate(zip(trajectory["qpos"], trajectory["qvel"], strict=True)):
                    if not viewer.is_running():
                        return {"manual_acceptance": "pending", "cancelled": True}
                    if i:
                        apply_action(task, trajectory["applied_actions"][i - 1], inputs.plan["action"]["named_layout"])
                    task.env.data.qpos[:] = qpos
                    task.env.data.qvel[:] = qvel
                    task.env.data.time = trajectory["state_timestamps"][i]
                    task.env._mujoco.mj_forward(task.env.model, task.env.data)
                    viewer.sync()
                    time.sleep(inputs.plan["action"]["control_interval_s"])
    return {"manual_acceptance": "pending_owner_confirmation", "reviewed": len(chosen)}


def level3_compatibility(checkpoint_path: Path, inputs) -> dict:
    """Inspect the pinned original policy; do not invent a new input/goal adapter."""
    expected = "d30f3122b50047693426aa7eb3e52616e7c0d1a8b3174370703ee313aade5356"
    actual = digest_file(checkpoint_path)
    if actual != expected:
        raise ValueError("Level 3 selected reach checkpoint differs from published evidence")
    legacy = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    schema = legacy["schema"]
    current = model_schema(inputs.plan, inputs.skill)
    return {"checkpoint_sha256": actual, "legacy_schema": schema, "compatible": False,
            "matching_full_action_layout": schema["dataset_action_names"] == current["full_action_names"],
            "reasons": ["Legacy contact-target xyz goal lacks entity id and approach orientation/dwell/disturbance semantics",
                        "Legacy observation includes tracking quality and an untyped object_state; frozen Level 5 uses causal entity features"],
            "compatible_level4_scenario_count": 0, "legacy_rollouts_run": 0,
            "historical_level3_4": {"attempts": 35, "workspace_violations": 7, "joint_limit_violations": 14},
            "historical_validation_selected": {"attempts": 35, "success_rate": 0.0, "workspace_violations": 20, "joint_limit_violations": 15},
            "comparison_limit": "Historical matrices and goals differ; no paired legacy transfer claim"}


def run_evaluation_command(config: Path, *, checkpoint: Path | None, output_dir: Path | None,
                           track="state", split="test", select_candidates: Path | None = None,
                           release_root=None, viewer=False, scenario_set=None):
    inputs = load_training_inputs(config)
    if inputs.skill["skill"] != "reach_object" or track != "state":
        raise ValueError("Checkpoint 5.2 implements reach_object/state only")
    if viewer or scenario_set:
        if not viewer or scenario_set != "manual_stratified" or checkpoint is None or select_candidates:
            raise ValueError("Use --checkpoint selected.pt --viewer --scenario-set manual_stratified")
        return review_saved_rollouts(inputs, checkpoint)
    if output_dir is None:
        raise ValueError("Headless evaluation requires a new --output-dir")
    if select_candidates:
        if checkpoint is not None:
            raise ValueError("Choose --select-candidates or --checkpoint")
        return select_reach_candidates(inputs, select_candidates, output_dir=output_dir, release_root=release_root)
    if checkpoint is None:
        raise ValueError("Evaluation requires --checkpoint")
    return evaluate_checkpoint(inputs, checkpoint, split=split, output_dir=output_dir, release_root=release_root)


def compare_reach_runs(config: Path, run_dirs: list[Path], output_dir: Path, legacy_checkpoint: Path):
    """Report every seed, frozen-family comparison, curves and legacy limitations."""
    inputs = load_training_inputs(config)
    if inputs.skill["skill"] != "reach_object":
        raise ValueError("Checkpoint 5.2 comparison accepts reach only")
    runs, identities = [], set()
    for directory in run_dirs:
        _verify_run(directory)
        checkpoint = load_checkpoint(directory / "selected.pt")
        ReachPolicy(checkpoint, inputs)
        selection = json.loads((directory / "selection.json").read_text())
        attempt = json.loads((directory / "test_state_attempt.json").read_text())
        report_dir = Path(attempt["output_dir"])
        _verify_run(report_dir)
        report = json.loads((report_dir / "report.json").read_text())
        if report["checkpoint_sha256"] != digest_file(directory / "selected.pt") or report["split"] != "test":
            raise ValueError("Test report/checkpoint mismatch")
        identity = (report["model"], report["seed"])
        if identity in identities:
            raise ValueError("Duplicate model/seed comparison")
        identities.add(identity)
        last = load_checkpoint(directory / "last.pt")
        runs.append({"model": report["model"], "seed": report["seed"], "selected_epoch": selection["epoch"],
                     "run_dir": str(directory), "checkpoint_sha256": report["checkpoint_sha256"],
                     "test_report_sha256": digest_file(report_dir / "report.json"),
                     "test_report_path": str(report_dir / "report.json"),
                     "training_history": last["training_state"]["history"],
                     "validation_candidates": [{k: r[k] for k in ("epoch", "success_rate", "worst_cell_success_rate", "normalized_terminal_error", "safety_violation_count")} for r in selection["reports"]],
                     "test": {k: report[k] for k in ("rollout_count", "unique_reset_count", "success_rate", "worst_family_success_rate", "mean_action_jerk", "normalized_terminal_error", "safety_counters", "terminal_reasons", "gates", "status", "state_numerical_gates_passed", "manual_acceptance", "breakdowns")}})
    expected = {(model, seed) for model in ("baseline", "reference") for seed in inputs.plan["training"]["seeds"]}
    if identities != expected:
        raise ValueError("Comparison requires both frozen models and all three seeds")
    legacy = level3_compatibility(legacy_checkpoint, inputs)
    result = {"version": "level5/reach-comparison-v1", "input_digests": inputs.digests, "runs": runs,
              "legacy": legacy, "no_test_selection_or_retuning": True,
              "state_only_default_status": "experimental_or_failed; no perception qualification",
              "physical_metric_caveat": "Frozen reach approach metrics use WorldState.robot.base_position, the mocap target pose; policy proprioception uses measured free-joint pose. Scene disturbance uses physical object positions.",
              "counter_semantics": "Raw action-bound violating values, matching Level 3; plus object workspace events. Joint counters are actuator target limits, not soft-joint qpos overshoot.",
              "render_condition_semantics": "All four independently executed state attempts share identical physics and state inputs; no RGB or perception claim."}
    output_dir.mkdir(parents=True, exist_ok=False)
    _json(output_dir / "comparison.json", result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for run in runs:
        axis = axes[0 if run["model"] == "baseline" else 1, inputs.plan["training"]["seeds"].index(run["seed"])]
        history = run["training_history"]
        axis.plot([r["epoch"] for r in history], [r["training_loss"] for r in history], label="Training")
        axis.plot([r["epoch"] for r in history], [r["validation_offline_loss"] for r in history], label="Validation")
        axis.axvline(run["selected_epoch"], color="gray", linestyle="--", label="Selected epoch")
        axis.set(title=f"{run['model']} / {run['seed']}\nHeld-out success {run['test']['success_rate']:.1%}", xlabel="Epoch", ylabel="Offline loss")
        axis.legend(fontsize=7)
    figure.savefig(output_dir / "training_curves.svg")
    plt.close(figure)
    _checksum_index(output_dir)
    return result
