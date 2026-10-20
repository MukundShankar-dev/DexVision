"""Paired 5.7 rollouts with explicit optical/sensor source amendment.

Accepted policy classes and rollout function bytecode are retained. A private
function-global mapping replaces only the declared observation, precondition
and place-preview inputs; no module globals or frozen source files are patched.
Truth remains the independent evaluation/termination oracle, never a feature,
phase, latch or preview input. Failed attempts remain in the matrix and reports.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
import tempfile
import time
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from dexvision.evaluation.evaluate_skill import SAFETY_KEYS, apply_action, safe_action
from dexvision.evaluation.instrumented_observer import (
    InstrumentedObserver,
    ObservedPolicy,
    PerceptionUnavailable,
)
from dexvision.evaluation.perception_rollouts import SKILLS
from dexvision.learning.train_skill import _json
from dexvision.learning.training_manifest import (
    ReleaseSource,
    content_digest,
    digest_file,
)
from dexvision.logging.level4_collection import (
    WorkcellPilotTask,
    sample_level4_procedural_variation,
)

VERSION = "level5/perception-rollouts-v2"


def storage_parent_digest(spec):
    """Permit a budget-only amendment while proving control/entry code unchanged."""
    amendment = spec["storage_amendment"]
    parent = Path(amendment["parent_recipe"])
    if digest_file(parent) != amendment["parent_sha256"]:
        raise ValueError("Storage amendment parent changed")
    original = yaml.safe_load(parent.read_text())
    ignored = {"maximum_saved_bytes", "implementation_sha256", "storage_amendment"}
    if (
        {k: v for k, v in spec.items() if k not in ignored}
        != {k: v for k, v in original.items() if k not in ignored}
        or "storage_amendment" in original
        or not original["maximum_saved_bytes"]
        < spec["maximum_saved_bytes"]
        <= 640 * 1024**2
    ):
        raise ValueError("Only the storage ceiling may change after validation")
    own = "dexvision/evaluation/instrumented_rollouts.py"
    if {k: v for k, v in spec["implementation_sha256"].items() if k != own} != {
        k: v for k, v in original["implementation_sha256"].items() if k != own
    }:
        raise ValueError("Storage amendment cannot change observation/policy inputs")
    old_source = parent.parents[2] / own
    if digest_file(old_source) != original["implementation_sha256"][own]:
        raise ValueError("Original integration snapshot changed")
    old_functions = {
        n.name: n
        for n in ast.parse(old_source.read_text()).body
        if isinstance(n, ast.FunctionDef)
    }
    new_functions = {
        n.name: n
        for n in ast.parse(Path(__file__).read_text()).body
        if isinstance(n, ast.FunctionDef)
    }
    for name, old in old_functions.items():
        if name == "load_instrumented_protocol":
            continue
        current = new_functions[name]
        if name == "run_instrumented_evaluation":
            current.body = [
                n
                for n in current.body
                if not (
                    isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == "entry_protocol"
                )
            ]

            class NormalizeProof(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == "entry_protocol":
                        return ast.parse(
                            'protocol["protocol_sha256"]', mode="eval"
                        ).body
                    return node

            current = NormalizeProof().visit(current)
        if ast.dump(old, include_attributes=False) != ast.dump(
            current, include_attributes=False
        ):
            raise ValueError(f"Storage amendment changed control or evaluation: {name}")
    return amendment["parent_sha256"]


def load_instrumented_protocol(path):
    path = Path(path)
    if (
        path.with_suffix(".sha256").read_text().strip()
        != f"{digest_file(path)}  {path.name}"
    ):
        raise ValueError("5.7 instrumented protocol checksum mismatch")
    spec = yaml.safe_load(path.read_text())
    if (
        spec["version"] != VERSION
        or spec["checkpoint"] != "5.7"
        or spec["training_seeds"] != [5001, 5002, 5003]
    ):
        raise ValueError("Unsupported 5.7 integration protocol")
    if (
        "development_case_limit" in spec
        or spec["minimum_confidence"] != 0.8
        or spec["maximum_age_s"] != 0.1
        or spec["camera_stride"] not in (1, 2, 3)
        or spec["bootstrap_frames"] != 3
        or spec["conditions"]
        != ["nominal", "mild_illumination", "partial_occlusion", "bounded_distractors"]
    ):
        raise ValueError("5.7 coverage/delivery contract cannot be changed silently")
    for name, sha in spec["implementation_sha256"].items():
        if digest_file(Path(name)) != sha:
            raise ValueError(f"Frozen 5.7 integration changed: {name}")
    if "storage_amendment" in spec:
        spec["_entry_protocol_sha256"] = storage_parent_digest(spec)
    spec["protocol_sha256"] = digest_file(path)
    return spec


def _load_runtime(skill):
    if skill == "reach":
        from dexvision.learning.reach_repair import load_repair_spec, RepairPolicy
        from dexvision.evaluation.evaluate_skill import run_reach_rollout

        spec, inputs = load_repair_spec("configs/level5/reach_repair_v5.yaml")
        return SimpleNamespace(
            spec=spec,
            inputs=inputs,
            policy=lambda p: RepairPolicy(p, spec, inputs),
            runner=run_reach_rollout,
        )
    if skill == "pick":
        from dexvision.learning.pick_policy import load_pick_spec, PickPolicy
        from dexvision.evaluation.pick_policy import run_pick_rollout

        spec, inputs = load_pick_spec("configs/level5/pick_policy_v7.yaml")
        return SimpleNamespace(
            spec=spec,
            inputs=inputs,
            policy=lambda p: PickPolicy(p, spec, inputs),
            runner=run_pick_rollout,
        )
    if skill == "push":
        from dexvision.learning.push_guarded import (
            load_push_spec,
            PushPolicy,
            runtime_spec,
        )
        from dexvision.evaluation.contact_policy import run_contact_rollout

        spec, inputs = load_push_spec("configs/level5/push_policy_v8.yaml")
        return SimpleNamespace(
            spec=spec,
            runtime_spec=runtime_spec(spec),
            inputs=inputs,
            policy=lambda p: PushPolicy(p, spec, inputs),
            runner=run_contact_rollout,
        )
    if skill == "press":
        from dexvision.learning.contact_policy import load_contact_spec, ContactPolicy
        from dexvision.evaluation.contact_policy import run_contact_rollout

        spec, inputs = load_contact_spec("configs/level5/press_policy_v1.yaml")
        return SimpleNamespace(
            spec=spec,
            runtime_spec=spec,
            inputs=inputs,
            policy=lambda p: ContactPolicy(p, spec, inputs),
            runner=run_contact_rollout,
        )
    from dexvision.learning.place_pad_release_policy import (
        load_pad_release_spec,
        PadReleasePlacePolicy,
    )
    from dexvision.evaluation.place_wrist_hold_rollout import run_wrist_hold_rollout
    from dexvision.evaluation.place_repair import load_place_teacher_spec

    config = "configs/level5/place_policy_pad_release_v25.yaml"
    _, spec, inputs = load_pad_release_spec(config)
    teacher = load_place_teacher_spec(spec["setup_teacher_recipe"])
    return SimpleNamespace(
        spec=spec,
        inputs=inputs,
        teacher=teacher,
        policy=lambda p: PadReleasePlacePolicy(p, config),
        runner=run_wrist_hold_rollout,
    )


def load_runtime(skill):
    from dexvision.sim.workcell import load_workcell_config

    runtime = _load_runtime(skill)
    config = load_workcell_config("configs/workcell.yaml")
    runtime.families = {o.object_id: o.family for o in config.objects}
    runtime.families.update({name: "button" for name in config.fixtures})
    runtime.families.update({name: "calibrated_target" for name in config.targets})
    return runtime


def checkpoint_paths(skill):
    receipt = json.loads(Path(f"docs/level5_{skill}_acceptance.json").read_text())
    paths = {}
    for run in receipt["runs"]:
        path = Path(run.get("checkpoint_path", run.get("selected_checkpoint")))
        expected = run.get("checkpoint_sha256", run.get("selected_checkpoint_sha256"))
        if digest_file(path) != expected:
            raise ValueError("Accepted selected weights changed")
        paths[run["seed"]] = path
    return paths


def case_plan(protocol, runtime, skill, split):
    if skill == "place":
        cells = sorted(
            c["id"]
            for c in runtime.inputs.requirements["coverage_cells"]
            if c["data_group"] == "pick_place"
        )
    else:
        parent = "validation" if split == "validation" else "test"
        cells = sorted(
            c["cell_id"] for c in runtime.inputs.skill["evaluation"][parent]["cells"]
        )
    repeats = 1 if split == "validation" else max(3, math.ceil(30 / len(cells)))
    first = protocol["seed_starts"][split] + SKILLS.index(skill) * 100000
    return [
        {
            "skill": skill,
            "cell_id": cell,
            "seed": first + i * 100 + j,
            "boundary": bool(j % 2),
            "episode_id": f"57_{split}_{skill}_{first + i * 100 + j}",
            "session": f"57_{split}_{skill}_{cell}",
        }
        for i, cell in enumerate(cells)
        for j in range(repeats)
    ]


def freeze_cases(protocol, runtime, skill, split, root):
    """Apply scripted pre-entry only, save compact prefixes, never filter failures."""
    root.mkdir(parents=True, exist_ok=False)
    plan = case_plan(protocol, runtime, skill, split)
    if protocol.get("development_case_limit"):
        plan = plan[: protocol["development_case_limit"]]
    _json(root / "planned_cases.json", plan)
    original_rows = json.loads(
        Path("datasets/level4-v1/splits/train.json").read_text()
    )["episodes"]
    first = next(r for r in original_rows if r["stream"] == "expert_success")
    member = f"data/demos/level4/{first['source_path']}/metadata.json"
    original = json.loads(
        ReleaseSource(runtime.inputs, runtime.inputs.root).read_many([member])[member]
    )
    neutral = np.array(original["initial_applied_action"], float)
    contact_fingers = {}
    if skill in {"push", "press"}:
        from dexvision.evaluation.contact_policy import (
            contact_scenarios,
            read_contact_entries,
        )

        sources = {}
        for scenario in contact_scenarios(runtime.inputs, "validation"):
            family = runtime.families[scenario[0]["entity_id"]]
            sources.setdefault(family, scenario)
        entries = read_contact_entries(
            runtime.inputs, list(sources.values()), runtime.inputs.root
        )
        for family, scenario in sources.items():
            entry = entries[scenario[0]["episode_id"]]
            if len(entry[1]):
                contact_fingers[family] = entry[1][-1, 7:].copy()
    records = []
    cells = {c["id"]: c for c in runtime.inputs.requirements["coverage_cells"]}
    for index, item in enumerate(plan):
        cell = cells[item["cell_id"]]
        pilot = (
            "pick_place_sequence"
            if skill in {"pick", "place"}
            else runtime.inputs.skill["skill"]
        )
        variation = sample_level4_procedural_variation(
            runtime.inputs.requirements["level4_5b_procedural_expansion"],
            skill_name=pilot,
            seed=item["seed"],
            boundary_case=item["boundary"],
        )
        prefix, setup = np.empty((0, 27)), {"reason": "ready"}
        with WorkcellPilotTask(
            workcell_config="configs/workcell.yaml",
            dataset_config="configs/level4_dataset.yaml",
            skill_name=pilot,
            goal_condition_id=item["cell_id"],
            seed=item["seed"],
            procedural_variation=variation,
        ) as task:
            initial = task.initial_world_state
            prior = neutral.copy()
            prior[:7] = [
                *initial.robot.base_position,
                *initial.robot.base_orientation_wxyz,
            ]
            metadata = {
                "episode_id": item["episode_id"],
                "recording_session_id": item["session"],
                "typed_goal": task.goal,
                "random_seed": item["seed"],
                "initial_online_phase": "approach",
                "initial_applied_action": prior.tolist(),
                "initial_commanded_action": prior.tolist(),
                "task_config": task.metadata_task_config(),
                "source": "57_fresh_scripted_preentry_only",
            }
            row = {
                **item,
                "variation": variation,
                "typed_goal": task.goal,
                "entity_id": cell.get(
                    "entity_id", cell.get("object_id", cell.get("button_id"))
                ),
                "recording_session_id": item["session"],
                "episode_digest": content_digest(metadata),
                "source": metadata["source"],
                "split": split,
                "render_condition": "nominal",
            }
            if skill == "pick":
                from dexvision.evaluation.pick_teacher import (
                    load_teacher_spec,
                    unassisted_pick_prefix,
                )

                teacher, _, _ = load_teacher_spec(runtime.spec["teacher_recipe"])
                prefix, setup = unassisted_pick_prefix(
                    task, teacher, runtime.spec, runtime.inputs
                )
            elif skill in {"push", "press"}:
                from dexvision.sim.level4_expert import (
                    DeterministicPushExpert,
                    DeterministicPushConfig,
                    DeterministicButtonPressExpert,
                    DeterministicButtonPressConfig,
                )
                from dexvision.sim.scripted_expert_settings import (
                    resolve_scripted_expert_settings,
                )

                names = [
                    f["name"].split("/")[-1]
                    for f in runtime.inputs.plan["action"]["named_layout"][7:]
                ]
                settings = resolve_scripted_expert_settings(
                    task, "scripted_push" if skill == "push" else "scripted_button"
                )
                fingers = contact_fingers.get(
                    runtime.families[row["entity_id"]], prior[7:]
                )
                expert = (
                    DeterministicPushExpert(
                        finger_targets=dict(zip(names, fingers, strict=True)),
                        config=DeterministicPushConfig.from_mapping(settings),
                    )
                    if skill == "push"
                    else DeterministicButtonPressExpert(
                        finger_targets=dict(zip(names, fingers, strict=True)),
                        config=DeterministicButtonPressConfig.from_mapping(settings),
                    )
                )
                expert.reset(task, initial)
                actions = []
                setup = {"reason": "setup_timeout"}
                for _ in range(500):
                    request, phase, done, error = expert.step(
                        task.workcell.get_world_state()
                    )
                    if error or done:
                        setup = {"reason": error or "setup_terminated"}
                        break
                    if phase == (
                        "push_contact" if skill == "push" else "fixture_contact"
                    ):
                        setup = {"reason": "ready"}
                        break
                    action, _, _, unsafe = safe_action(
                        request.as_array(), prior, runtime.inputs.plan
                    )
                    if unsafe:
                        setup = {"reason": unsafe}
                        break
                    apply_action(
                        task, action, runtime.inputs.plan["action"]["named_layout"]
                    )
                    task.step(
                        n_steps=runtime.inputs.plan["action"][
                            "simulation_steps_per_action"
                        ]
                    )
                    actions.append(action.copy())
                    prior = action
                prefix = np.asarray(actions).reshape(-1, 27)
                if skill == "push":
                    from dexvision.evaluation.push_settled import settled_prefix

                    if len(prefix):
                        prefix = settled_prefix(
                            prefix, runtime.spec["setup_hold_samples"]
                        )
        file = f"prefix_{index:03d}.npz"
        np.savez_compressed(root / file, actions=prefix)
        records.append(
            {"row": row, "metadata": metadata, "prefix": file, "setup": setup}
        )
        print(
            f"5.7 freeze {split}/{skill}: {index + 1}/{len(plan)} {setup.get('reason', 'pick_entry')}",
            flush=True,
        )
    _json(
        root / "cases.json",
        {"protocol_sha256": protocol["protocol_sha256"], "records": records},
    )
    return records


def clone_runner(function, replacements):
    """Same accepted bytecode, isolated declared producer substitutions only."""
    cloned = types.FunctionType(
        function.__code__,
        {**function.__globals__, **replacements},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    cloned.__kwdefaults__ = function.__kwdefaults__
    return cloned


class ObserverReference:
    def __init__(self, context):
        self.context = context

    def __getattr__(self, name):
        observer = self.context.get("observer")
        if observer is None:
            raise PerceptionUnavailable("Observation boundary not initialized")
        return getattr(observer, name)


def execute_attempt(
    protocol,
    runtime,
    skill,
    record,
    prefix,
    policy,
    output,
    *,
    track,
    condition,
    video=False,
):
    output.mkdir(parents=True, exist_ok=False)
    setup = record.get("setup", {})
    if setup.get("setup_failed") or (
        skill in {"push", "press"} and setup.get("reason") != "ready"
    ):
        result = {
            "track": track,
            "condition": condition,
            "reset_seed": record["row"]["seed"],
            "cell_id": record["row"]["cell_id"],
            "object_family": runtime.families[record["row"]["entity_id"]],
            "terminal_result": "rejected_before_execution",
            "terminal_reason": "scripted_entry_failed",
            "setup": setup,
            "executed_actions": 0,
            "safety_counters": dict.fromkeys(SAFETY_KEYS, 0),
            "mean_action_jerk": 0.0,
            "inference_latencies_s": [],
            "failure_attribution": "physics_setup",
            "checkpoint_sha256": digest_file(
                checkpoint_paths(skill)[policy_seed(policy)]
            ),
            "default_registry_admission": False,
        }
        np.savez_compressed(
            output / "trajectory.npz", applied_actions=np.empty((0, 27))
        )
        _json(output / "result.json", result)
        return result
    context = {
        "observer": None,
        "task": None,
        "actions": [],
        "proposals": [],
        "qpos": [],
        "qvel": [],
        "times": [],
        "safety_counts": dict.fromkeys(SAFETY_KEYS, 0),
    }
    reference = ObserverReference(context)
    observed_policy = (
        ObservedPolicy(policy, reference, skill) if track == "perception" else policy
    )
    original = runtime.runner
    replacements = {}

    def initialize(task):
        context["task"] = task
        if track == "perception":
            observer = InstrumentedObserver(
                task,
                runtime.inputs,
                record["row"],
                skill,
                protocol,
                condition=condition,
                video_path=output / "review.avi" if video else None,
            )
            context["observer"] = observer
        # A fixed predeclared prior-command hold gives independent image times.
        # Identical holds are applied to the paired state baseline.
        for _ in range(protocol.get("bootstrap_frames", 3)):
            if track == "perception":
                context["observer"].bootstrap()
            task.workcell.step(
                n_steps=runtime.inputs.plan["action"]["simulation_steps_per_action"]
            )
        context["entry_sha256"] = hashlib.sha256(
            task.env.data.qpos.tobytes()
            + task.env.data.qvel.tobytes()
            + np.float64(task.env.data.time).tobytes()
        ).hexdigest()
        np.savez_compressed(
            output / "entry.npz",
            qpos=task.env.data.qpos.copy(),
            qvel=task.env.data.qvel.copy(),
            timestamp=np.float64(task.env.data.time),
        )
        return task

    if skill == "place":
        base_setup = original.__globals__["_setup"]

        def setup(task, *args):
            result = base_setup(task, *args)
            if result[0]["success"]:
                initialize(task)
            return result

        replacements["_setup"] = setup
    else:
        factory_name = (
            "create_reach_task"
            if skill == "reach"
            else "create_pick_task"
            if skill == "pick"
            else "create_contact_task"
        )
        factory = original.__globals__[factory_name]
        replacements[factory_name] = lambda *args: initialize(factory(*args))

    def record_action(task, action, fields):
        if context["observer"]:
            observer = context["observer"]
            elapsed = time.perf_counter() - observer.latest_start
            inferred = observer.estimator.last_world.require_entity(
                record["row"]["entity_id"]
            )
            if elapsed + float(task.env.data.time) - inferred.timestamp > 0.1:
                raise PerceptionUnavailable(
                    "Frame became stale before action application"
                )
            current = observer.records[-1]
            current["action_ready_latency_s"] = elapsed
            # Evaluation oracle only, AFTER the inferred features and proposed
            # action exist. These errors never change observations or actions.
            inferred = observer.estimator.last_world.require_entity(
                record["row"]["entity_id"]
            )
            body = task.env.model.body(record["row"]["entity_id"]).id
            current["position_error_m"] = float(
                np.linalg.norm(
                    np.array(inferred.position) - observer.scratch.xpos[body]
                )
            )
            a, b = np.array(inferred.orientation_wxyz), observer.scratch.xquat[body]
            current["orientation_error_rad"] = float(
                2 * np.arccos(np.clip(abs(a @ b), 0, 1))
            )
        context["qpos"].append(task.env.data.qpos.copy())
        context["qvel"].append(task.env.data.qvel.copy())
        context["times"].append(float(task.env.data.time))
        context["actions"].append(np.array(action).copy())
        apply_action(task, action, fields)
        if context["observer"] and (len(context["actions"]) - 1) % 3 == 0:
            context["observer"].record_frame(
                "applied xyz=" + np.array2string(np.asarray(action[:3]), precision=3)
            )

    replacements["apply_action"] = record_action

    def audited_safe_action(raw, prior, plan):
        context["proposals"].append(np.asarray(raw).copy())
        result = safe_action(raw, prior, plan)
        for name, value in result[2].items():
            context["safety_counts"][name] += value
        return result

    replacements["safe_action"] = audited_safe_action
    if track == "perception":
        replacements["reach_features"] = lambda *args, **kwargs: reference.features(
            *args, **kwargs
        )
        for check_name in (
            "pick_preconditions",
            "place_preconditions",
            "contact_preconditions",
            "check_preconditions",
        ):
            if check_name not in original.__globals__:
                continue
            checker = original.__globals__[check_name]

            def measured_checks(_truth, *args, checker=checker):
                world = reference.observe()
                checks = checker(world, *args)
                if "fresh_observation" in checks:
                    checks["fresh_observation"] = all(
                        e.confidence >= 0.8 and world.timestamp - e.timestamp <= 0.1
                        for e in world.entities
                    )
                return checks

            replacements[check_name] = measured_checks
        if skill == "place":
            accepted_preview = original.__globals__["preview_place_action"]

            def preview(_live_task, action, prior, fields, inputs, row, current_policy):
                estimated = reference.estimator.preview_task(
                    skill_name="place_held_object",
                    goal=reference.row["typed_goal"],
                    initial_world=reference.initial,
                )
                return accepted_preview(
                    estimated, action, prior, fields, inputs, row, current_policy
                )

            replacements["preview_place_action"] = preview
    runner = clone_runner(original, replacements)
    result = None
    try:
        with tempfile.TemporaryDirectory(prefix="dexvision57_") as temporary:
            verbose = Path(temporary) / "attempt"
            row = record["row"]
            if skill == "reach":
                result = runner(
                    runtime.inputs,
                    row,
                    record["metadata"],
                    observed_policy,
                    condition=condition,
                    output_dir=verbose,
                )
            elif skill == "place":
                policy.bind(row)
                result = runner(
                    runtime.spec,
                    runtime.inputs,
                    runtime.teacher,
                    runtime.teacher[2],
                    row,
                    observed_policy,
                    verbose,
                )
            else:
                if skill == "push":
                    policy.reset()
                result = runner(
                    runtime.inputs,
                    getattr(runtime, "runtime_spec", runtime.spec),
                    row,
                    record["metadata"],
                    prefix,
                    observed_policy,
                    condition=condition,
                    output_dir=verbose,
                )
            with np.load(verbose / "trajectory.npz", allow_pickle=False) as arrays:
                if skill == "pick":
                    from dexvision.evaluation.pick_repair import physical_pick_audit

                    result["physical_audit"] = physical_pick_audit(
                        context["task"],
                        {k: arrays[k] for k in arrays.files},
                        result,
                        runtime.spec,
                    )
                if skill == "place":
                    from dexvision.evaluation.place_full_source_selection import (
                        _audit_attempt,
                    )

                    result["physical_audit"] = _audit_attempt(
                        runtime.spec, runtime.inputs, row, result, verbose
                    )
                keep = {
                    k: arrays[k]
                    for k in (
                        "qpos",
                        "qvel",
                        "applied_actions",
                        "state_timestamps",
                        "raw_actions",
                        "prior_applied_actions",
                    )
                    if k in arrays
                }
                if skill == "push":
                    from dexvision.learning.push_guarded import audit_decoder_events

                    result["decoder_audit"] = audit_decoder_events(
                        policy.events, verbose / "trajectory.npz", runtime.spec
                    )
                    for key in ("raw_actions", "prior_applied_actions"):
                        keep[key] = arrays[key]
                    keep["unbounded_learned_proposals"] = np.asarray(
                        [e["unbounded_learned_proposal"] for e in policy.events]
                    )
                    keep["settle_holds"] = np.array(
                        [e["settle_hold"] for e in policy.events]
                    )
                    keep["entry_aliases"] = np.array(
                        [e["entry_phase_alias"] for e in policy.events]
                    )
                    keep["workspace_fractions"] = np.array(
                        [e["workspace_fraction"] for e in policy.events]
                    )
                np.savez_compressed(output / "trajectory.npz", **keep)
            if skill == "reach" and result["terminal_result"] == "succeeded":
                from dexvision.evaluation.reach_repair_review import (
                    measured_pose_evidence,
                )

                with np.load(output / "trajectory.npz", allow_pickle=False) as data:
                    address = int(
                        context["task"].env.model.joint("rh_base_freejoint").qposadr[0]
                    )
                    result["physical_pose_audit"] = measured_pose_evidence(
                        data["qpos"],
                        address,
                        record["row"]["typed_goal"]["approach_pose"],
                        policy.names,
                        runtime.spec,
                        policy.recipe,
                        runtime.inputs.skill["contract"]["success_metric"],
                    )
            result["verbose_staging_discarded_after_compact_trace_readback"] = True
    except PerceptionUnavailable as exc:
        from dexvision.evaluation.benchmark_retargeters import mean_action_jerk

        bounds = np.array(
            [
                f["upper"] - f["lower"]
                for f in runtime.inputs.plan["action"]["named_layout"]
            ]
        )
        jerk = (
            mean_action_jerk(np.asarray(context["actions"]) / bounds)
            if context["actions"]
            else 0.0
        )
        np.savez_compressed(
            output / "trajectory.npz",
            qpos=np.asarray([*context["qpos"], context["task"].env.data.qpos.copy()]),
            qvel=np.asarray([*context["qvel"], context["task"].env.data.qvel.copy()]),
            applied_actions=np.asarray(context["actions"]).reshape(-1, 27),
            raw_actions=np.asarray(context["proposals"]).reshape(-1, 27),
            state_timestamps=np.asarray(
                [*context["times"], float(context["task"].env.data.time)]
            ),
        )
        result = {
            "terminal_result": "failed"
            if context["actions"]
            else "rejected_before_execution",
            "terminal_reason": "perception_unavailable: " + str(exc),
            "final_metrics": {},
            "safety_counters": context["safety_counts"],
            "mean_action_jerk": jerk,
            "evaluation_metrics_unavailable_reason": "perception rejection before original evaluator terminal reporting",
        }
    finally:
        if context["observer"]:
            observer = context["observer"]
            observer.refresh_scratch()
            observer.latest_rgb = observer.pixels()
            observer.observation_count = 0
            try:
                observer.observe()
            except PerceptionUnavailable:
                pass
            observer.record_frame(
                (result["terminal_result"] + ": " + result["terminal_reason"])
                if result
                else "ERROR incomplete attempt",
                force=True,
            )
            context["observer"].close()
    observer = context["observer"]
    if observer:
        records = observer.records
        np.savez_compressed(
            output / "observations.npz",
            timestamps=np.array([r["timestamp"] for r in records]),
            measurement_timestamps=np.array(
                [r["measurement_timestamp"] for r in records]
            ),
            consumed_pose=np.asarray([r["pose"] for r in records]).reshape(-1, 13),
            phases=np.array([r["phase"] for r in records]),
            latency_s=np.array([r["latency_s"] for r in records]),
            action_ready_latency_s=np.array(
                [
                    r.get("action_ready_latency_s")
                    if r.get("action_ready_latency_s") is not None
                    else np.nan
                    for r in records
                ]
            ),
            position_error_m=np.array(
                [
                    r.get("position_error_m")
                    if r.get("position_error_m") is not None
                    else np.nan
                    for r in records
                ]
            ),
            orientation_error_rad=np.array(
                [
                    r.get("orientation_error_rad")
                    if r.get("orientation_error_rad") is not None
                    else np.nan
                    for r in records
                ]
            ),
            feature_sha256=np.array(
                [
                    r.get("feature_sha256", "evaluation_only_final_frame")
                    for r in records
                ]
            ),
            features=np.asarray(
                [
                    r.get("features", np.full(len(observer.last_source), np.nan))
                    for r in records
                ]
            )
            if observer.last_source is not None
            else np.empty((0, 0)),
        )
    result.update(
        track=track,
        condition=condition,
        reset_seed=record["row"]["seed"],
        object_family=runtime.families[record["row"]["entity_id"]],
        cell_id=record["row"]["cell_id"],
        checkpoint_sha256=digest_file(checkpoint_paths(skill)[policy_seed(policy)]),
        executed_actions=len(context["actions"]),
        failure_attribution=(
            "perception"
            if result["terminal_reason"].startswith("perception_unavailable")
            else "policy_or_physics"
        ),
        inference_latencies_s=observer.latencies if observer else [],
        accepted_rollout_bytecode_sha256=hashlib.sha256(
            original.__code__.co_code
        ).hexdigest(),
        observation_source="visible_fiducial_rgb_plus_declared_simulated_sensors"
        if observer
        else "simulator_ground_truth",
        estimated_preview=bool(observer and skill == "place"),
        default_registry_admission=False,
        entry_state_sha256=context.get("entry_sha256"),
        execution_id=f"{skill}/{policy_seed(policy)}/{record['row']['seed']}/{track}/{condition}",
        retry_count=0,
        request_id=record["row"]["episode_id"],
    )
    result["failure_attribution"] = (
        "perception"
        if result["terminal_reason"].startswith("perception_unavailable")
        else "timeout"
        if result["terminal_result"] == "timed_out"
        else "supervisor"
        if any(result["safety_counters"].values())
        else "none"
        if result["terminal_result"] == "succeeded"
        else "policy_or_physics_unresolved"
    )
    result["target_type"] = runtime.inputs.skill.get("target_type_mapping", {}).get(
        result.get("target_id"), result.get("target_type", "approach_or_fixture")
    )
    _json(output / "result.json", result)
    return result


def policy_seed(policy):
    # Caller tags the strict verified policy object; this value never enters a
    # neural feature, decoder or observation.
    return policy._evaluation_seed


def summarize_paired(pairs, runtime, *, include_conditions=True):
    rates = {}

    def physical_success(row):
        return (
            row["terminal_result"] == "succeeded"
            and row.get("physical_pose_audit", {}).get(
                "measured_pose_and_dwell_passed", True
            )
            and row.get("physical_audit", {}).get("passed", True)
            and row.get("decoder_audit", {}).get("passed", True)
        )

    for track in ("state", "perception"):
        rows = [p[track] for p in pairs]
        rate = sum(physical_success(r) for r in rows) / len(rows)
        families, cells, goals = {}, {}, {}
        for row in rows:
            families.setdefault(row["object_family"], []).append(row)
            cells.setdefault(row["cell_id"], []).append(row)
            goals.setdefault(row.get("target_type", "approach_or_fixture"), []).append(
                row
            )
        worst = min(
            sum(physical_success(r) for r in g) / len(g)
            for g in [*families.values(), *goals.values()]
        )
        worst_cell = min(
            sum(physical_success(r) for r in g) / len(g) for g in cells.values()
        )
        wrong_contacts = sum(
            r.get("wrong_contact_count", int(r["terminal_reason"] == "wrong_contact"))
            for r in rows
        )
        wrong_buttons = sum(r.get("wrong_button_activation_count", 0) for r in rows)
        overshoots = sum(r["terminal_reason"] == "fixture_overshoot" for r in rows)
        physical_failures = sum(
            not r.get("physical_audit", {}).get("passed", True)
            or not r.get("physical_pose_audit", {}).get(
                "measured_pose_and_dwell_passed", True
            )
            or not r.get("decoder_audit", {}).get("passed", True)
            for r in rows
        )
        safety = (
            sum(sum(r["safety_counters"].values()) for r in rows)
            + wrong_contacts
            + wrong_buttons
            + overshoots
        )
        latencies = [v for r in rows for v in r["inference_latencies_s"]]
        rates[track] = {
            "attempts": len(rows),
            "unique_reset_count": len({r["reset_seed"] for r in rows}),
            "checkpoint_sha256": sorted({r["checkpoint_sha256"] for r in rows}),
            "family_breakdown": {
                name: {
                    "attempts": len(group),
                    "successes": sum(physical_success(r) for r in group),
                }
                for name, group in sorted(families.items())
            },
            "cell_breakdown": {
                name: {
                    "attempts": len(group),
                    "successes": sum(physical_success(r) for r in group),
                }
                for name, group in sorted(cells.items())
            },
            "goal_family_breakdown": {
                name: {
                    "attempts": len(group),
                    "successes": sum(physical_success(r) for r in group),
                }
                for name, group in sorted(goals.items())
            },
            "failure_attribution": dict(
                Counter(r["failure_attribution"] for r in rows)
            ),
            "success_rate": rate,
            "original_terminal_success_rate": sum(
                r["terminal_result"] == "succeeded" for r in rows
            )
            / len(rows),
            "worst_family_success_rate": worst,
            "worst_cell_success_rate": worst_cell,
            "executed_actions": sum(r["executed_actions"] for r in rows),
            "safety_count": safety,
            "wrong_contact_count": wrong_contacts,
            "wrong_button_activation_count": wrong_buttons,
            "physical_audit_failures": physical_failures,
            "drop_rate": sum(r.get("slip_drop", False) for r in rows) / len(rows),
            "premature_release_rate": sum(
                r.get("premature_release", False) for r in rows
            )
            / len(rows),
            "settled_among_nominally_placed": (
                sum(
                    r.get("settled", False) and r.get("nominally_placed", False)
                    for r in rows
                )
                / max(1, sum(r.get("nominally_placed", False) for r in rows))
            ),
            "terminal_reasons": dict(Counter(r["terminal_reason"] for r in rows)),
            "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
            "cadence_deadline_misses": sum(t > 0.034 for t in latencies),
            "mean_action_jerk": float(np.mean([r["mean_action_jerk"] for r in rows])),
        }
    rates["success_rate_delta"] = (
        rates["perception"]["success_rate"] - rates["state"]["success_rate"]
    )
    gates = {
        "aggregate_success": rates["perception"]["success_rate"] >= 0.65,
        "worst_family_success": rates["perception"]["worst_family_success_rate"]
        >= 0.50,
        "degradation": rates["success_rate_delta"] >= -0.15,
        "safety": rates["perception"]["safety_count"] == 0,
        "action_jerk": rates["perception"]["mean_action_jerk"]
        <= runtime.inputs.skill["gates"]["mean_action_jerk_max"],
        "cadence": rates["perception"]["cadence_deadline_misses"] == 0,
        "physical_audits": rates["perception"]["physical_audit_failures"] == 0,
    }
    skill = runtime.inputs.skill["skill"]
    if skill == "pick_object":
        gates["drop_rate"] = rates["perception"]["drop_rate"] <= 0.1
    if skill == "place_held_object":
        gates["premature_release"] = (
            rates["perception"]["premature_release_rate"] <= 0.1
        )
        gates["settled_placement"] = (
            rates["perception"]["settled_among_nominally_placed"] >= 0.9
        )
    summary = {
        **rates,
        "gates": gates,
        "status": "failed"
        if rates["perception"]["safety_count"]
        or not rates["perception"]["executed_actions"]
        else "experimental",
        "numerical_pass": all(gates.values()),
        "default_registry_admission": False,
        "status_reason": "manual acceptance pending"
        if all(gates.values())
        else "failed gates: " + ",".join(k for k, v in gates.items() if not v),
    }
    if include_conditions:
        summary["conditions"] = {
            condition: summarize_paired(
                [p for p in pairs if p["perception"]["condition"] == condition],
                runtime,
                include_conditions=False,
            )
            for condition in sorted({p["perception"]["condition"] for p in pairs})
        }
    return summary


def run_instrumented_evaluation(config, output_dir, *, split="test"):
    protocol = load_instrumented_protocol(config)
    entry_protocol = protocol.get("_entry_protocol_sha256", protocol["protocol_sha256"])
    from dexvision.learning.train_skill import _verify_run

    frozen_root = Path(protocol["cases_root"])
    receipt = json.loads((frozen_root / "freeze_receipt.json").read_text())
    if receipt["protocol_sha256"] != entry_protocol or receipt[
        "index_sha256"
    ] != digest_file(frozen_root / "SHA256SUMS"):
        raise ValueError("Frozen entry matrix index changed")
    _verify_run(frozen_root)
    root = Path(output_dir)
    if root.exists():
        raise ValueError(
            "Preserve existing 5.7 outputs; a new path cannot retry a consumed test"
        )
    reservation = Path(protocol["run_root"]) / f"{split}_attempt.json"
    reservation.parent.mkdir(parents=True, exist_ok=True)
    if split == "test":
        ready = Path(protocol["run_root"]) / "validation_complete.json"
        if (
            not ready.exists()
            or json.loads(ready.read_text())["protocol_sha256"] != entry_protocol
        ):
            raise ValueError(
                "Run frozen validation and isolation checks before held-out evaluation"
            )
        validation = json.loads(ready.read_text())
        if digest_file(Path(validation["report_path"])) != validation["report_sha256"]:
            raise ValueError("Frozen validation report changed")
    with reservation.open("x") as handle:
        json.dump(
            {"protocol_sha256": protocol["protocol_sha256"], "output_dir": str(root)},
            handle,
        )
    root.mkdir(parents=True, exist_ok=False)
    _json(root / "protocol.json", protocol)
    runtimes = {skill: load_runtime(skill) for skill in SKILLS}
    all_records = {}
    case_root = Path(protocol["cases_root"]) / split
    for skill in runtimes:
        case_file = case_root / skill / "cases.json"
        frozen = json.loads(case_file.read_text())
        if frozen["protocol_sha256"] != entry_protocol:
            raise ValueError("Frozen entry matrix protocol changed")
        all_records[skill] = frozen["records"]
    reports = {}
    for skill, runtime in runtimes.items():
        reports[skill] = {}
        for seed, path in checkpoint_paths(skill).items():
            if split == "validation" and seed != 5001:
                continue
            policies = {
                track: runtime.policy(path) for track in ("state", "perception")
            }
            for policy in policies.values():
                policy._evaluation_seed = seed
            pairs = []
            for index, record in enumerate(all_records[skill]):
                with np.load(
                    case_root / skill / record["prefix"], allow_pickle=False
                ) as data:
                    prefix = data["actions"]
                parent = root / skill / str(seed) / f"{index:03d}"
                state = execute_attempt(
                    protocol,
                    runtime,
                    skill,
                    record,
                    prefix,
                    policies["state"],
                    parent / "state",
                    track="state",
                    condition="nominal",
                )
                conditions = protocol["conditions"]
                if split == "validation" and index == 0:
                    conditions = [
                        *conditions,
                        "pixel_bias",
                        "dropout",
                        "latency",
                        "sensor_dropout",
                    ]
                for condition in conditions:
                    current = execute_attempt(
                        protocol,
                        runtime,
                        skill,
                        record,
                        prefix,
                        policies["perception"],
                        parent / condition,
                        track="perception",
                        condition=condition,
                        video=seed == 5001
                        and len(
                            list(
                                (root / skill / str(seed)).glob(
                                    f"*/{condition}/review.avi"
                                )
                            )
                        )
                        < protocol["review_cases_per_skill"],
                    )
                    pairs.append({"state": state, "perception": current})
                    if state.get("entry_state_sha256") != current.get(
                        "entry_state_sha256"
                    ):
                        raise ValueError(
                            "Paired state/perception entry state drift; preserve failed evidence"
                        )
                print(
                    f"5.7 {split}/{skill}/{seed}: {index + 1}/{len(all_records[skill])}",
                    flush=True,
                )
                size = sum(
                    p.stat().st_size
                    for p in Path(protocol["run_root"]).rglob("*")
                    if p.is_file()
                )
                if size > protocol["maximum_saved_bytes"]:
                    raise ValueError(
                        "5.7 evidence storage cap reached; preserve current results"
                    )
            reports[skill][str(seed)] = summarize_paired(pairs, runtime)
    report = {
        "checkpoint": "5.7",
        "protocol_sha256": protocol["protocol_sha256"],
        "split": split,
        "reports": reports,
        "checkpoint_complete": False,
        "manual_acceptance": "pending_owner_review",
        "method": "instrumented_simulator_fiducials_and_explicit_sensors",
        "default_registry_admission": False,
        "state_baseline_reuse": "One independent unchanged state execution per reset/seed shared by four RGB-condition comparisons; not four independent state resets",
        "storage_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
    }
    _json(root / "report.json", report)
    from dexvision.evaluation.instrumented_review import index_review
    from dexvision.perception.perception_baseline import save_artifact_index

    report["review_clips"] = len(index_review(root))
    _json(root / "report.json", report)
    save_artifact_index(root)
    if split == "validation":
        _json(
            Path(protocol["run_root"]) / "validation_complete.json",
            {
                "protocol_sha256": protocol["protocol_sha256"],
                "report_sha256": digest_file(root / "report.json"),
                "report_path": (root / "report.json").as_posix(),
                "qualification_claim": False,
            },
        )
    return report


def freeze_instrumented_matrices(config):
    """Freeze both disjoint entry matrices before any RGB/control evaluation."""
    protocol = load_instrumented_protocol(config)
    root = Path(protocol["cases_root"])
    root.mkdir(parents=True, exist_ok=False)
    _json(root / "protocol.json", protocol)
    snapshot = Path(protocol["run_root"]) / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=False)
    for name, expected in protocol["implementation_sha256"].items():
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
        if digest_file(destination) != expected:
            raise ValueError(f"5.7 source snapshot changed while copying: {name}")
    for source in (Path(config), Path(config).with_suffix(".sha256")):
        destination = snapshot / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    runtimes = {skill: load_runtime(skill) for skill in SKILLS}
    summary = {}
    for split in ("validation", "test"):
        summary[split] = {}
        for skill, runtime in runtimes.items():
            records = freeze_cases(
                protocol, runtime, skill, split, root / split / skill
            )
            summary[split][skill] = {
                "unique_resets": len(records),
                "cells": len({r["row"]["cell_id"] for r in records}),
            }
    from dexvision.perception.perception_baseline import save_artifact_index

    _json(root / "summary.json", summary)
    save_artifact_index(root)
    _json(
        root / "freeze_receipt.json",
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "index_sha256": digest_file(root / "SHA256SUMS"),
        },
    )
    return summary
