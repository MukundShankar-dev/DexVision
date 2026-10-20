"""5.7 observability preflight and saved rejection evidence.

The accepted baseline blocks every control bundle. We run fresh *validation
preflight requests*, not task entry scripts or purported qualifying rollouts.
No state comparison, control delta or physical success is invented on rejection.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from dexvision.learning.perception_policy import SKILLS, PerceptionPolicy
from dexvision.logging.visual_stream import digest_file, write_json

VERSION = "level5/perception-rollouts-v1"


def run_perception_evaluation(config, *, output_dir=None, viewer=False, review_dir=None):
    """Legacy preflight dispatch, outside the checksum-frozen state evaluator."""
    if viewer:
        if review_dir is None or output_dir is not None:
            raise ValueError("Saved rejection review requires --viewer --review-dir, without --output-dir")
        return review_rejections(review_dir)
    if output_dir is None or review_dir is not None:
        raise ValueError("5.7 evaluation requires a new --output-dir")
    return evaluate_perception_requests(config, output_dir)


def load_protocol(config: str | Path) -> dict:
    """Verify the new protocol independently of all historical recipe locks."""
    path = Path(config)
    if digest_file(path) != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("Perception rollout protocol checksum mismatch.")
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (spec["version"] != VERSION or spec["checkpoint"] != "5.7"
            or spec["role"] != "validation_observability_preflight"
            or spec["training_seeds"] != [5001, 5002, 5003]
            or spec["minimum_confidence"] != .8 or spec["maximum_age_s"] != .1
            or spec["scene_count"] < 30):
        raise ValueError("Unsupported perception rollout protocol or delivery limits.")
    for name, expected in spec["input_digests"].items():
        if digest_file(Path(name)) != expected:
            raise ValueError(f"Frozen 5.7 input changed: {name}")
    spec["protocol_sha256"] = digest_file(path)
    return spec


def accepted_bindings(recipe: dict) -> dict:
    """Hash the exact selected weights and receipt-bound runtime sidecars.

    Loading a checkpoint for its schema is an audit, not execution. Receipt-bound
    recipes, overlays, runtime manifests and source pins are retained verbatim;
    no training/selection/test entry point is called or reservation consumed.
    """
    from dexvision.evaluation.perception_audit import accepted_inventory

    inventory = accepted_inventory(recipe)
    for skill in SKILLS:
        reference = recipe["accepted_bundles"][skill]
        receipt_path = Path(reference["receipt"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["manual_acceptance"] != "owner_confirmed":
            raise ValueError(f"Unaccepted state bundle: {skill}")
        runs = []
        for run in receipt["runs"]:
            checkpoint = Path(run.get("checkpoint_path", run.get("selected_checkpoint")))
            expected = run.get("checkpoint_sha256", run.get("selected_checkpoint_sha256"))
            pins = {checkpoint.as_posix(): expected}
            for path in sorted(checkpoint.parent.iterdir()):
                if path.suffix in {".json", ".yaml", ".sha256"}:
                    pins[path.as_posix()] = digest_file(path)
            # Bind every explicit receipt path/checksum pair, including overlay,
            # selection/physical evidence and the true execution recipe (place V25).
            for record in (receipt, run):
                for key, value in record.items():
                    if key.endswith("_path") and key[:-5] + "_sha256" in record:
                        pins[value] = record[key[:-5] + "_sha256"]
            for name, sha in pins.items():
                if digest_file(Path(name)) != sha:
                    raise ValueError(f"Accepted runtime/evidence changed: {name}")
            runs.append({"training_seed": run["seed"], "selected_epoch": run["selected_epoch"],
                         "checkpoint": checkpoint.as_posix(), "sha256": expected,
                         "artifact_digests": pins})
        if sorted(r["training_seed"] for r in runs) != [5001, 5002, 5003]:
            raise ValueError(f"Missing independent accepted seed for {skill}")
        inventory[skill].update(runs=runs, receipt_sha256=digest_file(receipt_path))
    return inventory


def request_matrix(spec: dict, bindings: dict) -> list[dict]:
    """Freeze shared reset ids *before* inference, without task/truth features.

    These are neutral-scene preflight requests; they are not held-object/contact
    entry states or independent per-seed resets. No held-out set is read/reused.
    """
    return [
        {"request_id": f"{skill}_{run['training_seed']}_{i:03d}", "skill": skill,
         "reset_seed": spec["scene_seed_start"] + i, "scene_index": i,
         "training_seed": run["training_seed"], "checkpoint": run["checkpoint"],
         "checkpoint_sha256": run["sha256"], "condition": "nominal",
         "role": spec["role"], "entry": "neutral_scene_before_task_entry"}
        for i in range(spec["scene_count"]) for skill in SKILLS for run in bindings[skill]["runs"]
    ]


def rejection_overlay(rgb, decision, baseline, skill):
    """Display visual ids/position hypotheses; never label them semantic targets."""
    panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for entity in decision.estimate.entities:
        field = entity.evidence.position
        if field.value is None:
            continue
        xy = baseline.pose.project(np.array([field.value]))[0]
        if not np.isfinite(xy).all():
            continue
        point = tuple(np.clip(np.rint(xy), -10000, 10000).astype(int))
        cv2.circle(panel, point, 4, (0, 200, 255), 2)
        cv2.putText(panel, entity.observation_id, point, cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 200, 255), 1)
    cv2.rectangle(panel, (0, 0), (640, 66), (0, 0, 0), -1)
    for i, text in enumerate((f"{skill}: REJECTED BEFORE EXECUTION", "Visual hypotheses only; semantic target unavailable",
                              "Policy calls = 0; actions = 0; no task entry or truth fallback")):
        cv2.putText(panel, text, (8, 18 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1)
    return panel


def summarize_requests(results: list[dict], bindings: dict, spec: dict) -> dict:
    skills = {}
    for skill in SKILLS:
        rows = [r for r in results if r["skill"] == skill]
        if len(rows) != spec["scene_count"] * 3 or any(r["terminal_result"] != "rejected_before_execution" for r in rows):
            raise ValueError("Incomplete rejection matrix or unsupported control execution.")
        latencies = [r["inference_boundary_latency_s"] for r in rows]
        skills[skill] = {
            "status": "failed", "reason": bindings[skill]["reason"],
            "requests": len(rows), "rejected_before_execution": len(rows),
            "unique_scene_resets": spec["scene_count"], "policy_calls": 0, "applied_actions": 0,
            "qualified_rollouts": 0, "paired_state_rollouts": 0,
            "success_rate": None, "worst_family_success_rate": None, "state_success_rate": None,
            "success_rate_delta": None, "safety_gate_pass": None,
            "metric_reason": "No task entry or control executed; preflight rejections cannot qualify rollouts.",
            "latency_median_s": float(np.median(latencies)), "latency_p95_s": float(np.percentile(latencies, 95)),
            "cadence_s": bindings[skill]["control_cadence_s"],
            "cadence_deadline_misses": sum(t > bindings[skill]["control_cadence_s"] for t in latencies),
            "latency_scope": "RGB inference plus typed boundary only; rendering, setup and I/O excluded",
            "failure_attribution": {"perception_observability": len(rows), "policy": 0,
                                    "supervisor": 0, "physics": 0, "timeout": 0},
        }
    return {"version": VERSION, "checkpoint": "5.7", "role": spec["role"], "skills": skills,
            "software_matrix_complete": True, "qualification_pass": False, "checkpoint_complete": False,
            "manual_acceptance": "pending_rejection_review_only",
            "held_out_rollout_review": "blocked_no_executable_accepted_inferred_track",
            "unique_scene_resets_total": spec["scene_count"],
            "paired_comparison": "blocked; accepted bundles bound but not executed; no historical rates substituted",
            "robustness_sweeps": "deferred; no supported control track; no test or threshold tuning",
            "default_registry_admission": False, "truth_fallback": False}


def evaluate_perception_requests(config, output_dir):
    """Run actual new RGB inference, retain all rejections, execute zero actions."""
    import mujoco

    from dexvision.evaluation.perception_guarded_body import selected_context
    from dexvision.perception.guarded_body_localization import GuardedBodyBaseline
    from dexvision.perception.perception_baseline import model_digest, save_artifact_index
    from dexvision.perception.render_annotations import camera_calibration, camera_rotation
    from dexvision.sim.workcell import Workcell

    spec = load_protocol(config)
    output = Path(output_dir)
    if output.exists():
        raise ValueError("Existing 5.7 output must be preserved; use saved review.")
    recipe, model = selected_context(spec["perception_recipe"])
    acceptance = json.loads(Path(spec["perception_acceptance"]).read_text(encoding="utf-8"))
    if (not acceptance["checkpoint_complete"] or model_digest(model) != acceptance["model_semantic_sha256"]
            or acceptance["all_five_accepted_inferred_control_tracks"] != "unsupported"):
        raise ValueError("Accepted 5.6 model/compatibility mismatch.")
    bindings = accepted_bindings(recipe)
    matrix = request_matrix(spec, bindings)
    # Fixed protocol-owned reservation prevents a new output path from silently
    # repeating a run. Failed/incomplete runs and their evidence remain intact.
    reservation = Path(spec["attempt_path"])
    reservation.parent.mkdir(parents=True, exist_ok=True)
    try:
        with reservation.open("x", encoding="utf-8") as handle:
            json.dump({"protocol_sha256": spec["protocol_sha256"], "output_dir": str(output),
                       "role": spec["role"]}, handle)
    except FileExistsError as exc:
        raise ValueError("5.7 preflight already reserved; changing output paths cannot retry it.") from exc
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", spec)
    write_json(output / "accepted_bundles.json", bindings)
    write_json(output / "matrix.json", matrix)
    baseline = GuardedBodyBaseline(recipe, model)
    visual = yaml.safe_load(Path(spec["visual_config"]).read_text(encoding="utf-8"))
    reference = json.loads(Path(spec["camera_calibration"]).read_text(encoding="utf-8"))
    results, review = [], []
    with Workcell(spec["workcell_config"]) as workcell:
        env = workcell.env
        camera = visual["camera"]
        cam = env.model.camera(camera["id"])
        cam.pos[:] = camera["position_m"]
        mujoco.mju_mat2Quat(cam.quat, camera_rotation(camera["position_m"], camera["lookat_m"]).reshape(9))
        cam.fovy[:] = camera["vertical_fov_deg"]
        for body_id in range(env.model.nbody):
            if env.model.body(body_id).name.startswith("pilot_"):
                env.model.geom_rgba[env.model.geom_bodyid == body_id, 3] = 0
        with mujoco.Renderer(env.model, height=480, width=640) as renderer:
            for i in range(spec["scene_count"]):
                workcell.reset(seed=spec["scene_seed_start"] + i)
                mujoco.mj_forward(env.model, env.data)
                actual = camera_calibration(env.model, env.data, camera["id"], 640, 480)
                actual["version"] = "level4/visual-camera-v1"
                if actual != reference:
                    raise ValueError("Fresh render does not match accepted pixel-center calibration.")
                renderer.update_scene(env.data, camera=camera["id"])
                rgb = renderer.render().copy()
                if not cv2.imwrite(str(output / f"scene_{i:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise ValueError("Failed to save fresh RGB scene.")
                for row in (r for r in matrix if r["scene_index"] == i):
                    adapter = PerceptionPolicy(baseline, row["skill"], incompatibility=bindings[row["skill"]]["reason"])
                    adapter.reset()
                    before = [a.copy() for a in (env.data.qpos, env.data.qvel, env.data.ctrl,
                                                env.data.mocap_pos, env.data.mocap_quat)]
                    timestamp = float(env.data.time)
                    started = time.perf_counter()
                    decision = adapter.observe(rgb, timestamp=timestamp, now=timestamp)
                    latency = time.perf_counter() - started
                    after = (env.data.qpos, env.data.qvel, env.data.ctrl, env.data.mocap_pos, env.data.mocap_quat)
                    if env.data.time != timestamp or any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True)):
                        raise ValueError("Rejected request mutated simulator/control state.")
                    record = {**row, **asdict(decision), "inference_boundary_latency_s": latency,
                              "rgb": f"scene_{i:03d}.png", "simulator_unchanged": True,
                              "policy_calls": 0, "applied_actions": 0}
                    write_json(output / f"{row['request_id']}.json", record)
                    results.append(record)
                    if i == 0 and row["training_seed"] == 5001:
                        path = f"review_{row['skill']}.png"
                        if not cv2.imwrite(str(output / path), rejection_overlay(rgb, decision, baseline, row["skill"])):
                            raise ValueError("Failed to write rejection review overlay.")
                        review.append({"skill": row["skill"], "panel": path, "request": row["request_id"] + ".json"})
                print(f"5.7 validation preflight scene {i+1}/{spec['scene_count']}: 15/15 rejected, zero actions", flush=True)
    report = summarize_requests(results, bindings, spec)
    report["protocol_sha256"] = spec["protocol_sha256"]
    write_json(output / "review.json", review)
    write_json(output / "report.json", report)
    save_artifact_index(output)
    return report


def review_rejections(directory):
    """Review saved rejection panels only; never records manual acceptance."""
    output = Path(directory)
    # Verify *all* saved artifacts before opening any review panel.
    indexed = set()
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        path = output / name
        if name in indexed or not path.resolve().is_relative_to(output.resolve()) or digest_file(path) != sha:
            raise ValueError("Rejection review artifact missing or changed.")
        indexed.add(name)
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob("*")
                    if p.is_file() and p != output / "SHA256SUMS"}
    if indexed != actual_files:
        raise ValueError("Incomplete rejection review artifact index.")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    panels = json.loads((output / "review.json").read_text(encoding="utf-8"))
    if report["qualification_pass"] or sorted(p["skill"] for p in panels) != sorted(SKILLS):
        raise ValueError("Expected five unsupported-track rejection panels.")
    try:
        for panel in panels:
            image = cv2.imread(str(output / panel["panel"]))
            if image is None:
                raise ValueError("Missing/unreadable rejection panel.")
            cv2.imshow("5.7 rejection review: any key next; Esc cancels", image)
            if cv2.waitKey(0) & 0xff == 27:
                raise ValueError("Manual rejection review cancelled.")
    except cv2.error as exc:
        raise ValueError(f"OpenCV review display unavailable: {exc}") from exc
    finally:
        cv2.destroyAllWindows()
    return {"reviewed_panels": len(panels), "manual_acceptance": "pending_owner_confirmation",
            "checkpoint_complete": False, "qualification_pass": False,
            "scope": "rejection evidence only; required held-out rollout review remains blocked"}
