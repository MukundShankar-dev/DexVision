"""Rejections are retained as failed requests, never qualified control rollouts."""

import json
from pathlib import Path

import pytest

from dexvision.evaluation.perception_rollouts import (
    SKILLS, VERSION, load_protocol, request_matrix, review_rejections, summarize_requests,
)
from dexvision.logging.visual_stream import digest_file


def bindings():
    return {skill: {"reason": "unsupported measured state", "control_cadence_s": .034,
                    "runs": [{"training_seed": seed, "checkpoint": f"{skill}/{seed}/selected.pt",
                              "sha256": f"hash-{skill}-{seed}"} for seed in (5001, 5002, 5003)]}
            for skill in SKILLS}


def spec():
    return {"version": VERSION, "checkpoint": "5.7", "role": "validation_observability_preflight",
            "training_seeds": [5001, 5002, 5003], "minimum_confidence": .8,
            "maximum_age_s": .1, "scene_count": 30, "scene_seed_start": 57071000,
            "input_digests": {}}


def test_matrix_keeps_every_seed_and_shared_reset_identity_before_observation():
    rows = request_matrix(spec(), bindings())
    assert len(rows) == 450 and len({r["request_id"] for r in rows}) == 450
    assert len({r["reset_seed"] for r in rows}) == 30
    for skill in SKILLS:
        for seed in (5001, 5002, 5003):
            group = [r for r in rows if r["skill"] == skill and r["training_seed"] == seed]
            assert len(group) == 30
            assert {r["checkpoint_sha256"] for r in group} == {f"hash-{skill}-{seed}"}
            assert all(r["entry"] == "neutral_scene_before_task_entry" for r in group)


def test_rejections_cannot_be_relabelled_as_rollout_success_safety_or_paired_delta():
    rows = [{**r, "terminal_result": "rejected_before_execution", "inference_boundary_latency_s": .05}
            for r in request_matrix(spec(), bindings())]
    report = summarize_requests(rows, bindings(), spec())
    assert report["software_matrix_complete"] and not report["qualification_pass"]
    assert not report["checkpoint_complete"] and report["unique_scene_resets_total"] == 30
    for skill in report["skills"].values():
        assert skill["requests"] == skill["rejected_before_execution"] == 90
        assert skill["qualified_rollouts"] == skill["paired_state_rollouts"] == 0
        assert skill["success_rate"] is skill["success_rate_delta"] is skill["safety_gate_pass"] is None
        assert skill["status"] == "failed" and skill["cadence_deadline_misses"] == 90
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_requests(rows[:-1], bindings(), spec())
    rows[0]["terminal_result"] = "succeeded"
    with pytest.raises(ValueError, match="unsupported control"):
        summarize_requests(rows, bindings(), spec())


def write_protocol(tmp_path, value):
    import yaml
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(value))
    path.with_suffix(".sha256").write_text(digest_file(path) + "  protocol.yaml\n")
    return path


def test_protocol_locks_limits_role_and_input_bytes(tmp_path):
    payload = tmp_path / "input.json"
    payload.write_text("{}")
    value = {**spec(), "input_digests": {str(payload): digest_file(payload)}}
    path = write_protocol(tmp_path, value)
    assert load_protocol(path)["scene_count"] == 30
    payload.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Frozen 5.7 input"):
        load_protocol(path)
    path = write_protocol(tmp_path, {**spec(), "role": "test"})
    with pytest.raises(ValueError, match="Unsupported"):
        load_protocol(path)
    path.write_text("version: different")
    with pytest.raises(ValueError, match="checksum"):
        load_protocol(path)


def test_overwrite_refusal_precedes_model_or_simulator_loading(tmp_path, monkeypatch):
    from dexvision.evaluation.perception_rollouts import evaluate_perception_requests
    monkeypatch.setattr("dexvision.evaluation.perception_rollouts.load_protocol", lambda _: spec())
    with pytest.raises(ValueError, match="preserved"):
        evaluate_perception_requests("unused", tmp_path)


def test_review_requires_integrity_and_never_writes_acceptance(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    panels = []
    for skill in SKILLS:
        name = f"{skill}.png"
        assert cv2.imwrite(str(tmp_path / name), np.zeros((20, 20, 3), np.uint8))
        panels.append({"skill": skill, "panel": name})
    (tmp_path / "report.json").write_text(json.dumps({"qualification_pass": False}))
    (tmp_path / "review.json").write_text(json.dumps(panels))
    files = sorted(tmp_path.iterdir())
    index = tmp_path / "SHA256SUMS"
    index.write_text("".join(f"{digest_file(p)}  {p.name}\n" for p in files))
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda title, image: shown.append(image))
    monkeypatch.setattr(cv2, "waitKey", lambda _: 32)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    result = review_rejections(tmp_path)
    assert len(shown) == 5 and result["manual_acceptance"] == "pending_owner_confirmation"
    assert not result["checkpoint_complete"]
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir()}
    original = index.read_text()
    index.write_text("\n".join(original.splitlines()[1:]) + "\n")
    with pytest.raises(ValueError, match="Incomplete"):
        review_rejections(tmp_path)
    index.write_text(original)
    Path(tmp_path / panels[0]["panel"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        review_rejections(tmp_path)


def test_cli_routes_only_explicit_all_required_perception_requests(tmp_path, monkeypatch):
    from dexvision.apps.evaluate_skill import main
    path = tmp_path / "config.yaml"
    path.write_text(f"version: {VERSION}\n")
    calls = []
    monkeypatch.setattr("dexvision.evaluation.perception_rollouts.run_perception_evaluation",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {"checkpoint_complete": False})
    assert main(["--config", str(path), "--all-required-skills", "--track", "perception",
                 "--output-dir", str(tmp_path / "new")]) == 0
    assert len(calls) == 1
    assert main(["--config", str(path), "--all-required-skills", "--track", "state"]) == 2
    assert main(["--config", str(path), "--all-required-skills", "--track", "perception", "--split", "test"]) == 2
    assert len(calls) == 1


def test_state_cli_does_not_receive_perception_only_options(tmp_path, monkeypatch):
    from dexvision.apps.evaluate_skill import main

    path = tmp_path / "state.yaml"
    path.write_text("version: level5/reach-test\n")
    calls = []

    def original_signature(config, *, checkpoint, output_dir, release_root, track,
                           split, select_candidates, viewer, scenario_set):
        calls.append((config, track, split))
        return {"state_dispatch": "preserved"}

    monkeypatch.setattr("dexvision.evaluation.evaluate_skill.run_evaluation_command", original_signature)
    assert main(["--config", str(path), "--track", "state"]) == 0
    assert calls == [(path, "state", "test")]
    assert main(["--config", str(path), "--all-required-skills", "--track", "state"]) == 2
    assert len(calls) == 1
