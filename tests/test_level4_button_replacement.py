"""A button amendment cannot relax ownership, provenance or immutable-source gates."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dexvision.apps.record_demo import build_parser
from dexvision.evaluation.split_audit import content_digest, file_inventory
from dexvision.logging.button_amendment import (
    apply_button_replacements, collect_button_replacements, load_button_plan, recording_variation,
)
from dexvision.logging.level4_collection import sample_level4_procedural_variation
from dexvision.logging.visual_stream import digest_file


@pytest.fixture
def amendment(tmp_path):
    config = Path("configs/level4_dataset.yaml")
    requirements = yaml.safe_load(config.read_text())
    parent = tmp_path / "parent.json"
    receipt_path = tmp_path / "receipt.json"
    plan = {"version": "level4/button-replacement-plan-v1", "operator_id": "expert",
            "receipt_path": str(receipt_path), "assignments": []}
    episodes, sessions = [], {}
    receipt = {"version": "level4/button-replacement-receipt-v1", "episodes": {}}
    for i in range(8):
        split = "validation" if i < 4 else "test"
        cell = next(c for c in requirements["coverage_cells"]
                    if c["data_group"] == "button" and c["split_owner"] == split)
        variation = dict(sample_level4_procedural_variation(
            requirements["level4_5b_procedural_expansion"], skill_name="press_button", seed=i))
        pair = []
        for prefix in ("old", "new"):
            eid = f"{prefix}_{i}"
            path = tmp_path / eid
            path.mkdir()
            (path / "actions.npy").write_bytes(eid.encode())
            episode = SimpleNamespace(episode_id=eid, session_id=eid, path=path,
                source="scripted", skill_name="press_button", goal_condition_id=cell["id"],
                expert_accepted=True, metadata={"random_seed": i, "operator_id": "expert",
                "task_config": {"procedural_variation": variation}})
            episodes.append(episode)
            pair.append(episode)
            sessions[eid] = SimpleNamespace(split=split)
        old, new = pair
        plan["assignments"].append({"original_episode_id": old.episode_id,
            "original_episode_digest": content_digest(file_inventory(old.path)),
            "replacement_episode_id": new.episode_id, "recording_session_id": new.session_id,
            "cell_id": cell["id"], "split": split, "seed": i, "variation": variation})
        receipt["episodes"][new.episode_id] = {"episode_digest": content_digest(file_inventory(new.path))}
    parent.write_text(json.dumps({"splits": {"cross_split_action_groups": [{"episodes": [
        {"episode_id": r["original_episode_id"], "split": r["split"]} for r in plan["assignments"]]}]}}))
    for key, path in (("dataset_config", config), ("workcell_config", Path("configs/workcell.yaml")),
                      ("parent_report", parent)):
        plan[key] = str(path)
        plan[key + "_sha256"] = digest_file(path)
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    receipt["plan_sha256"] = digest_file(path)
    receipt_path.write_text(json.dumps(receipt))
    return path, plan, episodes, sessions


def test_eight_replacements_preserve_sources_and_explicit_lineage(amendment):
    path, plan, episodes, sessions = amendment
    before = [file_inventory(e.path) for e in episodes]
    active, excluded, hashes = apply_button_replacements(episodes, plan_path=path, sessions=sessions)
    assert [e.episode_id for e in active] == [r["replacement_episode_id"] for r in plan["assignments"]]
    assert len(excluded) == 8
    assert [r["episode_digest"] for r in excluded] == [r["original_episode_digest"] for r in plan["assignments"]]
    assert hashes[str(path)] == digest_file(path)
    assert before == [file_inventory(e.path) for e in episodes]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "old_bytes", "new_bytes",
                                      "split", "seed", "variation", "receipt", "plan"])
def test_corrupt_amendment_fails_closed(amendment, mutation):
    path, plan, episodes, sessions = amendment
    if mutation == "missing":
        episodes = episodes[1:]
    elif mutation == "duplicate":
        episodes.append(episodes[1])
    elif mutation.endswith("_bytes"):
        (episodes[0 if mutation == "old_bytes" else 1].path / "actions.npy").write_bytes(b"changed")
    elif mutation == "split":
        sessions[episodes[1].session_id].split = "train"
    elif mutation == "seed":
        episodes[1].metadata["random_seed"] = 999
    elif mutation == "variation":
        episodes[1].metadata["task_config"]["procedural_variation"] = {}
    elif mutation == "receipt":
        receipt_path = Path(plan["receipt_path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["episodes"].pop("new_0")
        receipt_path.write_text(json.dumps(receipt))
    else:
        path.write_text(path.read_text() + "# altered\n")
    with pytest.raises(ValueError):
        apply_button_replacements(episodes, plan_path=path, sessions=sessions)


@pytest.mark.parametrize("mutation", ["duplicate_seed", "duplicate_id", "training", "sampler", "parent"])
def test_plan_rejects_unfrozen_or_out_of_scope_assignments(amendment, mutation):
    path, plan, _, _ = amendment
    if mutation == "duplicate_seed":
        plan["assignments"][1]["seed"] = 0
    elif mutation == "duplicate_id":
        plan["assignments"][1]["replacement_episode_id"] = "new_0"
    elif mutation == "training":
        plan["assignments"][0]["split"] = "train"
    elif mutation == "sampler":
        plan["assignments"][0]["variation"]["goal_position_offset_m"] = [0, 0, 0]
    else:
        Path(plan["parent_report"]).write_text("{}")
    path.write_text(yaml.safe_dump(plan))
    with pytest.raises(ValueError):
        load_button_plan(path)


def test_recording_recipe_is_enforced_before_appending_a_session(amendment):
    path, plan, _, _ = amendment
    row = plan["assignments"][0]
    args = build_parser().parse_args([
        "--task", "level4_workcell", "--skill", "press_button", "--source", "scripted",
        "--episode-id", row["replacement_episode_id"], "--task-seed", str(row["seed"]),
        "--session-id", row["recording_session_id"], "--session-split", row["split"],
        "--goal-condition-id", row["cell_id"], "--operator-id", plan["operator_id"],
        "--enforce-frozen-cell-owner"])
    assert recording_variation(args, path) == row["variation"]
    args.task_seed += 1
    with pytest.raises(ValueError, match="Recording arguments"):
        recording_variation(args, path)


def test_existing_evidence_cannot_be_overwritten(amendment, tmp_path):
    path, _, _, _ = amendment
    before = file_inventory(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        collect_button_replacements(path, tmp_path / "missing_data")
    assert file_inventory(tmp_path) == before
