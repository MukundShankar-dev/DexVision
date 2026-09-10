"""Adversarial split ownership, content manifests and normalization boundaries."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dexvision.evaluation.split_audit import (
    audit_splits, checked_path, content_digest, file_inventory, load_split_config,
    split_manifests, verify_files,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    requirements = yaml.safe_load((ROOT / "configs/level4_dataset.yaml").read_text())
    records, sessions = [], {}
    for i, split in enumerate(("train", "train", "validation", "test")):
        cell = next(c for c in requirements["coverage_cells"]
                    if c["split_owner"] == split and c["data_group"] == "reach")
        eid = f"episode_{i}"
        sessions[eid] = SimpleNamespace(recording_session_id=eid, split=split, operator_id="script")
        records.append({"episode_id": eid, "recording_session_id": eid, "split": split,
                        "operator_id": "script", "cell_id": cell["id"],
                        "entity_id": cell["entity_id"], "target_id": None,
                        "stream": "expert_success", "action_sha256": str(i),
                        "initial_state_digest": str(i), "source_episode_id": None,
                        "episode_digest": str(i), "frames": 5, "audit_passed": True,
                        "typed_goal": {"entity_id": cell["entity_id"]}, "schema_versions": {}})
    return requirements, records, sessions


def test_frozen_whole_sessions_pass_and_shared_training_pool_ids_are_permitted():
    requirements, records, sessions = fixture()
    assert audit_splits(records, sessions, requirements)["passed"]


@pytest.mark.parametrize("mutation,expected", [
    (lambda rows: rows.append(deepcopy(rows[0])), "Duplicate episode"),
    (lambda rows: rows[0].update(split="test"), "Session ownership"),
    (lambda rows: rows[0].update(entity_id="block_large"), "Held-out object"),
    (lambda rows: rows[0].update(target_id="return_bin_right"), "Held-out goal"),
    (lambda rows: rows[0].update(cell_id="unknown"), "Cell ownership"),
    (lambda rows: rows[3].update(action_sha256=rows[0]["action_sha256"]), "action_sha256"),
    (lambda rows: rows[3].update(initial_state_digest=rows[0]["initial_state_digest"]), "initial_state"),
    (lambda rows: rows[3].update(source_episode_id=rows[0]["episode_id"]), "source lineage"),
    (lambda rows: rows[0].update(source_episode_id="absent"), "source lineage"),
])
def test_leakage_and_missing_lineage_fail(mutation, expected):
    requirements, records, sessions = fixture()
    mutation(records)
    report = audit_splits(records, sessions, requirements)
    assert not report["passed"]
    assert any(expected in issue for issue in report["issues"])


def test_manifests_are_stable_and_normalization_never_uses_corrections_or_holdouts():
    _, records, _ = fixture()
    correction = {**records[0], "episode_id": "correction", "stream": "corrective_intervention"}
    failure = {**records[0], "episode_id": "failure", "stream": "ordinary_failure"}
    manifests = split_manifests([*records, correction, failure], [], "config")
    assert manifests == split_manifests([failure, correction, *records[::-1]], [], "config")
    assert [r["episode_id"] for r in manifests["train"]["normalization_inputs"]] == [
        "episode_0", "episode_1"]
    assert not manifests["test"]["normalization_inputs"]
    assert not manifests["validation"]["normalization_inputs"]
    changed = deepcopy(records)
    changed[0]["episode_digest"] = "tampered"
    assert manifests["train"]["manifest_digest"] != split_manifests(
        changed, [], "config")["train"]["manifest_digest"]


def test_checksums_detect_modified_and_missing_payloads(tmp_path):
    (tmp_path / "data.npy").write_bytes(b"original")
    hashes = file_inventory(tmp_path)
    assert verify_files(tmp_path, hashes) == []
    (tmp_path / "data.npy").write_bytes(b"changed")
    assert verify_files(tmp_path, hashes)
    (tmp_path / "data.npy").unlink()
    assert verify_files(tmp_path, hashes)
    assert content_digest({"a": 1, "b": 2}) == content_digest({"b": 2, "a": 1})


@pytest.mark.parametrize("path", ["../outside", "/absolute"])
def test_manifest_paths_cannot_escape_root(tmp_path, path):
    with pytest.raises(ValueError, match="Unsafe manifest"):
        checked_path(tmp_path, path)


def test_frozen_config_rejects_heldout_or_normalization_changes(tmp_path):
    requirements, _, _ = fixture()
    config_path = ROOT / "configs/level4_splits.yaml"
    load_split_config(config_path, requirements)
    config = yaml.safe_load(config_path.read_text())
    config["normalization_source"] = "all_splits"
    path = tmp_path / "splits.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="normalization_source"):
        load_split_config(path, requirements)


def test_failed_replay_cannot_become_normalization_input_or_release_manifest():
    _, records, _ = fixture()
    records[0]["audit_passed"] = False
    manifests = split_manifests(records, [], "config", audit_passed=False)
    for manifest in manifests.values():
        assert manifest["dataset_audit_passed"] is False
        assert manifest["status"] == "diagnostic_only_do_not_train_or_release"
    assert [r["episode_id"] for r in manifests["train"]["normalization_inputs"]] == ["episode_1"]
