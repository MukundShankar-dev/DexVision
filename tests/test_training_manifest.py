"""Frozen ownership, explicit streams and immutable-payload failure checks."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from dexvision.learning.training_manifest import (
    ReleaseSource, content_digest, load_training_inputs, select_streams, validate_split_ownership,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    return load_training_inputs("configs/level5/skills/reach_object.yaml")


def test_frozen_inputs_and_baseline_counts(inputs):
    assert inputs.root == ROOT
    selected = select_streams(inputs)
    assert set(selected) == {"expert_success"}
    assert {s: len(r) for s, r in selected["expert_success"].items()} == {
        "train": 80, "validation": 32, "test": 48}
    assert len(inputs.digests["dataset_sha256"]) == 64


@pytest.mark.parametrize("failure", ["session", "episode", "cell", "object", "goal", "lineage", "image", "normalization", "schema"])
def test_split_leakage_fails_closed(inputs, failure):
    splits = copy.deepcopy(inputs.splits)
    train, val = splits["train"]["episodes"], splits["validation"]["episodes"]
    if failure == "session":
        val[0]["recording_session_id"] = train[0]["recording_session_id"]
    elif failure == "episode":
        val[0]["episode_id"] = train[0]["episode_id"]
    elif failure == "cell":
        val[0]["cell_id"] = train[0]["cell_id"]
    elif failure == "object":
        train[0]["entity_id"] = "block_large"
    elif failure == "goal":
        train[0]["target_id"] = "return_bin_right"
    elif failure == "lineage":
        train[0]["source_episode_id"] = val[0]["episode_id"]
    elif failure == "image":
        splits["validation"]["images"][0]["rgb_sha256"] = splits["train"]["images"][0]["rgb_sha256"]
    elif failure == "normalization":
        splits["validation"]["normalization_inputs"] = splits["train"]["normalization_inputs"][:1]
    else:
        train[0]["observation_schema"]["layouts"]["robot_qpos"]["names"][0] = "invented"
    for doc in splits.values():
        doc["manifest_digest"] = content_digest({k: v for k, v in doc.items() if k != "manifest_digest"})
    with pytest.raises(ValueError):
        validate_split_ownership(splits, inputs.requirements, inputs.schemas)


def test_optional_streams_remain_distinguishable(inputs):
    streams = select_streams(inputs, ("expert_success", "ordinary_failure", "corrective_intervention", "visual", "legacy"))
    assert sum(len(rows) for rows in streams["ordinary_failure"].values()) == 90
    assert sum(len(rows) for rows in streams["corrective_intervention"].values()) == 30
    assert sum(len(rows) for rows in streams["visual"].values()) == 2633
    assert streams["legacy"]["recording_session_ids_available"] is False
    assert streams["legacy"]["baseline_target_count"] == 0
    assert all(r["training_target_interval"] is None for r in streams["ordinary_failure"]["train"])
    assert all(r["training_target_interval"][0] > 0 for r in streams["corrective_intervention"]["train"])
    with pytest.raises(ValueError, match="distinct streams"):
        select_streams(inputs, ("policy_rollout",))


def test_payload_paths_and_checksums(inputs, tmp_path):
    payload = tmp_path / "example.npy"
    payload.write_bytes(b"changed")
    release = {**inputs.release, "files": {"example.npy": {"size_bytes": 7, "sha256": "0" * 64}}}
    source = ReleaseSource(replace(inputs, release=release), tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        source.read_many(["example.npy"])
    with pytest.raises(ValueError, match="unlisted"):
        source.read_many(["../escape"])
    with pytest.raises(ValueError, match="Duplicate"):
        source.read_many(["example.npy", "example.npy"])


def test_changed_protocol_is_rejected(inputs, tmp_path):
    import shutil

    shutil.copytree(ROOT / "configs/level5", tmp_path / "configs/level5")
    (tmp_path / "docs").mkdir()
    shutil.copy(ROOT / "docs/level5_learning_plan.md", tmp_path / "docs/level5_learning_plan.md")
    path = tmp_path / "configs/level5/skills/reach_object.yaml"
    path.write_text(path.read_text().replace("state_success_min: 0.8", "state_success_min: 0.1"))
    with pytest.raises(ValueError, match="protocol checksum"):
        load_training_inputs("configs/level5/skills/reach_object.yaml", root=tmp_path)


def test_legacy_reader_keeps_original_missing_session(inputs, tmp_path):
    import io
    import json
    import tarfile

    import numpy as np

    from dexvision.learning.training_manifest import digest_file

    directory = "data/demos/raw/reach_touch_target/episode_000001"
    array = io.BytesIO()
    np.save(array, np.zeros((2, 27)))
    files = {f"{directory}/metadata.json": b'{"episode_id": "legacy-original"}',
             f"{directory}/actions.npy": array.getvalue()}
    archive = tmp_path / "legacy.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, value in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            tar.addfile(info, io.BytesIO(value))
    manifest = tmp_path / "legacy.json"
    manifest.write_text(json.dumps({"archive": archive.name, "archive_sha256": digest_file(archive)}))
    release = {**inputs.release, "legacy_release": {"manifest": manifest.name, "sha256": digest_file(manifest)}}
    source = ReleaseSource(replace(inputs, root=tmp_path, release=release))
    records = source.read_legacy_records(episode_paths=[directory])
    assert records[directory]["actions"].shape == (2, 27)
    assert "recording_session_id" not in records[directory]["metadata"]
    with pytest.raises(ValueError, match="explicitly select"):
        source.read_legacy_records(episode_paths=["data/demos/rejected/example"])
