"""Union is a pure derivation of frozen development parents, never new physics."""

import json
from pathlib import Path
import pytest
from dexvision.logging.visual_stream import write_json, digest_file
from dexvision.perception.perception_baseline import save_artifact_index
from dexvision.logging.perception_union import union_development
from dexvision.logging.perception_development import load_protocol


def parent(tmp_path, name, seed, role="development"):
    directory = tmp_path / name
    directory.mkdir()
    root = directory / "data"
    root.mkdir()
    case = {
        "episode_id": name,
        "session_id": "s" + name,
        "seed": seed,
        "split": "train",
    }
    protocol = {
        "version": "level5/perception-development-v1",
        "dataset_role": role,
        "cases": [case],
        "pins": {},
        "splits": {"train": {}, "validation": {}},
    }
    p = directory / "protocol.json"
    write_json(p, protocol)
    p.with_suffix(".sha256").write_text(f"{digest_file(p)}  protocol.json\n")
    (root / "calibration.json").write_text('{"camera":"same"}')
    (root / "images").mkdir()
    (root / f"images/{name}.png").write_bytes(b"image" + name.encode())
    row = {
        "source_episode_id": name,
        "recording_session_id": "s" + name,
        "split": "train",
        "rgb": f"images/{name}.png",
        "mask": f"images/{name}.png",
        "rgb_pixel_sha256": name,
    }
    (root / "frames.jsonl").write_text(json.dumps(row) + "\n")
    write_json(root / "owners.json", {name: {"entity_id": "cube"}})
    write_json(
        root / "manifest.json",
        {
            "protocol_sha256": digest_file(p),
            "training_eligible": True,
            "splits": {"train": {}, "validation": {}},
            "historical_pixel_collisions": 0,
            "cross_split_pixel_collisions": 0,
            "no_filtered_attempts": True,
        },
    )
    save_artifact_index(root)
    return str(p), str(root)


def test_union_preserves_parents_copied_images_and_split(tmp_path):
    a = parent(tmp_path, "a", 1)
    b = parent(tmp_path, "b", 2)
    before = digest_file(Path(a[1]) / "SHA256SUMS")
    out = tmp_path / "union"
    result = union_development([a, b], out)
    assert result["episode_count"] == 2 and result["frame_count"] == 2
    assert result["source_kind"] == "derived_union_of_frozen_development_releases"
    assert before == digest_file(Path(a[1]) / "SHA256SUMS")
    assert (out / "dataset/images/a.png").read_bytes() == b"imagea"
    assert load_protocol(out / "protocol/protocol.json")["no_new_sessions"]
    with pytest.raises(FileExistsError):
        union_development([a, b], out)


def test_union_refuses_consumed_parent_and_overlap_before_output(tmp_path):
    a = parent(tmp_path, "a", 1)
    b = parent(tmp_path, "b", 1)
    out = tmp_path / "union"
    with pytest.raises(ValueError, match="overlap"):
        union_development([a, b], out)
    assert not out.exists()
    c = parent(tmp_path, "c", 3, "confirmation")
    with pytest.raises(ValueError, match="TRAIN/validation"):
        union_development([c], out)
    assert not out.exists()


def test_union_rejects_consumed_role_before_payload_access(tmp_path, monkeypatch):
    import dexvision.logging.perception_union as module

    monkeypatch.setattr(
        module, "load_protocol", lambda _: {"dataset_role": "confirmation"}
    )
    monkeypatch.setattr(
        module,
        "verify_dataset",
        lambda *_: pytest.fail("Consumed payload must not be accessed"),
    )
    with pytest.raises(ValueError, match="TRAIN/validation"):
        module.union_development([("protocol", "absent")], tmp_path / "out")
