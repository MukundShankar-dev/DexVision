"""Release integrity, immutable output and safe clean-directory restoration."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from dexvision.apps.verify_dataset_release import main
from dexvision.logging.dataset_release import (
    VERSION, collect_audited_files, digest_file, prepare_release, safe_path,
    verify_archive, verify_release, write_archive,
)


def inventory(root):
    return {p.relative_to(root).as_posix(): {"sha256": digest_file(p),
            "size_bytes": p.stat().st_size} for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def payload(tmp_path):
    root = tmp_path / "source"
    (root / "data").mkdir(parents=True)
    (root / "data/state.npy").write_bytes(b"synthetic-state\x00" * 100)
    (root / "rgb.png").write_bytes(b"synthetic-pixels")
    return root, inventory(root)


def test_deterministic_archive_and_clean_restore_preserve_source(payload, tmp_path):
    root, files = payload
    first, second = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
    write_archive(root, first, files)
    write_archive(root, second, files)
    assert first.read_bytes() == second.read_bytes()
    restored = tmp_path / "clean"
    verify_archive(first, files, restored)
    assert inventory(restored) == files == inventory(root)
    with pytest.raises(FileExistsError):
        verify_archive(first, files, restored)
    with pytest.raises(FileExistsError):
        write_archive(root, first, files)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a//b", "./a",
                                  "C:/windows", "a\\b", "", "a\nb"])
def test_cross_platform_unsafe_paths_fail(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe release path"):
        safe_path(tmp_path, name)


def test_symlink_parent_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation unavailable on this platform")
    with pytest.raises(ValueError, match="Symlink"):
        safe_path(tmp_path, "link/file")


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "link", "traversal",
                                  "wrong_size", "wrong_hash"])
def test_corrupt_or_unsafe_members_fail(payload, tmp_path, kind):
    root, files = payload
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in files:
            if kind == "missing" and name == "rgb.png":
                continue
            data = (root / name).read_bytes()
            if name == "rgb.png":
                if kind == "wrong_size":
                    data += b"!"
                if kind == "wrong_hash":
                    data = b"X" * len(data)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if kind in ("extra", "duplicate", "link", "traversal"):
            name = {"extra": "unlisted", "duplicate": "rgb.png",
                    "link": "link", "traversal": "../escape"}[kind]
            info = tarfile.TarInfo(name)
            if kind == "link":
                info.type = tarfile.SYMTYPE
                info.linkname = "outside"
            tar.addfile(info, io.BytesIO())
    with pytest.raises(ValueError):
        verify_archive(archive, files, tmp_path / "restore")
    assert not (tmp_path / "escape").exists()


def make_release(tmp_path):
    root = tmp_path / "input"
    audit = root / "audit"
    audit.mkdir(parents=True)
    report = {"passed": True, "issues": [], "dataset_digest": "frozen",
              "active_episode_count": 0, "visual": {"file_sha256": {}}}
    (audit / "report.json").write_text(json.dumps(report))
    for split in ("train", "validation", "test"):
        (audit / f"{split}.json").write_text(json.dumps({"episodes": []}))
    files = inventory(root)
    release = tmp_path / "level4-v1"
    (release / "splits").mkdir(parents=True)
    (release / "audit_report.json").write_text(json.dumps(report))
    for split in ("train", "validation", "test"):
        (release / f"splits/{split}.json").write_bytes((audit / f"{split}.json").read_bytes())
    archive = tmp_path / "level4-v1.tar.gz"
    write_archive(root, archive, files)
    manifest = {"version": VERSION, "files": files, "file_count": len(files),
                "uncompressed_bytes": sum(f["size_bytes"] for f in files.values()),
                "audit_path": "audit", "dataset_digest": "frozen",
                "active_episode_count": 0, "visual_frame_count": 0,
                "archive": {"name": archive.name, "sha256": digest_file(archive),
                            "size_bytes": archive.stat().st_size},
                "status": "candidate", "publication_blockers": ["owner_license_decision"]}
    (release / "manifest.json").write_text(json.dumps(manifest))
    refresh_checksums(release)
    return release, archive


def refresh_checksums(release):
    (release / "SHA256SUMS").write_text("".join(
        f"{digest_file(p)}  {p.relative_to(release).as_posix()}\n"
        for p in sorted(release.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))


def test_candidate_integrity_never_implies_release_readiness(tmp_path):
    release, _ = make_release(tmp_path)
    result = verify_release(release, restore_dir=tmp_path / "clean")
    assert result["integrity_passed"]
    assert not result["publication_ready"]
    with pytest.raises(ValueError, match="Publication is blocked"):
        verify_release(release, require_ready=True)


def test_changed_metadata_and_archive_fail(tmp_path):
    release, archive = make_release(tmp_path)
    original = archive.read_bytes()
    archive.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Archive SHA-256/size"):
        verify_release(release)
    archive.write_bytes(original)
    (release / "splits/train.json").write_text("{}")
    with pytest.raises(ValueError, match="metadata checksum"):
        verify_release(release)
    refresh_checksums(release)
    with pytest.raises(ValueError, match="Frozen split differs"):
        verify_release(release)


def test_checksum_index_cannot_omit_manifest(tmp_path):
    release, _ = make_release(tmp_path)
    sums = release / "SHA256SUMS"
    sums.write_text("\n".join(line for line in sums.read_text().splitlines()
                              if not line.endswith("  manifest.json")))
    with pytest.raises(ValueError, match="omits required"):
        verify_release(release)


def test_cli_missing_payload_and_metadata_fail_gracefully(tmp_path, capsys):
    release, archive = make_release(tmp_path)
    assert main(["--release-dir", str(release)]) == 0
    assert "candidate only" in capsys.readouterr().out
    archive.unlink()
    assert main(["--release-dir", str(release)]) == 2
    assert "Payload missing" in capsys.readouterr().err
    assert main(["--release-dir", str(tmp_path / "missing")]) == 2


def test_packager_refuses_existing_output_before_reading_source(tmp_path):
    release = tmp_path / "existing"
    release.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        prepare_release(root=tmp_path, audit_relative="missing", release_dir=release,
                        archive=tmp_path / "new.tar.gz", source_commit="0" * 40)


def test_packager_rejects_incomplete_audit_without_creating_archive(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "file_sha256.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum index is incomplete"):
        collect_audited_files(tmp_path, "audit")


def test_checkpoint_status_requires_completion_evidence_and_stops_at_level4():
    root = Path(__file__).resolve().parents[1]
    status = (root / "docs/CURRENT_STATUS.md").read_text()
    progress = (root / "docs/progress_level_4.md").read_text()
    if "## Last Completed Checkpoint\n\nLevel 4.9" in status:
        receipt = json.loads((root / "datasets/level4-v1/completion_receipt.json").read_text())
        assert receipt["clean_clone_retrieval_passed"]
        assert receipt["owner_accepted_delegated_verification"]
        assert "[x] 4.9 immutable release restores" in progress
        assert "## Next Target Checkpoint\n\nNone" in status
    else:
        assert "## Last Completed Checkpoint\n\nLevel 4.8" in status
        assert "## Next Target Checkpoint\n\nLevel 4.9" in status
        assert "[ ] 4.9 immutable release restores" in progress


def test_lfs_retrieval_uses_empty_repository_and_exact_object(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from dexvision.logging.dataset_release import retrieve_lfs_payload

    release, archive = make_release(tmp_path)
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["storage"] = {"provider": "github_git_lfs",
                           "repository": "https://example.invalid/release.git"}
    manifest_path.write_text(json.dumps(manifest))
    refresh_checksums(release)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "smudge" in command:
            assert manifest["archive"]["sha256"].encode() in kwargs["input"]
            assert "lfs.storage=.lfs" in command
            kwargs["stdout"].write(archive.read_bytes())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("dexvision.logging.dataset_release.subprocess.run", run)
    directory = tmp_path / "download"
    downloaded = retrieve_lfs_payload(release, directory)
    assert downloaded.read_bytes() == archive.read_bytes()
    assert len(calls) == 3
    with pytest.raises(FileExistsError):
        retrieve_lfs_payload(release, directory)


def test_lfs_failed_transfer_is_not_success(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from dexvision.logging.dataset_release import retrieve_lfs_payload

    release, _ = make_release(tmp_path)
    path = release / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["storage"] = {"provider": "github_git_lfs",
                           "repository": "https://example.invalid/release.git"}
    path.write_text(json.dumps(manifest))
    refresh_checksums(release)

    def run(command, **kwargs):
        if "smudge" in command:
            kwargs["stdout"].write(kwargs["input"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("dexvision.logging.dataset_release.subprocess.run", run)
    with pytest.raises(ValueError, match="download failed or returned a pointer"):
        retrieve_lfs_payload(release, tmp_path / "download")


def completed_fixture(tmp_path):
    release, archive = make_release(tmp_path)
    path = release / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(licenses={"custom_workcell": "Apache-2.0", "generated_dataset": "CC-BY-4.0"},
                    storage={"provider": "github_git_lfs"})
    path.write_text(json.dumps(manifest))
    receipt = {
        "version": "level4/release-completion-v1", "manifest_sha256": digest_file(path),
        "archive_sha256": digest_file(archive), "file_count": manifest["file_count"],
        "clean_clone_retrieval_passed": True, "independent_restored_readback_passed": True,
        "legacy_retrieval_and_checksum_passed": True, "automated_checks_passed": True,
        "owner_accepted_delegated_verification": True, "clean_clone_commit": "a" * 40,
        "legacy_archive_sha256": "b" * 64, "owner_completion_request": "Complete 4.9",
    }
    (release / "completion_receipt.json").write_text(json.dumps(receipt))
    refresh_checksums(release)
    return release, receipt


def test_completion_receipt_enables_readiness_without_rewriting_frozen_manifest(tmp_path):
    release, _ = completed_fixture(tmp_path)
    original = (release / "manifest.json").read_bytes()
    assert verify_release(release, require_ready=True)["publication_ready"]
    assert (release / "manifest.json").read_bytes() == original


@pytest.mark.parametrize("key,value", [
    ("archive_sha256", "c" * 64), ("manifest_sha256", "d" * 64),
    ("owner_accepted_delegated_verification", False),
    ("legacy_retrieval_and_checksum_passed", False), ("clean_clone_commit", "main"),
])
def test_completion_receipt_rejects_mismatched_or_incomplete_evidence(tmp_path, key, value):
    release, receipt = completed_fixture(tmp_path)
    receipt[key] = value
    (release / "completion_receipt.json").write_text(json.dumps(receipt))
    refresh_checksums(release)
    with pytest.raises(ValueError, match="Completion receipt"):
        verify_release(release, require_ready=True)


def test_unindexed_completion_receipt_cannot_clear_pending_status(tmp_path):
    release, _ = completed_fixture(tmp_path)
    path = release / "SHA256SUMS"
    path.write_text("\n".join(line for line in path.read_text().splitlines()
                              if not line.endswith("  completion_receipt.json")))
    with pytest.raises(ValueError, match="missing from SHA256SUMS"):
        verify_release(release, require_ready=True)
