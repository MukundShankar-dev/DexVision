"""Append-only Level 4 packaging and portable, streaming release verification.

The frozen audit selects every payload file. Local candidates are verifiable
before owner licensing and hosting decisions, but are never publication-ready.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

SPLITS = ("train", "validation", "test")
VERSION = "level4/dataset-release-v1"


def retrieve_lfs_payload(release_dir: Path, download_dir: Path) -> Path:
    """Retrieve the exact object through a new Git repository and empty LFS cache.

    No local working dataset, Git index, existing LFS cache, or mutable branch
    selects the object. SHA256SUMS/manifest must come from a trusted checkout.
    """
    manifest_path = safe_path(release_dir, "manifest.json")
    manifest = read_json(manifest_path)
    expected_line = f"{digest_file(manifest_path)}  manifest.json"
    if expected_line not in (release_dir / "SHA256SUMS").read_text().splitlines():
        raise ValueError("Release metadata checksum mismatch: manifest.json")
    if manifest["storage"]["provider"] != "github_git_lfs":
        raise ValueError("Release has no Git LFS retrieval source")
    archive = manifest["archive"]
    if not re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]):
        raise ValueError("Invalid archive SHA-256")
    download_dir.mkdir(parents=True, exist_ok=False)
    target = safe_path(download_dir, archive["name"])
    pointer = ("version https://git-lfs.github.com/spec/v1\n"
               f"oid sha256:{archive['sha256']}\nsize {archive['size_bytes']}\n").encode()
    commands = [
        ["git", "init", "--quiet", str(download_dir)],
        ["git", "-C", str(download_dir), "remote", "add", "origin",
         manifest["storage"]["repository"]],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode:
            raise ValueError("Git LFS retrieval setup failed; check Git and remote access")
    with target.open("xb") as stream:
        result = subprocess.run(
            ["git", "-C", str(download_dir), "-c", "lfs.storage=.lfs",
             "lfs", "smudge", "--", archive["name"]],
            input=pointer, stdout=stream, stderr=subprocess.PIPE, check=False)
    if (result.returncode or target.stat().st_size != archive["size_bytes"]
            or digest_file(target) != archive["sha256"]):
        raise ValueError("Git LFS download failed or returned a pointer; check remote publication, "
                         "authentication and quota. Retry into a new directory.")
    return target


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def safe_path(root: Path, name: str) -> Path:
    """Reject traversal, Windows drive paths, ambiguous names and all symlinks."""
    parts = PurePosixPath(name).parts
    if (not parts or name != PurePosixPath(name).as_posix() or name.startswith("/")
            or any(p in ("..", ".") for p in parts)
            or any(c in name for c in ("\\", ":", "\n", "\r", "\x00"))):
        raise ValueError(f"Unsafe release path: {name!r}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink in release path: {name}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Release path escapes root: {name}")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _add_files(files: dict, root: Path, prefix: str, hashes: dict) -> None:
    for name, expected in hashes.items():
        relative = f"{prefix}/{name}" if prefix else name
        path = safe_path(root, relative)
        if not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"Missing or changed audited source: {relative}")
        entry = {"sha256": expected, "size_bytes": path.stat().st_size}
        if relative in files and files[relative] != entry:
            raise ValueError(f"Conflicting inventory: {relative}")
        files[relative] = entry


def collect_audited_files(root: Path, audit_relative: str) -> tuple[dict, dict]:
    """Select active episodes, complete visual export and bound provenance only."""
    audit_dir = safe_path(root, audit_relative)
    files = {}
    audit_hashes = read_json(audit_dir / "file_sha256.json")
    if set(audit_hashes) != {"report.json", *(f"{s}.json" for s in SPLITS)}:
        raise ValueError("Audit checksum index is incomplete")
    _add_files(files, root, audit_relative, audit_hashes)
    _add_files(files, root, audit_relative,
               {"file_sha256.json": digest_file(audit_dir / "file_sha256.json")})
    report = read_json(audit_dir / "report.json")
    if (report.get("passed") is not True or report.get("issues")
            or report["visual"].get("passed") is not True):
        raise ValueError("Only a passing frozen audit may be packaged")
    seen = set()
    for split in SPLITS:
        manifest = read_json(audit_dir / f"{split}.json")
        if (manifest.get("status") != "frozen"
                or manifest.get("dataset_audit_passed") is not True
                or manifest.get("split") != split
                or manifest["config_digest"] != report["config_digest"]
                or manifest["manifest_digest"] != content_digest(
                    {k: v for k, v in manifest.items() if k != "manifest_digest"})):
            raise ValueError(f"Invalid frozen split: {split}")
        if split != "train" and manifest["normalization_inputs"]:
            raise ValueError("Held-out normalization input")
        for row in manifest["episodes"]:
            if row["episode_id"] in seen or not row["audit_passed"]:
                raise ValueError("Duplicate or unqualified active episode")
            seen.add(row["episode_id"])
            _add_files(files, root, f"data/demos/level4/{row['source_path']}",
                       row["file_sha256"])
    if len(seen) != report["active_episode_count"]:
        raise ValueError("Active episode count differs from audit")
    _add_files(files, root, "data/visual/level4", report["visual"]["file_sha256"])
    _add_files(files, root, "data/demos/level4", report["provenance_file_sha256"])
    _add_files(files, root, "", report["config_sha256"])
    for name in report["config_sha256"]:
        if name.endswith("/receipt.json"):
            receipt = read_json(safe_path(root, name))
            snapshot = str(PurePosixPath(name).parent / "recording_snapshot")
            _add_files(files, root, snapshot, receipt["recording_snapshot_sha256"])
    _add_files(files, root, "", report["implementation_sha256"])
    for asset in report["visual"]["asset_provenance"]:
        if "files" in asset:
            _add_files(files, root, "assets/mujoco", asset["files"])
            _add_files(files, root, "assets/mujoco/menagerie/shadow_hand",
                       {"LICENSE": asset["license_sha256"]})
        else:
            _add_files(files, root, "assets/mujoco", {"workcell_scene.xml": asset["sha256"]})
    # Frozen dependency/config files needed by the archived source and scene.
    for name in ("pyproject.toml", "environment.yml", "configs/level1_teleop.yaml",
                 "configs/level4_visual_dataset.yaml", "assets/mujoco/hand_scene.xml",
                 "assets/mujoco/README.md", "assets/mujoco/menagerie/shadow_hand/README.md"):
        _add_files(files, root, "", {name: digest_file(safe_path(root, name))})
    # Some visual sources were superseded by 4.8. Preserve their exact archival
    # bytes separately from active split membership; never promote them to targets.
    sources = json.loads((root / "data/visual/level4/sources.json").read_text())
    for source in sources:
        _add_files(files, root, "data/demos/level4/" + source["source_path"],
                   source["source_file_sha256"])
    return dict(sorted(files.items())), report


def write_archive(root: Path, archive: Path, files: dict) -> None:
    """Write a deterministic gzip/tar without replacing an existing artifact."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for name, entry in sorted(files.items()):
                    source = safe_path(root, name)
                    if source.stat().st_size != entry["size_bytes"]:
                        raise ValueError(f"Source size changed: {name}")
                    info = tarfile.TarInfo(name)
                    info.size = entry["size_bytes"]
                    info.mode = 0o644
                    with source.open("rb") as stream:
                        tar.addfile(info, stream)


def verify_archive(archive: Path, files: dict, restore_dir: Path | None = None) -> None:
    """Check exact member set, sizes and streamed bytes; optionally restore safely.

    Restoration requires a nonexistent directory. A partial failed restore is
    retained for diagnosis and cannot be silently reused or overwritten.
    """
    if restore_dir is not None:
        restore_dir.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = member.name
            safe_path(restore_dir or archive.parent, name)
            if not member.isfile() or member.issparse() or name in seen or name not in files:
                raise ValueError(f"Unexpected, duplicate or nonregular archive member: {name}")
            expected = files[name]
            if member.size != expected["size_bytes"]:
                raise ValueError(f"Archive member size mismatch: {name}")
            seen.add(name)
            digest = hashlib.sha256()
            stream = tar.extractfile(member)
            destination = None
            try:
                if restore_dir is not None:
                    target = safe_path(restore_dir, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    destination = target.open("xb")
                with stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                        if destination is not None:
                            destination.write(block)
            finally:
                if destination is not None:
                    destination.close()
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Archive member checksum mismatch: {name}")
    if seen != set(files):
        raise ValueError(f"Archive is missing {len(set(files) - seen)} expected files")


def prepare_release(*, root: Path, audit_relative: str, release_dir: Path,
                    archive: Path, source_commit: str,
                    release_config: Path | None = None) -> dict:
    """Prepare a local candidate; licensing/remote availability remain owner gates."""
    if release_dir.exists() or archive.exists():
        raise ValueError("Release directory/archive already exists; use a new version")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("A full source commit SHA is required")
    files, report = collect_audited_files(root, audit_relative)
    config = None
    if release_config is not None:
        import yaml

        config = yaml.safe_load(release_config.read_text(encoding="utf-8"))
        if (config["release_id"] != release_dir.name or config["source_commit"] != source_commit
                or config["audit_dir"] != audit_relative
                or config["licenses"]["custom_workcell"] != "Apache-2.0"
                or config["licenses"]["generated_dataset"] != "CC-BY-4.0"
                or not config["licenses"].get("owner_authorized_on")
                or config["storage"]["provider"] != "github_git_lfs"):
            raise ValueError("Release config does not bind the approved release/license/storage")
        extra = [release_config.resolve().relative_to(root.resolve()).as_posix(),
                 "dexvision/logging/dataset_release.py",
                 "dexvision/apps/verify_dataset_release.py",
                 "dexvision/apps/prepare_dataset_release.py",
                 "docs/level4_release_handoff.md"]
        license_dir = safe_path(root, config["license_directory"])
        extra.extend(p.relative_to(root).as_posix() for p in license_dir.rglob("*")
                     if p.is_file())
        for name in extra:
            _add_files(files, root, "", {name: digest_file(safe_path(root, name))})
    write_archive(root, archive, files)
    verify_archive(archive, files)
    if config is not None:
        storage = config["storage"]
        legacy_bytes = (root / "datasets/dexvision_level2_v1.tar.gz").stat().st_size
        if (archive.stat().st_size > storage["maximum_file_bytes"]
                or archive.stat().st_size + legacy_bytes
                > storage["documented_included_storage_bytes"]):
            raise ValueError("Archive exceeds the documented Git LFS quota; retain candidate")
    release_dir.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        target = release_dir / "splits" / f"{split}.json"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(root / audit_relative / f"{split}.json", target)
    shutil.copyfile(root / audit_relative / "report.json", release_dir / "audit_report.json")
    rows = [row for split in SPLITS
            for row in read_json(root / audit_relative / f"{split}.json")["episodes"]]
    layouts = {}
    for row in rows:
        layout = {key: row[key] for key in ("action_schema", "observation_schema", "schema_versions")}
        layouts[content_digest(layout)] = layout
    write_json(release_dir / "schemas.json", {
        "version": "level4/release-layouts-v1", "layouts_by_digest": layouts,
        "episode_layout_digest": {row["episode_id"]: content_digest({
            key: row[key] for key in ("action_schema", "observation_schema", "schema_versions")})
            for row in rows},
        "goal_authority": "configs/level4_dataset.yaml:skills.*.goal_fields",
        "episode_resolved_goals": "splits/*.json:episodes[].typed_goal",
    })
    manifest = {
        "version": VERSION, "release_id": release_dir.name,
        "status": "local_candidate_pending_owner_decisions_and_manual_restore",
        "source_commit": source_commit, "dataset_digest": report["dataset_digest"],
        "config_digest": report["config_digest"], "audit_path": audit_relative,
        "archive": {"name": archive.name, "sha256": digest_file(archive),
                    "size_bytes": archive.stat().st_size},
        "files": files, "file_count": len(files),
        "uncompressed_bytes": sum(f["size_bytes"] for f in files.values()),
        "active_episode_count": report["active_episode_count"],
        "split_episode_counts": report["splits"]["episode_counts"],
        "stream_counts": report["stream_counts"],
        "visual_frame_count": report["visual"]["frame_count"],
        "licenses": {"custom_workcell": None, "generated_dataset": None,
                     "shadow_hand": "Apache-2.0"},
        "storage": {"provider": None, "immutable_uri": None,
                    "available_bytes": None, "maximum_file_bytes": None},
        "limitations": report["limitations"],
        "publication_blockers": ["owner_license_decision", "documented_host_quota",
                                 "remote_retrieval_not_verified", "manual_restore_confirmation"],
        "legacy_release": {"manifest": "datasets/dexvision_level2_v1_manifest.json",
                           "sha256": digest_file(root / "datasets/dexvision_level2_v1_manifest.json")},
    }
    if config is not None:
        manifest.update(
            status="release_candidate_pending_publication_and_manual_restore",
            licenses=config["licenses"], storage=config["storage"],
            publication_blockers=["git_lfs_publication_not_verified", "manual_restore_confirmation"],
            license_notice_path=config["license_directory"] + "/NOTICE.md",
            handoff_path="docs/level4_release_handoff.md",
            source_version_note="Audit source is pinned by source_commit and implementation hashes. "
                                "Release tooling, licenses and handoff additions are pinned by "
                                "their individual archived SHA-256 values.",
        )
        shutil.copytree(root / config["license_directory"], release_dir / "licenses")
        shutil.copyfile(root / "docs/level4_release_handoff.md", release_dir / "HANDOFF.md")
    write_json(release_dir / "manifest.json", manifest)
    with (release_dir / "SHA256SUMS").open("x", encoding="utf-8") as stream:
        for path in sorted(release_dir.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                stream.write(f"{digest_file(path)}  {path.relative_to(release_dir).as_posix()}\n")
    with archive.with_suffix(archive.suffix + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(f"{manifest['archive']['sha256']}  {archive.name}\n")
    return manifest


def publication_blockers(release_dir: Path, manifest: dict, checksums: dict) -> list[str]:
    """Apply a checksum-bound completion receipt to build-time status.

    Receipts are project assertions obtained from a trusted Git checkout, not
    cryptographic proof of authorship. The frozen payload manifest is unchanged.
    """
    receipt_path = release_dir / "completion_receipt.json"
    if not receipt_path.exists():
        return list(manifest["publication_blockers"]) or ["completion_receipt_missing"]
    if "completion_receipt.json" not in checksums:
        raise ValueError("Completion receipt is missing from SHA256SUMS")
    receipt = read_json(receipt_path)
    expected = {
        "version": "level4/release-completion-v1",
        "manifest_sha256": checksums["manifest.json"],
        "archive_sha256": manifest["archive"]["sha256"],
        "file_count": manifest["file_count"],
        "clean_clone_retrieval_passed": True,
        "independent_restored_readback_passed": True,
        "legacy_retrieval_and_checksum_passed": True,
        "automated_checks_passed": True,
        "owner_accepted_delegated_verification": True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Completion receipt does not bind passing release evidence")
    if (not re.fullmatch(r"[0-9a-f]{40}", receipt.get("clean_clone_commit", ""))
            or not receipt.get("owner_completion_request")
            or not re.fullmatch(r"[0-9a-f]{64}", receipt.get("legacy_archive_sha256", ""))):
        raise ValueError("Completion receipt lacks commit, legacy digest or owner acceptance")
    if (not manifest.get("licenses", {}).get("custom_workcell")
            or not manifest.get("licenses", {}).get("generated_dataset")
            or manifest.get("storage", {}).get("provider") != "github_git_lfs"):
        raise ValueError("Completion receipt cannot bypass license or storage decisions")
    return []


def verify_release(release_dir: Path, *, archive: Path | None = None,
                   restore_dir: Path | None = None, require_ready: bool = False) -> dict:
    """Verify frozen metadata, complete archive and optional fresh restoration."""
    checksums = {}
    for line in (release_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or name in checksums:
            raise ValueError("Invalid or duplicate SHA256SUMS entry")
        path = safe_path(release_dir, name)
        if not path.is_file() or digest_file(path) != digest:
            raise ValueError(f"Release metadata checksum mismatch: {name}")
        checksums[name] = digest
    required = {"manifest.json", "audit_report.json", *(f"splits/{s}.json" for s in SPLITS)}
    if not required <= checksums.keys():
        raise ValueError("SHA256SUMS omits required metadata")
    manifest = read_json(release_dir / "manifest.json")
    if manifest["version"] != VERSION:
        raise ValueError("Unsupported release version")
    files = manifest["files"]
    if (manifest["file_count"] != len(files)
            or manifest["uncompressed_bytes"] != sum(f["size_bytes"] for f in files.values())):
        raise ValueError("Release inventory totals disagree")
    for name in files:
        safe_path(release_dir, name)
    audit = read_json(release_dir / "audit_report.json")
    if (audit["passed"] is not True or audit["issues"]
            or audit["dataset_digest"] != manifest["dataset_digest"]
            or audit["active_episode_count"] != manifest["active_episode_count"]):
        raise ValueError("Release audit binding mismatch")
    if files.get(f"{manifest['audit_path']}/report.json", {}).get("sha256") != checksums[
            "audit_report.json"]:
        raise ValueError("Release audit differs from archived evidence")
    for split in SPLITS:
        name = f"{manifest['audit_path']}/{split}.json"
        if files.get(name, {}).get("sha256") != checksums[f"splits/{split}.json"]:
            raise ValueError(f"Frozen split differs from archived audit: {split}")
        split_manifest = read_json(release_dir / f"splits/{split}.json")
        for row in split_manifest["episodes"]:
            for relative, digest in row["file_sha256"].items():
                name = f"data/demos/level4/{row['source_path']}/{relative}"
                if files.get(name, {}).get("sha256") != digest:
                    raise ValueError(f"Inventory omits/changes active episode file: {name}")
    for name, digest in audit["visual"]["file_sha256"].items():
        if files.get(f"data/visual/level4/{name}", {}).get("sha256") != digest:
            raise ValueError(f"Inventory omits/changes visual stream: {name}")
    blockers = publication_blockers(release_dir, manifest, checksums)
    if require_ready and blockers:
        raise ValueError("Publication is blocked: " + ", ".join(blockers))
    archive = archive or safe_path(release_dir.parent, manifest["archive"]["name"])
    if not archive.is_file():
        raise ValueError(f"Payload missing: {archive}; retrieve it or supply --archive")
    if (archive.stat().st_size != manifest["archive"]["size_bytes"]
            or digest_file(archive) != manifest["archive"]["sha256"]):
        raise ValueError("Archive SHA-256/size mismatch (check for an unpulled Git LFS pointer)")
    verify_archive(archive, files, restore_dir)
    return {"integrity_passed": True, "file_count": len(files),
            "active_episode_count": manifest["active_episode_count"],
            "visual_frame_count": manifest["visual_frame_count"],
            "publication_ready": not blockers,
            "publication_blockers": blockers,
            "restored_to": str(restore_dir) if restore_dir else None}
