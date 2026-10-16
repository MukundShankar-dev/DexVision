"""Verify and restore immutable archives transported as ordered Git LFS parts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def _local_file(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or any(char in name for char in "/\\:\n\r\0")):
        raise ValueError(f"Unsafe multipart filename: {name!r}")
    path = root / name
    if path.is_symlink():
        raise ValueError(f"Symlink is not a release payload: {name}")
    return path


def _check_file(path: Path, expected: dict) -> None:
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
            size += len(block)
    if size != expected["size_bytes"] or sha.hexdigest() != expected["sha256"]:
        raise ValueError(f"Release size/checksum mismatch: {path.name}")


def restore_multipart_releases(manifest_path: Path, *, verify_only: bool = False) -> dict:
    """Stream parts in manifest order; never replace an existing archive.

    Parts and their concatenation must match their published size and SHA-256.
    Existing originals are verified, never rewritten. Newly restored archives
    are published only after full verification, using an exclusive hard link.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "dexvision-lfs-multipart-v1" or not manifest.get("archives"):
        raise ValueError("Unsupported or empty multipart release manifest")
    root = manifest_path.parent
    results = []
    names = set()
    for entry in manifest["archives"]:
        archive = _local_file(root, entry["name"])
        if archive.name in names or not entry["parts"]:
            raise ValueError("Duplicate archive or empty part list")
        names.add(archive.name)
        parts = [_local_file(root, part["name"]) for part in entry["parts"]]
        if len(set(parts)) != len(parts) or archive in parts:
            raise ValueError("Duplicate or overlapping multipart payload names")
        existing = archive.exists()
        if existing:
            _check_file(archive, entry)
        temporary = None
        output = None
        try:
            if not verify_only and not existing:
                output = tempfile.NamedTemporaryFile(
                    dir=root, prefix=f".{archive.name}.", suffix=".partial", delete=False,
                )
                temporary = Path(output.name)
            sha = hashlib.sha256()
            total = 0
            for path, part in zip(parts, entry["parts"], strict=True):
                part_sha = hashlib.sha256()
                size = 0
                with path.open("rb") as stream:
                    while block := stream.read(8 * 1024 * 1024):
                        part_sha.update(block)
                        sha.update(block)
                        size += len(block)
                        if output is not None:
                            output.write(block)
                if size != part["size_bytes"] or part_sha.hexdigest() != part["sha256"]:
                    raise ValueError(f"Multipart size/checksum mismatch: {path.name}")
                total += size
            if total != entry["size_bytes"] or sha.hexdigest() != entry["sha256"]:
                raise ValueError(f"Concatenated archive checksum mismatch: {archive.name}")
            if output is not None:
                output.close()
                os.link(temporary, archive)  # Fails rather than replacing any existing path.
            results.append({"archive": archive.name, "size_bytes": total,
                            "sha256": sha.hexdigest(), "restored": output is not None})
        finally:
            if output is not None:
                output.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"passed": True, "verify_only": verify_only, "archives": results}
