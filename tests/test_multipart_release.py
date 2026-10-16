"""Multipart transport preserves immutable releases and rejects corrupt inputs."""
import hashlib
import json
from pathlib import Path

import pytest

from dexvision.apps.restore_multipart_releases import main
from dexvision.logging.multipart_release import restore_multipart_releases


def _entry(name, data):
    return {"name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def release(tmp_path):
    data = b"immutable-archive\x00" * 13
    parts = []
    for index, chunk in enumerate([data[:37], data[37:99], data[99:]]):
        name = f"release.tar.gz.part-{index + 1:03d}"
        (tmp_path / name).write_bytes(chunk)
        parts.append(_entry(name, chunk))
    archive = _entry("release.tar.gz", data)
    archive["parts"] = parts
    manifest = tmp_path / "transport.json"
    manifest.write_text(json.dumps({"schema": "dexvision-lfs-multipart-v1", "archives": [archive]}))
    return manifest, data


def test_clean_restore_and_existing_archive_are_immutable(release):
    manifest, data = release
    report = restore_multipart_releases(manifest)
    archive = manifest.parent / "release.tar.gz"
    assert archive.read_bytes() == data and report["archives"][0]["restored"]
    before = archive.stat().st_mtime_ns
    assert not restore_multipart_releases(manifest)["archives"][0]["restored"]
    assert archive.stat().st_mtime_ns == before
    assert not list(manifest.parent.glob("*.partial"))


def test_verify_only_does_not_create_original(release):
    manifest, _ = release
    assert restore_multipart_releases(manifest, verify_only=True)["passed"]
    assert not (manifest.parent / "release.tar.gz").exists()


@pytest.mark.parametrize("kind", ["corrupt", "missing", "reordered", "duplicate", "bad_total"])
def test_bad_parts_never_publish_an_archive(release, kind):
    manifest, _ = release
    obj = json.loads(manifest.read_text())
    entry = obj["archives"][0]
    first = manifest.parent / entry["parts"][0]["name"]
    if kind == "corrupt":
        first.write_bytes(b"corrupt")
    elif kind == "missing":
        first.unlink()
    elif kind == "reordered":
        entry["parts"].reverse()
    elif kind == "duplicate":
        entry["parts"].append(entry["parts"][0])
    else:
        entry["sha256"] = "0" * 64
    manifest.write_text(json.dumps(obj))
    with pytest.raises((ValueError, OSError)):
        restore_multipart_releases(manifest)
    assert not (manifest.parent / "release.tar.gz").exists()
    assert not list(manifest.parent.glob("*.partial"))


def test_existing_wrong_original_is_preserved(release):
    manifest, _ = release
    archive = manifest.parent / "release.tar.gz"
    archive.write_bytes(b"operator-owned")
    with pytest.raises(ValueError, match="checksum"):
        restore_multipart_releases(manifest)
    assert archive.read_bytes() == b"operator-owned"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:\\escape", "..", "bad\nname"])
def test_unsafe_payload_names_fail(release, name):
    manifest, _ = release
    obj = json.loads(manifest.read_text())
    obj["archives"][0]["parts"][0]["name"] = name
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="Unsafe"):
        restore_multipart_releases(manifest)


def test_cli_reports_missing_payload_and_succeeds_after_restore(release, capsys):
    manifest, _ = release
    assert main(["--manifest", str(manifest)]) == 0
    assert main(["--manifest", str(Path('missing-transport.json'))]) == 2
    assert 'ERROR:' in capsys.readouterr().err
