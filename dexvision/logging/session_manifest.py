"""Append-only manifests for genuine Level 4 recording sessions."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class SessionManifestError(ValueError):
    """Raised when a session manifest is missing, duplicated, or malformed."""


@dataclass(frozen=True)
class RecordingSession:
    """Stable provenance for one process/calibration recording block."""

    recording_session_id: str
    operator_id: str
    split: str
    process_start_timestamp: str
    reset_seed: int
    calibration_record_digest: str

    def validate(self) -> None:
        for field_name in (
            "recording_session_id",
            "operator_id",
            "process_start_timestamp",
            "calibration_record_digest",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise SessionManifestError(f"{field_name} must be a non-empty string.")
        if self.split not in {"train", "validation", "test"}:
            raise SessionManifestError("split must be train, validation, or test.")
        if not isinstance(self.reset_seed, int) or isinstance(self.reset_seed, bool):
            raise SessionManifestError("reset_seed must be an integer.")


@dataclass(frozen=True)
class SessionManifest:
    """Versioned collection of unique recording sessions."""

    version: str
    sessions: tuple[RecordingSession, ...]

    def validate(self) -> None:
        if not self.version:
            raise SessionManifestError("session manifest version is required.")
        seen: set[str] = set()
        for session in self.sessions:
            session.validate()
            if session.recording_session_id in seen:
                raise SessionManifestError(
                    "duplicate recording_session_id in session manifest: "
                    f"{session.recording_session_id}"
                )
            seen.add(session.recording_session_id)

    def append(self, session: RecordingSession) -> "SessionManifest":
        session.validate()
        if any(
            existing.recording_session_id == session.recording_session_id
            for existing in self.sessions
        ):
            raise SessionManifestError(
                f"recording_session_id already exists: {session.recording_session_id}"
            )
        result = SessionManifest(self.version, (*self.sessions, session))
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "sessions": [asdict(item) for item in self.sessions]}


def load_session_manifest(path: str | Path) -> SessionManifest:
    """Load and validate an existing session manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise SessionManifestError(f"session manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionManifestError(f"could not read session manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SessionManifestError("session manifest root must be a mapping.")
    version = payload.get("version")
    raw_sessions = payload.get("sessions")
    if not isinstance(version, str) or not version:
        raise SessionManifestError("session manifest version must be a non-empty string.")
    if isinstance(raw_sessions, str) or not isinstance(raw_sessions, Sequence):
        raise SessionManifestError("session manifest sessions must be a sequence.")
    sessions: list[RecordingSession] = []
    for raw in raw_sessions:
        if not isinstance(raw, Mapping):
            raise SessionManifestError("each session manifest entry must be a mapping.")
        try:
            sessions.append(
                RecordingSession(
                    recording_session_id=raw["recording_session_id"],
                    operator_id=raw["operator_id"],
                    split=raw["split"],
                    process_start_timestamp=raw["process_start_timestamp"],
                    reset_seed=raw["reset_seed"],
                    calibration_record_digest=raw["calibration_record_digest"],
                )
            )
        except KeyError as exc:
            raise SessionManifestError(
                f"session manifest entry is missing required field: {exc.args[0]}"
            ) from exc
    manifest = SessionManifest(version=version, sessions=tuple(sessions))
    manifest.validate()
    return manifest


def append_session_manifest(
    path: str | Path,
    session: RecordingSession,
    *,
    version: str = "level4/session-manifest-v1",
) -> SessionManifest:
    """Atomically append one new session, refusing id collisions."""

    manifest_path = Path(path)
    if manifest_path.exists():
        manifest = load_session_manifest(manifest_path)
        if manifest.version != version:
            raise SessionManifestError(
                f"session manifest version mismatch: expected {version}, got {manifest.version}."
            )
    else:
        manifest = SessionManifest(version=version, sessions=())
    updated = manifest.append(session)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(updated.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)
    return updated


def next_episode_directory(
    dataset_dir: str | Path,
    *,
    recording_session_id: str,
    prefix: str = "episode",
) -> Path:
    """Return the first unused append-only episode path for a resumed session."""

    if not recording_session_id.strip():
        raise SessionManifestError("recording_session_id must be a non-empty string.")
    if not prefix.strip():
        raise SessionManifestError("episode prefix must be a non-empty string.")
    session_dir = Path(dataset_dir) / recording_session_id
    index = 1
    while True:
        candidate = session_dir / f"{prefix}_{index:06d}"
        if not candidate.exists():
            return candidate
        index += 1
