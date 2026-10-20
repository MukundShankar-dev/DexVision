"""What code produced a result: the git commit and whether the tree had uncommitted changes."""

from __future__ import annotations

import subprocess

from dexvision import REPO_ROOT


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def code_version() -> dict[str, object]:
    """{"commit": sha or None, "dirty": bool or None}. ``dirty`` means results may not reproduce
    from the commit alone."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}
