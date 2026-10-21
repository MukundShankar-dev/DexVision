"""What produced a result: git commit, uncommitted changes, platform and simulator version.

MuJoCo is deterministic on one platform but not bit-identical across CPU architectures (pick
expert on val: same 200/200, but 48 of 200 episodes differ in detail between arm64 macOS and
x86-64 Linux). Results to be compared must come from the same platform.
"""

from __future__ import annotations

import platform
import subprocess

import mujoco

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
    """{"commit", "dirty", "platform", "mujoco"}. ``dirty`` means results may not reproduce from
    the commit alone."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "platform": f"{platform.system()}-{platform.machine()}",
        "mujoco": mujoco.__version__,
    }
