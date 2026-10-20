"""The test ledger: every test-split evaluation, appended automatically and committed.

It does not stop anyone from iterating against the test set. It makes doing so visible, and
enforces the per-(skill, policy) budget from the task spec.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from dexvision import REPO_ROOT

DEFAULT_LEDGER = REPO_ROOT / "results" / "test_ledger.jsonl"


class BudgetExceeded(RuntimeError):
    """The (skill, policy) pair has used its test budget."""


def read_ledger(path: Path = DEFAULT_LEDGER) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def check_test_budget(skill: str, policy: str, budget: int, path: Path = DEFAULT_LEDGER) -> int:
    """Return how many test runs remain, or raise if none do. Call before running."""
    used = sum(1 for e in read_ledger(path) if e["skill"] == skill and e["policy"] == policy)
    if used >= budget:
        raise BudgetExceeded(
            f"{skill}/{policy} has used {used} of {budget} test runs. A changed method is a new "
            "policy name, and it needs a new test seed range written down before you look."
        )
    return budget - used


def append_test_run(entry: dict, path: Path = DEFAULT_LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
