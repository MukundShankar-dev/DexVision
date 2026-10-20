"""Canonical 5.7 family metrics from frozen requests, including rejected entries.

Early failure reports can lack legacy target labels. Those omissions must not
invent an extra zero-success goal family. No labels or group boundaries are fit
from outcomes; the frozen request and calibrated asset/target mappings define it.
"""

from __future__ import annotations


def physical_success(result):
    if result["terminal_result"] != "succeeded":
        return False
    if "physical_pose_audit" in result:
        valid = result["physical_pose_audit"]["measured_pose_and_dwell_passed"]
    elif "physical_audit" in result:
        valid = result["physical_audit"]["passed"]
    else:
        raise ValueError("Successful control lacks independent physical evidence")
    return bool(valid and result.get("decoder_audit", {}).get("passed", True))


def canonical_family_metrics(rows, *, object_families, target_types):
    """Rows are (frozen request row, immutable result) pairs for one seed/arm."""
    groups = {}
    for request, result in rows:
        entity = request["entity_id"]
        goal = request["typed_goal"]
        target = goal.get(
            "target_id",
            goal.get(
                "target_zone", goal.get("button_id", goal.get("entity_id", entity))
            ),
        )
        obj = object_families[entity]
        category = target_types.get(
            target, object_families.get(target, "calibrated_target")
        )
        for name in (f"object/{obj}", f"goal/{category}"):
            group = groups.setdefault(name, {"attempts": 0, "successes": 0})
            group["attempts"] += 1
            group["successes"] += int(physical_success(result))
    if not groups:
        raise ValueError("Canonical family coverage is empty")
    for group in groups.values():
        group["success_rate"] = group["successes"] / group["attempts"]
    return {
        "groups": groups,
        "worst_family_success_rate": min(g["success_rate"] for g in groups.values()),
    }
