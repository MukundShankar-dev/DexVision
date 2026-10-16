"""Additional coverage gates prevent vacuous zero-switch qualification."""

from collections import defaultdict


def identity_coverage(instances):
    primary = [r for r in instances if r["primary_object"] and r["eligible"]]

    def summarize(rows, minimum):
        matched = [r for r in rows if r["matched"]]
        certified = sum(
            not r["rejection_reason"].startswith("ambiguous_identity") for r in matched
        )
        fraction = certified / len(matched) if matched else 0.0
        return {
            "visible": len(rows), "matched": len(matched),
            "unambiguous": certified, "fraction_of_matches": fraction,
            "minimum_fraction": minimum, "passed": bool(matched) and fraction >= minimum,
        }

    groups = defaultdict(list)
    for row in primary:
        groups[row["class_id"] + "/" + row["phase_stratum"]].append(row)
    overall = summarize(primary, 0.99)
    strata = {k: summarize(v, 0.95) for k, v in groups.items() if len(v) >= 8}
    return {
        "primary": overall, "supported_primary_class_phase": strata,
        "passed": overall["passed"] and all(v["passed"] for v in strata.values()),
        "additional_gate_original_gates_unchanged": True,
    }
