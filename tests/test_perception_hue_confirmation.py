"""Fresh late strata need sufficient independent sources, not conditional omission."""

from dexvision.evaluation.perception_hue_confirmation import fresh_late_evidence


def report_and_rows(sources=4):
    group = {
        "visible_eligible_instances": 20,
        "detections_matched_iou_0_5": 20,
        "position_hypothesis_count": 20,
        "position_error_median_m": 0.002,
        "position_error_p95_m": 0.007,
    }
    report = {
        "primary_class_phase": {
            "rigid_cuboid/" + p: dict(group) for p in ("release", "settle")
        }
    }
    rows = [
        {
            "episode": str(i),
            "primary_object": True,
            "eligible": True,
            "class_id": "rigid_cuboid",
            "phase_stratum": p,
        }
        for p in ("release", "settle")
        for i in range(sources)
    ]
    return report, rows


def test_fresh_late_requires_both_phases_and_four_sources():
    r, rows = report_and_rows()
    assert fresh_late_evidence(r, rows)["passed"]
    r, rows = report_and_rows(3)
    assert not fresh_late_evidence(r, rows)["passed"]
    r, rows = report_and_rows()
    r["primary_class_phase"].pop("rigid_cuboid/settle")
    assert not fresh_late_evidence(r, rows)["passed"]


def test_fresh_late_recall_cannot_be_hidden_by_pooled_metrics():
    r, rows = report_and_rows()
    r["primary_class_phase"]["rigid_cuboid/settle"].update(
        visible_eligible_instances=9,
        detections_matched_iou_0_5=7,
        position_hypothesis_count=7,
    )
    assert not fresh_late_evidence(r, rows)["passed"]
