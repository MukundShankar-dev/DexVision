"""Additional phase coverage cannot pass vacuously or relax the original limits."""

import pytest
from dexvision.evaluation.perception_phase_queries import phase_query_pass


def good():
    return {
        "visible_queries": 100,
        "visible_physical_source_episodes": 5,
        "matches": 100,
        "poses": 100,
        "median_m": 0.005,
        "p95_m": 0.015,
    }


def test_phase_query_gate_supports_original_metric_bounds():
    assert phase_query_pass(good())


@pytest.mark.parametrize(
    "key,value",
    [
        ("visible_queries", 7),
        ("visible_physical_source_episodes", 3),
        ("matches", 94),
        ("poses", 98),
        ("median_m", 0.01001),
        ("p95_m", 0.02501),
        ("median_m", None),
    ],
)
def test_unsupported_or_failing_phase_query_gate_refuses(key, value):
    result = good()
    result[key] = value
    assert not phase_query_pass(result)


def test_freeze_requires_additional_phase_coverage_before_a_new_test(
    tmp_path, monkeypatch
):
    import json
    import yaml
    import dexvision.evaluation.perception_phase_confirmation as module

    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({"selected_recipe": "fixed"}))
    (tmp_path / "validation.json").write_text(
        json.dumps({"additional_phase_validation": {"passed": False}})
    )
    monkeypatch.setattr(
        module, "selected_context", lambda _: ({"model_dir": str(tmp_path)}, {})
    )
    monkeypatch.setattr(module, "strict_validation", lambda _: True)
    monkeypatch.setattr(
        module,
        "freeze_context_confirmation",
        lambda *_: pytest.fail("No test before sufficient phase validation"),
    )
    with pytest.raises(ValueError, match="dense stabilization"):
        module.freeze_phase_confirmation(p, tmp_path / "out")
