"""Context labels never fit; weighting uses original physical source ownership."""

import numpy as np


def test_phase_mass_and_physical_source_invariance():
    from dexvision.learning.perception_balanced_examples import balanced_examples

    def row(source, phase, primary=True):
        return np.zeros(706), np.zeros(3), source, phase, primary

    entries = (
        [row("a", "lift")] * 50
        + [row("b", "lift")]
        + [row("a", "stabilize")]
        + [row("b", "stabilize")] * 70
        + [row("c", "rest", False)] * 30
    )
    _, _, w = balanced_examples(entries)
    assert np.isclose(w[:50].sum(), w[50])
    assert np.isclose(w[51], w[52:122].sum())
    assert np.isclose(w[:51].sum(), w[51:122].sum())
    assert np.isclose(w[:51].sum(), 4 * w[122:].sum())


def test_raw_cache_preserves_provenance_and_rejects_changed_base(tmp_path):
    import pytest
    from dexvision.learning.perception_balanced_phase import (
        save_raw_cache,
        read_raw_cache,
        verify_base_cache,
    )

    entries = [(np.zeros(706), np.array([0.0, 0.0, 0.1]), "physical", "lift", True)]
    p = tmp_path / "raw.npz"
    save_raw_cache(p, entries)
    actual = read_raw_cache(p)
    assert actual[0][2:] == entries[0][2:]
    verify_base_cache(
        actual, (np.zeros((1, 706)), np.array([[0.0, 0.0, 0.1]]), np.ones(1))
    )
    with pytest.raises(ValueError, match="exists"):
        save_raw_cache(p, entries)
    with pytest.raises(ValueError, match="changed"):
        verify_base_cache(actual, (np.zeros((1, 706)), np.zeros((1, 3)), np.ones(1)))


def recovery_fixture():
    from copy import deepcopy

    cal = {
        "intrinsics": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "camera_from_world": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    row = {
        "split": "train",
        "source_episode_id": "window",
        "timestamp": 0.17,
        "source_frame_index": 5,
        "phase_window_query": True,
        "evaluation_only_source_owner": {
            "entity_id": "cube",
            "evaluation_only_parent_episode": "physical",
            "phase_intervals": [
                {"start_frame": 5, "end_frame": 6, "phase": "stabilize"}
            ],
        },
        "annotations": [
            {
                "visible": True,
                "class_id": "rigid_cuboid",
                "object_id": "cube",
                "pose_6d": {"translation_m": [0.0, 0.0, 0.02]},
            }
        ],
    }
    context = deepcopy(row)
    context["phase_window_query"] = False
    context["timestamp"] = 0.0
    context["annotations"][0]["pose_6d"]["translation_m"] = [999.0] * 3
    val = deepcopy(context)
    val["split"] = "validation"
    val["phase_window_query"] = True
    x = np.zeros((1, 706))
    x[0, :2] = 0.5
    y = np.array([[0.0, 0.0, 0.2]])
    return [val, context, row], x, y, cal


def test_recover_only_exact_train_query_provenance():
    from dexvision.learning.perception_balanced_examples import recover_entries

    rows, x, y, cal = recovery_fixture()
    values = recover_entries(rows, x, y, cal)
    assert values[0][2:] == ("physical", "stabilize", True)
    np.testing.assert_array_equal(values[0][0], x[0])
    np.testing.assert_array_equal(values[0][1], y[0])


def test_recovery_refuses_missing_coverage_and_changed_target():
    import pytest
    from dexvision.learning.perception_balanced_examples import recover_entries

    rows, x, y, cal = recovery_fixture()
    with pytest.raises(ValueError, match="coverage"):
        recover_entries(rows, np.r_[x, x], np.r_[y, y], cal)
    y[0, 2] = 0.5
    with pytest.raises(ValueError, match="target/order"):
        recover_entries(rows, x, y, cal)
