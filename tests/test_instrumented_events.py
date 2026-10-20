"""Unavailable events cannot turn into passing qualification rates."""

from types import SimpleNamespace

import numpy as np

from dexvision.evaluation.instrumented_events import event_rates, reconstruct_events


def test_unknown_drop_and_release_events_fail_coverage():
    result = event_rates([{"slip_drop": None}], "pick")
    assert not result["gates"]["event_coverage"] and not result["gates"]["drop_rate"]
    result = event_rates(
        [{"premature_release": False, "nominally_placed": False, "settled": False}],
        "place",
    )
    assert result["settled_among_nominally_placed"] is None
    assert not result["gates"]["settled_placement"]
    result = event_rates([{"slip_drop": True}] + [{"slip_drop": False}] * 9, "pick")
    assert result["drop_rate"] == 0.1 and result["gates"]["drop_rate"]


def test_interrupted_pick_drop_reconstruction_only_forwards_saved_states(monkeypatch):
    data = SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1), time=0.0)
    forwards = []
    entity = SimpleNamespace(position=(0.0, 0.0, 0.1), linear_velocity=(0.0, 0.0, 0.0))

    def world():
        held = 0 < data.time < 0.05
        relation = SimpleNamespace(
            object_id="object",
            held_by="rh_palm" if held else None,
            supported_by=None if held else "table",
        )
        return SimpleNamespace(
            require_entity=lambda _: entity,
            relation_for=lambda _: relation,
            contacts=(("object", "rh_finger1"), ("object", "rh_finger2"))
            if held
            else (),
            relations=(relation,),
        )

    class Task:
        def __init__(self, **kwargs):
            self.env = SimpleNamespace(
                data=data,
                model=None,
                _mujoco=SimpleNamespace(
                    mj_forward=lambda *args: forwards.append(data.time)
                ),
            )
            self.workcell = SimpleNamespace(get_world_state=world)
            self.initial_world_state = SimpleNamespace(
                require_entity=lambda _: SimpleNamespace(position=(0.0, 0.0, 0.0))
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        "dexvision.evaluation.instrumented_events.WorkcellPilotTask", Task
    )
    record = {"row": dict(cell_id="cell", seed=1, variation={}, entity_id="object")}
    terminal = dict(
        minimum_height_m=0.05,
        maximum_speed_mps=0.1,
        maximum_retention_gap_samples=2,
        consecutive_samples=10,
    )
    arrays = dict(
        qpos=np.zeros((8, 1)),
        qvel=np.zeros((8, 1)),
        state_timestamps=np.arange(8) * 0.034,
    )
    assert reconstruct_events("pick", record, arrays, terminal) == {"slip_drop": True}
    assert len(forwards) == 7 and forwards[-1] == arrays["state_timestamps"][-1]
