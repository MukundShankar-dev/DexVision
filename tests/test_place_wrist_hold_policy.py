"""V24 reset ownership and source-entry wrist hold scope."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dexvision.evaluation.place_wrist_hold_policy import wrist_hold_matrix
from dexvision.evaluation.place_wrist_hold_selection import load_selection_spec, selection_matrix
from dexvision.evaluation.place_wrist_hold_preview import preview_place_action


POLICY = 'configs/level5/place_policy_wrist_hold_v24.yaml'
SELECTION = 'configs/level5/place_wrist_hold_selection_v24.yaml'


def test_v24_selection_has_all_cells_and_fresh_reset_boundaries():
    protocol, overlay, spec, inputs = load_selection_spec(SELECTION)
    rows = selection_matrix(protocol, overlay, spec, inputs)
    other = (wrist_hold_matrix(overlay, spec, inputs, 'validation')
             + wrist_hold_matrix(overlay, spec, inputs, 'confirmation'))
    assert len(rows) == 60
    assert len({row['cell_id'] for row in rows}) == 30
    assert {row['seed'] for row in rows}.isdisjoint({row['seed'] for row in other})


def test_v24_hold_uses_setup_command_and_preserves_latched_release(monkeypatch):
    import dexvision.evaluation.place_wrist_hold_preview as module

    monkeypatch.setattr(module, '_parent_preview', lambda *args: args[1])
    policy = SimpleNamespace(object_id='block_large', target_id='return_bin_right',
        wrist_hold_recipe={'wrist_hold_cells': [['block_large', 'return_bin_right']]},
        entry_quaternion=None, latch=None)
    prior = np.zeros(27)
    prior[3:7] = [1., 0., 0., 0.]
    action = prior.copy()
    action[3:7] = [0.9, 0.1, 0.2, 0.3]
    held = preview_place_action(None, action, prior, None, None, None, policy)
    assert np.array_equal(held[3:7], prior[3:7])
    assert np.array_equal(held[:3], action[:3])
    assert np.array_equal(held[7:], action[7:])
    prior[3:7] = action[3:7]
    held = preview_place_action(None, action, prior, None, None, None, policy)
    assert np.array_equal(held[3:7], [1., 0., 0., 0.])
    policy.latch = np.ones(7)
    assert np.array_equal(preview_place_action(None, action, prior, None, None, None, policy), action)


def test_v24_other_cells_retain_parent_preview(monkeypatch):
    import dexvision.evaluation.place_wrist_hold_preview as module

    marker = np.ones(27)
    monkeypatch.setattr(module, '_parent_preview', lambda *args: marker)
    policy = SimpleNamespace(object_id='block_large', target_id='return_bin_left',
        wrist_hold_recipe={'wrist_hold_cells': [['block_large', 'return_bin_right']]},
        entry_quaternion=None, latch=None)
    result = preview_place_action(None, np.zeros(27), np.zeros(27), None, None, None, policy)
    assert result is marker
    assert policy.entry_quaternion is None
