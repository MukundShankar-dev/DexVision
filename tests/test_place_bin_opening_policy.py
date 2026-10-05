"""V23 recipe boundaries and narrow held-object opening changes."""
from __future__ import annotations

import numpy as np

from dexvision.evaluation.place_bin_opening_selection import (
    load_selection_spec, selection_matrix,
)
from dexvision.evaluation.place_bin_opening_policy import bin_opening_matrix
from dexvision.evaluation.place_collision_preview import preview_place_action
from dexvision.learning.place_bin_opening_policy import (
    BinOpeningPlacePolicy, load_bin_opening_spec,
)
from dexvision.learning.place_supported_descent_policy import SupportedDescentPlacePolicy


POLICY = 'configs/level5/place_policy_bin_opening_v23.yaml'
SELECTION = 'configs/level5/place_bin_opening_selection_v23.yaml'


def test_v23_recipe_has_fresh_all_cell_selection():
    overlay, spec, inputs = load_bin_opening_spec(POLICY)
    protocol, _, _, _ = load_selection_spec(SELECTION)
    rows = selection_matrix(protocol, overlay, spec, inputs)
    assert len(rows) == 60
    assert len({row['cell_id'] for row in rows}) == 30
    other = (bin_opening_matrix(overlay, spec, inputs, 'validation')
             + bin_opening_matrix(overlay, spec, inputs, 'confirmation'))
    assert {row['seed'] for row in rows}.isdisjoint({row['seed'] for row in other})


def test_v23_only_changes_held_short_cylinder_and_puck_bin_opening(monkeypatch):
    monkeypatch.setattr(SupportedDescentPlacePolicy, '__call__',
                        lambda self, source, prior, phase=None: (prior.copy(), np.zeros(26)))
    policy = object.__new__(BinOpeningPlacePolicy)
    policy.bin_opening_recipe = load_bin_opening_spec(POLICY)[0]
    policy.index = {'entity.held.0': 0, 'entity.position_m.2': 1,
                    'goal.target_position.2': 2}
    policy.spec = {'workspace_bounds': {'min': [0., 0., 0.01]}}
    policy.latch = np.zeros(7)
    prior = np.zeros(27)
    prior[2] = 0.08
    source = np.array([1., 0.04, 0.02])
    for object_id, step in (('cylinder_short', 0.003), ('puck_heavy', 0.001)):
        policy.object_id = object_id
        policy.target_id = 'return_bin_left'
        action, _ = policy(source, prior)
        assert np.isclose(action[2], prior[2] - step)
        assert np.isclose(policy.latch[2], action[2])
        policy.target_id = 'setup_slot_b'
        assert np.array_equal(policy(source, prior)[0], prior)
    policy.object_id = 'block_small'
    policy.target_id = 'return_bin_left'
    assert np.array_equal(policy(source, prior)[0], prior)
    policy.object_id = 'puck_heavy'
    source[0] = 0.
    assert np.array_equal(policy(source, prior)[0], prior)


def test_v23_preview_dispatches_only_small_block_slot_b(monkeypatch):
    import dexvision.evaluation.place_collision_preview as module

    calls = []
    monkeypatch.setattr(module, '_collision_only_preview',
                        lambda *args: calls.append('collision') or args[1])
    monkeypatch.setattr(module, '_original_preview',
                        lambda *args: calls.append('original') or args[1])
    policy = object.__new__(BinOpeningPlacePolicy)
    action = np.zeros(27)
    policy.object_id, policy.target_id = 'block_small', 'setup_slot_b'
    preview_place_action(None, action, action, None, None, None, policy)
    policy.target_id = 'return_bin_left'
    preview_place_action(None, action, action, None, None, None, policy)
    assert calls == ['collision', 'original']
