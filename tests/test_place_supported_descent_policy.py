"""V22 recipe ownership and narrow supported opening behavior."""
from __future__ import annotations

import numpy as np

from dexvision.evaluation.place_supported_descent_selection import (
    load_selection_spec, selection_matrix,
)
from dexvision.learning.place_shadow_policy import ShadowPlacePolicy
from dexvision.learning.place_supported_descent_policy import (
    SupportedDescentPlacePolicy, load_supported_descent_spec,
)


POLICY = 'configs/level5/place_policy_supported_descent_v22.yaml'
SELECTION = 'configs/level5/place_supported_descent_selection_v22.yaml'


def test_v22_recipe_has_fresh_all_cell_selection():
    overlay, spec, inputs = load_supported_descent_spec(POLICY)
    protocol, _, _, _ = load_selection_spec(SELECTION)
    rows = selection_matrix(protocol, overlay, spec, inputs)
    assert len(rows) == 60
    assert len({row['cell_id'] for row in rows}) == 30
    assert {row['seed'] for row in rows}.isdisjoint({
        overlay['confirmation_seed_start'] + 100 * index + offset
        for index in range(30) for offset in range(2)})


def test_v22_only_changes_held_short_cylinder_bin_opening(monkeypatch):
    monkeypatch.setattr(ShadowPlacePolicy, '__call__',
        lambda self, source, prior, phase=None: (prior.copy(), np.zeros(26)))
    policy = object.__new__(SupportedDescentPlacePolicy)
    policy.supported_descent_recipe = {
        'supported_descent_object_id': 'cylinder_short',
        'supported_descent_target_ids': ['return_bin_left', 'return_bin_right'],
        'supported_descent_min_object_height_above_target_m': 0.008,
        'supported_descent_step_m': 0.002,
    }
    policy.index = {'entity.held.0': 0, 'entity.position_m.2': 1,
                    'goal.target_position.2': 2}
    policy.spec = {'workspace_bounds': {'min': [0., 0., 0.01]}}
    policy.object_id = 'cylinder_short'
    policy.target_id = 'return_bin_left'
    policy.latch = np.zeros(7)
    prior = np.zeros(27)
    prior[2] = 0.08
    source = np.array([1., 0.04, 0.02])
    action, _ = policy(source, prior)
    assert np.isclose(action[2], 0.078)
    assert np.isclose(policy.latch[2], 0.078)
    policy.target_id = 'inspection_pad'
    assert np.array_equal(policy(source, prior)[0], prior)
    policy.target_id = 'return_bin_right'
    source[0] = 0.
    assert np.array_equal(policy(source, prior)[0], prior)
