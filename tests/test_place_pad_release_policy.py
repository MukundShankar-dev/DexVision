"""V25 release tolerance and independent qualification resets."""
from __future__ import annotations

import numpy as np

from dexvision.evaluation.place_pad_release_policy import pad_release_matrix
from dexvision.evaluation.place_pad_release_selection import load_selection_spec, selection_matrix
from dexvision.learning.place_pad_release_policy import PadReleasePlacePolicy, load_pad_release_spec
from dexvision.learning.place_wrist_hold_policy import WristHoldPlacePolicy


POLICY = 'configs/level5/place_policy_pad_release_v25.yaml'
SELECTION = 'configs/level5/place_pad_release_selection_v25.yaml'


def test_v25_keeps_release_bound_and_fresh_all_cell_matrices():
    overlay, spec, inputs = load_pad_release_spec(POLICY)
    protocol, _, _, _ = load_selection_spec(SELECTION)
    assert overlay['pad_release_controls']['distance_m'] == spec[
        'geometry']['release_distance_by_family_m']['cuboid']
    rows = selection_matrix(protocol, overlay, spec, inputs)
    other = (pad_release_matrix(overlay, spec, inputs, 'validation')
             + pad_release_matrix(overlay, spec, inputs, 'confirmation'))
    assert len(rows) == 60 and len({row['cell_id'] for row in rows}) == 30
    assert {row['seed'] for row in rows}.isdisjoint(row['seed'] for row in other)


def test_v25_pad_trigger_requires_held_object_inside_original_bound(monkeypatch):
    monkeypatch.setattr(WristHoldPlacePolicy, '__call__',
                        lambda self, source, prior, phase=None: (prior.copy(), np.zeros(26)))
    overlay, spec, _ = load_pad_release_spec(POLICY)
    policy = object.__new__(PadReleasePlacePolicy)
    policy.pad_release_controls = overlay['pad_release_controls']
    policy.spec = spec
    policy.release_raise_m = 0.0015
    policy.object_id, policy.target_id = 'block_large', 'inspection_pad'
    policy.index = {'entity.held.0': 0,
                    **{f'entity.position_m.{j}': 1 + j for j in range(3)},
                    **{f'goal.target_position.{j}': 4 + j for j in range(3)}}
    prior = np.zeros(27)
    prior[2] = 0.07
    prior[7:] = spec['closed_targets']
    source = np.array([1., 0.0246, 0., 0., 0., 0., 0.])
    policy.latch = None
    assert np.array_equal(policy(source, prior)[0], prior)
    source[1] = 0.0244
    source[0] = 0.
    assert np.array_equal(policy(source, prior)[0], prior)
    source[0] = 1.
    action, _ = policy(source, prior)
    assert np.isclose(action[2], prior[2] + policy.release_raise_m)
    assert np.linalg.norm(action[7:] - prior[7:]) > 0
    assert np.array_equal(action[:2], prior[:2])
    assert np.array_equal(action[3:7], prior[3:7])
