"""V15 state-feedback safety and opening boundary."""
from pathlib import Path

import numpy as np
import pytest

from dexvision.evaluation.place_physical_selection import load_selection_spec, selection_matrix
from dexvision.evaluation.place_repair_policy import repair_matrix
from dexvision.learning.place_physical_policy import (
    FROZEN_PHYSICAL_CONTROLS, PhysicalPlacePolicy, load_physical_spec,
)
from dexvision.learning.place_wrist_policy import WristPlacePolicy


POLICY = Path('configs/level5/place_policy_physical_v15.yaml')
SELECTION = Path('configs/level5/place_physical_selection_v15.yaml')


def test_physical_recipe_reserves_a_new_selection_matrix():
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    overlay, spec, inputs = load_physical_spec(POLICY)
    protocol, _, _ = load_selection_spec(SELECTION)
    matrix = selection_matrix(protocol, spec, inputs)
    assert overlay['physical_controls'] == FROZEN_PHYSICAL_CONTROLS
    assert protocol['selection_screen']['seed_start'] == 1140260900
    assert protocol['candidate_epochs'] == {'5001': [80], '5002': [20], '5003': [20]}
    assert len(matrix) == 60 and len({row['cell_id'] for row in matrix}) == 30
    assert not {row['seed'] for row in matrix} & {
        row['seed'] for row in repair_matrix(spec, inputs, 'confirmation')}


def _synthetic_policy(monkeypatch, object_id, target_id):
    monkeypatch.setattr(WristPlacePolicy, '__call__',
                        lambda self, source, prior, phase=None: (prior.copy(), np.zeros(9)))
    policy = object.__new__(PhysicalPlacePolicy)
    policy.physical_controls = FROZEN_PHYSICAL_CONTROLS
    policy.object_id, policy.target_id = object_id, target_id
    policy.initial_offset = np.array([0., 0., 0.05])
    policy.index = {'entity.linear_velocity_mps.2': 0, 'entity.position_m.2': 1,
                    'entity.position_m.1': 2, 'entity.held.0': 3}
    policy.spec = {'release_step_by_family': {'cylinder': 0.00625, 'flat_puck': 0.00625}}
    policy.latch = None
    return policy


def test_rising_block_gets_bounded_inboard_lift_only_while_held(monkeypatch):
    policy = _synthetic_policy(monkeypatch, 'block_large', 'return_bin_left')
    prior = np.zeros(27)
    prior[:3] = [0.1, 0.1, 0.09]
    source = np.array([0.02, 0.07, 0.11, 1.0])
    action, _ = policy(source, prior)
    assert np.allclose(action[:3], [0.1, 0.098, 0.093])
    source[3] = 0.0
    action, _ = policy(source, prior)
    assert np.array_equal(action, prior)
    source[3] = 1.0
    source[0] = -0.02
    action, _ = policy(source, prior)
    assert np.array_equal(action, prior)


def test_opening_choice_uses_pre_action_velocity_and_grasp(monkeypatch):
    policy = _synthetic_policy(monkeypatch, 'cylinder_short', 'return_bin_left')
    prior = np.zeros(27)
    prior[3] = 0.71
    source = np.array([-0.026, 0.06, 0.1, 1.0])
    policy(source, prior)
    assert policy.spec['release_step_by_family']['cylinder'] == 0.0125
    source[0] = -0.03
    policy(source, prior)
    assert policy.spec['release_step_by_family']['cylinder'] == 0.00625
    policy.object_id, policy.target_id = 'puck_heavy', 'setup_slot_a'
    policy.initial_offset[2] = 0.049
    policy(source, prior)
    assert policy.spec['release_step_by_family']['flat_puck'] == 0.00625
    policy.initial_offset[2] = 0.057
    source[0] = -0.015
    policy(source, prior)
    assert policy.spec['release_step_by_family']['flat_puck'] == 0.0125
