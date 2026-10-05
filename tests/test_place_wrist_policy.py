"""V14 setup-wrist guard and untouched confirmation boundary."""
from pathlib import Path

import numpy as np
import pytest

from dexvision.evaluation.place_repair_policy import repair_matrix
from dexvision.evaluation.place_wrist_selection import load_selection_spec, selection_matrix
from dexvision.learning.place_recovery_policy import RecoveryPlacePolicy
from dexvision.learning.place_wrist_policy import (
    FROZEN_WRIST_CONTROLS, WristPlacePolicy, load_wrist_spec,
)


POLICY = Path('configs/level5/place_policy_wrist_v14.yaml')
SELECTION = Path('configs/level5/place_wrist_selection_v14.yaml')


def test_wrist_recipe_uses_new_selection_resets():
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    overlay, spec, inputs = load_wrist_spec(POLICY)
    protocol, _, _ = load_selection_spec(SELECTION)
    matrix = selection_matrix(protocol, spec, inputs)
    assert overlay['wrist_controls'] == FROZEN_WRIST_CONTROLS
    assert protocol['candidate_epochs'] == {'5001': [80], '5002': [20], '5003': [20]}
    assert protocol['selection_screen']['seed_start'] == 1120260900
    assert len(matrix) == 60 and len({row['cell_id'] for row in matrix}) == 30
    assert not {row['seed'] for row in matrix} & {
        row['seed'] for row in repair_matrix(spec, inputs, 'confirmation')}


@pytest.mark.parametrize(('object_id', 'target_id', 'entry_w', 'initial_z', 'proposed_w', 'freeze'), [
    ('block_large', 'return_bin_left', .83, .05, .84, True),
    ('block_large', 'return_bin_left', .79, .05, .78, False),
    ('cylinder_short', 'setup_slot_b', .76, .05, .759, True),
    ('cylinder_short', 'setup_slot_b', .76, .05, .761, False),
    ('puck_heavy', 'setup_slot_a', .76, .049, .761, True),
    ('puck_heavy', 'setup_slot_a', .76, .057, .759, True),
    ('puck_heavy', 'setup_slot_a', .76, .057, .761, False),
    ('puck_heavy', 'setup_slot_b', .76, .049, .759, False),
])
def test_wrist_guard_uses_only_entry_action_and_measured_grasp(
        monkeypatch, object_id, target_id, entry_w, initial_z, proposed_w, freeze):
    def parent_call(self, source, prior, phase=None):
        action = prior.copy()
        action[3:7] = [proposed_w, .6, .1, -.1]
        return action, np.zeros(9)

    monkeypatch.setattr(RecoveryPlacePolicy, '__call__', parent_call)
    policy = object.__new__(WristPlacePolicy)
    policy.wrist_controls = FROZEN_WRIST_CONTROLS
    policy.object_id = object_id
    policy.target_id = target_id
    policy.initial_offset = np.array([0., 0., initial_z])
    policy.freeze_wrist = None
    prior = np.zeros(27)
    prior[3:7] = [entry_w, .5, 0., 0.]
    action, _ = policy(np.zeros(1), prior)
    assert policy.freeze_wrist is freeze
    assert np.array_equal(action[3:7], prior[3:7]) is freeze
    if freeze:
        followup, _ = policy(np.zeros(1), prior)
        assert np.array_equal(followup[3:7], prior[3:7])
