"""V13 release guard and fresh selection boundary."""
from pathlib import Path

import numpy as np
import pytest

from dexvision.evaluation.place_recovery_selection import load_selection_spec, selection_matrix
from dexvision.evaluation.place_repair_policy import repair_matrix
from dexvision.learning.place_adaptive_policy import AdaptivePlacePolicy
from dexvision.learning.place_recovery_policy import RecoveryPlacePolicy, load_recovery_spec


POLICY = Path('configs/level5/place_policy_recovery_v13.yaml')
SELECTION = Path('configs/level5/place_recovery_selection_v13.yaml')


def test_recovery_recipe_and_fresh_matrix():
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    overlay, spec, inputs = load_recovery_spec(POLICY)
    protocol, _, _ = load_selection_spec(SELECTION)
    matrix = selection_matrix(protocol, spec, inputs)
    assert overlay['confirmation_seed_start'] == protocol['confirmation']['seed_start']
    assert protocol['selection_screen']['seed_start'] == 1100260900
    assert len(matrix) == 60 and len({row['cell_id'] for row in matrix}) == 30
    assert not {row['seed'] for row in matrix} & {
        row['seed'] for row in repair_matrix(spec, inputs, 'confirmation')}


def test_negative_drift_fallback_respects_distance_and_motion(monkeypatch):
    def parent_call(self, source, prior, phase=None):
        return prior.copy(), np.zeros(9)

    monkeypatch.setattr(AdaptivePlacePolicy, '__call__', parent_call)
    policy = object.__new__(RecoveryPlacePolicy)
    policy.recovery_controls = {
        'negative_lateral_gap_threshold_m': -0.003,
        'early_release_distance_m': 0.024,
        'early_release_downward_speed_mps': -0.02,
        'early_opening_step': 0.0125,
    }
    policy.adaptive_controls = {'short_grasp_offset_threshold_m': 0.053}
    policy.spec = {'release_step_by_family': {'flat_puck': 0.00625},
                   'open_targets': np.zeros(20), 'closed_targets': np.ones(20)}
    policy.index = {name: index for index, name in enumerate((
        'entity.position_m.0', 'entity.position_m.1', 'entity.position_m.2',
        'goal.target_position.0', 'goal.target_position.1', 'goal.target_position.2',
        'entity.held.0', 'entity.linear_velocity_mps.2'))}
    policy.object_id = 'puck_heavy'
    policy.target_id = 'inspection_pad'
    policy.initial_offset = np.array([0., 0., 0.049])
    policy.release_raise_m = 0.01
    policy.latch = None
    prior = np.zeros(27)
    prior[1] = -0.004
    prior[7:] = 1.0
    source = np.array([0., 0., 0.02, 0., 0., 0., 1., -0.03])

    # Beyond the frozen physical release limit: no early opening.
    source[2] = 0.026
    action, _ = policy(source, prior)
    assert np.array_equal(action, prior) and policy.latch is None
    source[2] = 0.023
    source[7] = 0.
    action, _ = policy(source, prior)
    assert np.array_equal(action, prior) and policy.latch is None
    source[7] = -0.03
    action, _ = policy(source, prior)
    assert policy.latch is not None
    assert action[2] == pytest.approx(0.01)
    assert np.any(action[7:] != prior[7:])


def test_centered_grasp_does_not_trigger_fallback(monkeypatch):
    monkeypatch.setattr(AdaptivePlacePolicy, '__call__',
                        lambda self, source, prior, phase=None: (prior.copy(), np.zeros(9)))
    policy = object.__new__(RecoveryPlacePolicy)
    policy.recovery_controls = {
        'negative_lateral_gap_threshold_m': -0.003,
        'early_release_distance_m': 0.024,
        'early_release_downward_speed_mps': -0.02,
        'early_opening_step': 0.0125,
    }
    policy.adaptive_controls = {'short_grasp_offset_threshold_m': 0.053}
    policy.spec = {'release_step_by_family': {'flat_puck': 0.00625}}
    policy.index = {'entity.position_m.1': 0}
    policy.object_id = 'puck_heavy'
    policy.target_id = 'inspection_pad'
    policy.initial_offset = np.array([0., 0., 0.049])
    policy.latch = None
    prior = np.zeros(27)
    policy(np.array([0.]), prior)
    assert policy.latch is None
    assert policy.spec['release_step_by_family']['flat_puck'] == 0.00625
