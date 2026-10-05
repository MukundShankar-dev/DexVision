"""Causal V12 place decoder rules and fresh selection boundary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dexvision.evaluation.place_adaptive_selection import load_selection_spec, selection_matrix
from dexvision.evaluation.place_repair_policy import repair_matrix
from dexvision.learning.place_adaptive_policy import AdaptivePlacePolicy, load_adaptive_spec
from dexvision.learning.place_slot_clearance_policy import SlotClearancePolicy


POLICY_RECIPE = Path('configs/level5/place_policy_adaptive_v12.yaml')
SELECTION_RECIPE = Path('configs/level5/place_adaptive_selection_v12.yaml')
CONTROLS = {
    'short_grasp_offset_threshold_m': 0.053,
    'cuboid_left_bin_slow_gap_m': 0.03,
    'cuboid_left_bin_slow_velocity_mps': -0.02,
    'cuboid_left_bin_slow_opening_step': 0.00625,
    'cuboid_left_bin_fast_opening_step': 0.0125,
}


def test_adaptive_recipe_preserves_parent_and_one_time_confirmation():
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    overlay, spec, inputs = load_adaptive_spec(POLICY_RECIPE)
    protocol, _, _ = load_selection_spec(SELECTION_RECIPE)
    matrix = selection_matrix(protocol, spec, inputs)
    assert overlay['adaptive_controls'] == CONTROLS
    assert protocol['candidate_epochs'] == {'5001': [80], '5002': [20], '5003': [20]}
    assert protocol['selection_screen']['seed_start'] == 1080260900
    assert len(matrix) == 60 and len({row['cell_id'] for row in matrix}) == 30
    assert not {row['seed'] for row in matrix} & {
        row['seed'] for row in repair_matrix(spec, inputs, 'confirmation')}


def test_short_grasp_freezes_only_heavy_puck_pad_descent(monkeypatch):
    def parent_waypoint(self, object_position, target_position, prior, family):
        return np.array([prior[0], prior[1], target_position[2] + object_position[2]]), 0.001

    monkeypatch.setattr(SlotClearancePolicy, '_waypoint', parent_waypoint)
    policy = object.__new__(AdaptivePlacePolicy)
    policy.adaptive_controls = CONTROLS
    policy.object_id = 'puck_heavy'
    policy.target_id = 'inspection_pad'
    policy.initial_offset = np.array([0., 0., 0.049])
    policy.stage = 'lower'
    policy.fixed_descent_z = None
    first, _ = policy._waypoint(np.array([0., 0., 0.07]), np.array([0., 0., 0.01]),
                                np.array([0., 0., 0.12]), 'flat_puck')
    second, _ = policy._waypoint(np.array([0., 0., 0.04]), np.array([0., 0., 0.01]),
                                 np.array([0., 0., 0.09]), 'flat_puck')
    assert first[2] == second[2] == pytest.approx(0.08)
    policy.initial_offset[2] = 0.057
    third, _ = policy._waypoint(np.array([0., 0., 0.04]), np.array([0., 0., 0.01]),
                                np.array([0., 0., 0.09]), 'flat_puck')
    assert third[2] == pytest.approx(0.05)
    policy.initial_offset[2] = 0.049
    policy.target_id = 'return_bin_left'
    fourth, _ = policy._waypoint(np.array([0., 0., 0.04]), np.array([0., 0., 0.01]),
                                 np.array([0., 0., 0.09]), 'flat_puck')
    assert fourth[2] == pytest.approx(0.05)


def test_left_bin_opening_uses_only_pre_action_geometry(monkeypatch):
    def parent_call(self, source, prior, phase=None):
        return np.asarray(prior).copy(), np.zeros(9)

    monkeypatch.setattr(SlotClearancePolicy, '__call__', parent_call)
    policy = object.__new__(AdaptivePlacePolicy)
    policy.adaptive_controls = CONTROLS
    policy.index = {'entity.position_m.2': 0, 'entity.linear_velocity_mps.2': 1}
    policy.overlay = {'cuboid_opening_step_nonplanar': 0.00625}
    policy.object_id = 'block_small'
    policy.target_id = 'return_bin_left'
    policy.latch = None
    prior = np.zeros(27)
    prior[2] = 0.08
    policy(np.array([0.045, -0.03]), prior)
    assert policy.overlay['cuboid_opening_step_nonplanar'] == 0.00625
    policy(np.array([0.06, 0.0]), prior)
    assert policy.overlay['cuboid_opening_step_nonplanar'] == 0.0125
    policy.latch = np.zeros(7)
    policy(np.array([0.045, -0.03]), prior)
    assert policy.overlay['cuboid_opening_step_nonplanar'] == 0.0125
    policy.target_id = 'return_bin_right'
    policy.latch = None
    policy.overlay['cuboid_opening_step_nonplanar'] = 0.00625
    policy(np.array([0.06, 0.0]), prior)
    assert policy.overlay['cuboid_opening_step_nonplanar'] == 0.00625
