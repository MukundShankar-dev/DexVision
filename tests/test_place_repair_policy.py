"""Additive place policy ownership, causal labels and fresh-reset contracts."""
from pathlib import Path

import numpy as np
import pytest
import torch

from dexvision.evaluation.place_repair_policy import repair_matrix
from dexvision.evaluation.place_selection import load_selection_spec, selection_matrix
from dexvision.evaluation.place_safety_selection import (
    load_selection_spec as load_safety_selection_spec,
    selection_matrix as safety_selection_matrix,
)
from dexvision.evaluation.place_slow_selection import (
    load_selection_spec as load_slow_selection_spec,
    selection_matrix as slow_selection_matrix,
)
from dexvision.evaluation.place_clearance_selection import (
    load_selection_spec as load_clearance_selection_spec,
    selection_matrix as clearance_selection_matrix,
)
from dexvision.evaluation.place_slot_clearance_selection import (
    load_selection_spec as load_slot_clearance_selection_spec,
    selection_matrix as slot_clearance_selection_matrix,
)
from dexvision.learning.place_cadence_policy import CadencePolicy, load_cadence_spec
from dexvision.learning.place_feedback_policy import FeedbackPolicy, load_feedback_spec
from dexvision.learning.place_guarded_policy import GuardedPolicy, load_guarded_spec
from dexvision.learning.place_repair_policy import load_repair_spec, repair_datasets
from dexvision.learning.place_safety_policy import SafetyPolicy, load_safety_spec
from dexvision.learning.place_slow_policy import SlowSafetyPolicy, load_slow_spec
from dexvision.learning.place_clearance_policy import (
    ClearancePolicy, FROZEN_CLEARANCE_M, load_clearance_spec,
)
from dexvision.learning.place_slot_clearance_policy import (
    SlotClearancePolicy, load_slot_clearance_spec,
)
from dexvision.learning.skill_datasets import feature_schema

CONFIG = Path('configs/level5/place_policy_repair_v2.yaml')


def test_place_repair_matrix_is_frozen_and_fresh():
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    spec, inputs = load_repair_spec(CONFIG)
    validation = repair_matrix(spec, inputs, 'validation')
    confirmation = repair_matrix(spec, inputs, 'confirmation')
    assert len(validation) == 48
    assert len(confirmation) == 60
    assert len({r['cell_id'] for r in confirmation}) == 30
    assert not {r['seed'] for r in validation} & {r['seed'] for r in confirmation}
    assert not {r['episode_id'] for r in validation} & {r['episode_id'] for r in confirmation}


def test_place_repair_source_has_only_causal_inputs_and_coupled_release_label():
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    spec, inputs = load_repair_spec(CONFIG)
    datasets = repair_datasets(spec, inputs)
    assert (len(datasets['train'].episodes), len(datasets['validation'].episodes)) == (140, 47)
    assert (len(datasets['train']), len(datasets['validation'])) == (15448, 4603)
    assert not any('expert_phase' in name for name in datasets['train'].normalization['ordered_feature_names'])
    for dataset in datasets.values():
        assert set(np.unique(dataset.targets[:, 8])) == {-1., 1.}
        assert np.all(dataset.masks[:, 8] == 1)
        assert np.all(dataset.masks[:, 6:8] == 0)
        assert np.all(dataset.masks[:, 9:] == 0)
        assert np.max(np.abs(dataset.targets[:, :6])) <= 1 + 1e-5


def test_place_feedback_waypoint_uses_current_object_position():
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    spec, _ = load_feedback_spec('configs/level5/place_policy_feedback_v4.yaml')
    policy = object.__new__(FeedbackPolicy)
    policy.spec = spec
    policy.object_id = 'block_small'
    policy.stage = 'raise'
    prior = np.array([-0.10, -0.09, 0.11])
    obj = np.array([-0.12, -0.09, 0.07])
    target = np.array([-0.07, 0.06, 0.01])
    raised, step = policy._waypoint(obj, target, prior, 'cuboid')
    assert step == spec['geometry']['transport_step_m']
    assert raised[2] > prior[2]
    policy.stage = 'planar'
    planar, _ = policy._waypoint(np.array([-0.12, -0.09, 0.12]), target, prior, 'cuboid')
    desired = target - np.asarray(spec['geometry']['release_goal_error_by_family_m']['cuboid'])
    assert np.allclose(planar[:2] - prior[:2], desired[:2] - obj[:2])
    policy.stage = 'lower'
    shifted_object = np.array([-0.10, 0.04, 0.08])
    lower, step = policy._waypoint(shifted_object, target, prior, 'cuboid')
    assert step == spec['geometry']['descent_step_m']
    assert np.allclose(lower - prior, desired - shifted_object)
    assert np.linalg.norm(np.asarray(spec['geometry']['release_goal_error_by_family_m']['cuboid'])) \
        < spec['geometry']['release_distance_by_family_m']['cuboid']


def test_place_cadence_matches_resolved_source_step_without_changing_release_geometry():
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    previous, _ = load_feedback_spec('configs/level5/place_policy_feedback_v4.yaml')
    spec, _ = load_cadence_spec('configs/level5/place_policy_cadence_v5.yaml')
    assert spec['geometry']['transport_step_m'] == 0.005
    assert previous['geometry']['transport_step_m'] == 0.002
    for key in ('release_goal_error_by_family_m', 'release_distance_by_family_m',
                'descent_step_m'):
        assert spec['geometry'][key] == previous['geometry'][key]
    policy = object.__new__(CadencePolicy)
    policy.spec = spec
    policy.object_id = 'block_small'
    policy.stage = 'planar'
    prior = np.array([-0.10, -0.09, 0.11])
    obj = np.array([-0.12, -0.09, 0.12])
    target = np.array([-0.07, 0.06, 0.01])
    waypoint, step = policy._waypoint(obj, target, prior, 'cuboid')
    assert step == 0.005
    desired = target - np.asarray(spec['geometry']['release_goal_error_by_family_m']['cuboid'])
    assert np.allclose(waypoint[:2] - prior[:2], desired[:2] - obj[:2])


def test_guarded_place_moves_while_held_even_when_learned_move_score_is_negative(monkeypatch):
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    spec, inputs = load_guarded_spec('configs/level5/place_policy_guarded_v6.yaml')
    assert spec['geometry']['release_distance_by_family_m']['cuboid'] == 0.0245
    assert max(spec['geometry']['release_distance_by_family_m'].values()) <= \
        spec['terminal']['release_distance_m']
    assert spec['release_step_by_family']['cylinder'] == 0.0125
    policy = object.__new__(GuardedPolicy)
    policy.spec, policy.inputs, policy.object_id = spec, inputs, 'block_small'
    policy.names = list(feature_schema(inputs.plan, inputs.skill)[0])
    policy.index = {name: index for index, name in enumerate(policy.names)}
    policy.plan, policy.latch, policy.stage = {}, None, 'planar'
    policy.model = lambda _: torch.tensor([[-1., 0., 0., 0., 0., 0., 0., 0., -1.]])
    monkeypatch.setattr('dexvision.learning.place_guarded_policy.place_features',
                        lambda source, names, spec: (np.zeros(9), []))
    monkeypatch.setattr('dexvision.learning.place_guarded_policy.decode_delta',
                        lambda rotation, prior, phase, plan: prior.copy())
    source = np.zeros(len(policy.names))
    source[[policy.index[f'entity.position_m.{j}'] for j in range(3)]] = [-.12, -.09, .12]
    source[[policy.index[f'goal.target_position.{j}'] for j in range(3)]] = [-.07, .06, .01]
    source[policy.index['entity.class.cuboid']] = 1
    source[policy.index['entity.held.0']] = 1
    prior = np.zeros(27)
    prior[:3] = [-.10, -.09, .11]
    action, prediction = policy(source, prior)
    assert prediction[0] < 0
    assert np.linalg.norm(action[:3] - prior[:3]) == pytest.approx(0.005)
    assert np.allclose(action[7:], prior[7:])


def test_place_selection_screen_covers_all_cells_without_using_confirmation():
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    protocol, spec, inputs = load_selection_spec('configs/level5/place_selection_v7.yaml')
    screen = selection_matrix(protocol, spec, inputs)
    confirmation = repair_matrix(spec, inputs, 'confirmation')
    assert len(screen) == 60
    assert len({row['cell_id'] for row in screen}) == 30
    assert {row['boundary'] for row in screen} == {False, True}
    assert not {row['seed'] for row in screen} & {row['seed'] for row in confirmation}


def test_place_safety_overlay_and_new_selection_stay_inside_frozen_gates():
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    overlay, spec, inputs = load_safety_spec('configs/level5/place_policy_safety_v8.yaml')
    assert overlay['command_object_lag_hold_m'] == 0.05
    assert overlay['cuboid_opening_step_nonplanar'] < overlay['cuboid_opening_step_planar']
    assert max(spec['geometry']['release_distance_by_family_m'].values()) <= \
        spec['terminal']['release_distance_m']
    protocol, source, data = load_safety_selection_spec(
        'configs/level5/place_safety_selection_v8.yaml')
    screen = safety_selection_matrix(protocol, source, data)
    confirmation = repair_matrix(source, data, 'confirmation')
    assert len(screen) == 60
    assert len({row['cell_id'] for row in screen}) == 30
    assert not {row['seed'] for row in screen} & {row['seed'] for row in confirmation}


def test_place_safety_overlay_holds_when_command_object_lag_grows(monkeypatch):
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    overlay, spec, inputs = load_safety_spec('configs/level5/place_policy_safety_v8.yaml')
    policy = object.__new__(SafetyPolicy)
    policy.overlay, policy.spec, policy.latch = overlay, spec, None
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    policy.index = {name: index for index, name in enumerate(names)}
    policy.initial_offset = np.array([0.03, 0., 0.])
    monkeypatch.setattr(GuardedPolicy, '__call__',
                        lambda self, source, prior, phase=None: (prior + 0.005, np.zeros(9)))
    source = np.zeros(len(names))
    source[[policy.index[f'entity.position_m.{j}'] for j in range(3)]] = [.01, 0., .1]
    source[[policy.index[f'goal.target_position.{j}'] for j in range(3)]] = [-.07, .06, .01]
    source[policy.index['entity.held.0']] = 1
    prior = np.zeros(27)
    prior[:3] = [.1, 0., .1]
    action, _ = policy(source, prior)
    assert np.array_equal(action, prior)


def test_place_slow_overlay_preserves_v8_geometry_and_freezes_fresh_selection(monkeypatch):
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    overlay, spec, inputs = load_slow_spec('configs/level5/place_policy_slow_v9.yaml')
    assert overlay['cuboid_opening_step_nonplanar'] == 0.00625
    assert overlay['cylinder_opening_step'] == 0.00625
    assert overlay['flat_puck_opening_step'] == 0.00625
    assert max(spec['geometry']['release_distance_by_family_m'].values()) <= \
        spec['terminal']['release_distance_m']

    def fake_parent_init(self, checkpoint_path, config):
        assert str(config) == overlay['parent_policy_recipe']
        self.overlay = {'cuboid_opening_step_planar': 0.025,
                        'cuboid_opening_step_nonplanar': 0.0125}
        self.spec = {'release_step_by_family': {'cylinder': 0.0125,
                                                'flat_puck': 0.0125}}

    monkeypatch.setattr(SafetyPolicy, '__init__', fake_parent_init)
    policy = SlowSafetyPolicy('unused.pt', 'configs/level5/place_policy_slow_v9.yaml')
    assert policy.overlay['cuboid_opening_step_planar'] == 0.025
    assert policy.overlay['cuboid_opening_step_nonplanar'] == 0.00625
    assert policy.spec['release_step_by_family'] == {'cylinder': 0.00625,
                                                      'flat_puck': 0.00625}

    protocol, source, data = load_slow_selection_spec(
        'configs/level5/place_slow_selection_v9.yaml')
    screen = slow_selection_matrix(protocol, source, data)
    confirmation = repair_matrix(source, data, 'confirmation')
    old, _, _ = load_safety_selection_spec('configs/level5/place_safety_selection_v8.yaml')
    previous = safety_selection_matrix(old, source, data)
    assert protocol['candidate_epochs'] == {'5001': [80], '5002': [20], '5003': [20]}
    assert len(screen) == 60
    assert len({row['cell_id'] for row in screen}) == 30
    assert not {row['seed'] for row in screen} & {row['seed'] for row in confirmation}
    assert not {row['seed'] for row in screen} & {row['seed'] for row in previous}


def test_place_clearance_latches_one_target_aware_raise_and_new_selection(monkeypatch):
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    overlay, spec, inputs = load_clearance_spec(
        'configs/level5/place_policy_clearance_v10.yaml')
    assert overlay['release_clearance_m'] == FROZEN_CLEARANCE_M
    protocol, _, _ = load_clearance_selection_spec(
        'configs/level5/place_clearance_selection_v10.yaml')
    screen = clearance_selection_matrix(protocol, spec, inputs)
    old, _, _ = load_slow_selection_spec('configs/level5/place_slow_selection_v9.yaml')
    exposed = slow_selection_matrix(old, spec, inputs)
    confirmation = repair_matrix(spec, inputs, 'confirmation')
    assert len(screen) == 60
    assert len({row['cell_id'] for row in screen}) == 30
    assert not {row['seed'] for row in screen} & {row['seed'] for row in exposed}
    assert not {row['seed'] for row in screen} & {row['seed'] for row in confirmation}

    policy = object.__new__(ClearancePolicy)
    policy.clearance = FROZEN_CLEARANCE_M
    calls = 0

    def latch_once(self, source, prior, phase=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            self.latch = prior[:7].copy()
        return prior.copy(), np.zeros(9)

    monkeypatch.setattr(SlowSafetyPolicy, '__call__', latch_once)
    prior = np.zeros(27)
    prior[2] = 0.1
    for object_id, target_id, raise_m in (
            ('block_small', 'inspection_pad', 0.0015),
            ('cylinder_short', 'return_bin_left', 0.003),
            ('cylinder_short', 'setup_slot_a', 0.0),
            ('puck_heavy', 'return_bin_right', 0.0),
            ('puck_heavy', 'setup_slot_b', 0.003)):
        calls = 0
        policy.latch = None
        policy.bind({'entity_id': object_id,
                     'typed_goal': {'target_id': target_id}})
        first, _ = policy(np.zeros(1), prior)
        second, _ = policy(np.zeros(1), first)
        assert np.isclose(first[2], prior[2] + raise_m)
        assert np.isclose(second[2], first[2])


def test_place_slot_clearance_removes_cuboid_slot_rise_and_keeps_fresh_boundary(monkeypatch):
    if not Path('datasets/level5-place-unassisted-v2.tar.gz').is_file():
        pytest.skip('Retrieve the immutable LFS place release first')
    overlay, spec, inputs = load_slot_clearance_spec(
        'configs/level5/place_policy_slot_clearance_v11.yaml')
    assert overlay['cuboid_placement_slot_clearance_m'] == 0.0
    protocol, _, _ = load_slot_clearance_selection_spec(
        'configs/level5/place_slot_clearance_selection_v11.yaml')
    screen = slot_clearance_selection_matrix(protocol, spec, inputs)
    older, _, _ = load_clearance_selection_spec(
        'configs/level5/place_clearance_selection_v10.yaml')
    previous = clearance_selection_matrix(older, spec, inputs)
    assert len(screen) == 60
    assert len({row['cell_id'] for row in screen}) == 30
    assert not {row['seed'] for row in screen} & {row['seed'] for row in previous}
    assert not {row['seed'] for row in screen} & {
        row['seed'] for row in repair_matrix(spec, inputs, 'confirmation')}

    def fake_parent_bind(self, row):
        self.release_raise_m = FROZEN_CLEARANCE_M['cuboid']['placement_slot']

    monkeypatch.setattr(ClearancePolicy, 'bind', fake_parent_bind)
    policy = object.__new__(SlotClearancePolicy)
    policy.slot_overlay = overlay
    policy.bind({'entity_id': 'block_small',
                 'typed_goal': {'target_id': 'setup_slot_b'}})
    assert policy.release_raise_m == 0.0
    policy.bind({'entity_id': 'block_small',
                 'typed_goal': {'target_id': 'return_bin_left'}})
    assert policy.release_raise_m == 0.0015
