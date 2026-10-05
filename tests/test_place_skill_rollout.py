"""Physical place preconditions, outcomes and frozen entry replay."""
from __future__ import annotations

from types import SimpleNamespace

from dexvision.evaluation.pick_policy import physical_evidence_world
from dexvision.evaluation.place_policy import (
    PlaceEvidence, create_place_task, place_preconditions, place_scenarios, read_place_entries,
    summarize_place,
)
from dexvision.learning.place_policy import load_place_spec


class World:
    def __init__(self, *, held, supported=None, x=0., speed=0., age=0.):
        self.timestamp = 1.
        self.object = SimpleNamespace(object_id='block_small', position=(x, 0., 0.),
            linear_velocity=(speed, 0., 0.), angular_velocity=(0., 0., 0.),
            timestamp=1. - age, confidence=1.)
        self.target = SimpleNamespace(object_id='setup_slot_b', position=(0., 0., 0.),
            timestamp=1., confidence=1.)
        self.relations = [SimpleNamespace(object_id='block_small', held_by='rh_palm' if held else None,
            supported_by=supported)]

    def require_entity(self, object_id):
        return self.object if object_id == 'block_small' else self.target

    def relation_for(self, object_id):
        assert object_id == 'block_small'
        return self.relations[0]


def test_place_rejects_wrong_multiple_missing_and_stale_held_state():
    row = {'typed_goal': {'object_id': 'block_small', 'target_id': 'setup_slot_b'},
        'entity_id': 'block_small'}
    assert all(place_preconditions(World(held=True), row).values())
    assert not place_preconditions(World(held=False), row)['exactly_one_held']
    extra = World(held=True)
    extra.relations.append(SimpleNamespace(object_id='puck_light', held_by='rh_palm', supported_by=None))
    assert not place_preconditions(extra, row)['exactly_one_held']
    assert not place_preconditions(World(held=True, age=0.1), row)['fresh_observation']
    assert not place_preconditions(World(held=True, supported='table'), row)['unsupported']
    assert not place_preconditions(World(held=True), {**row, 'entity_id': 'puck_light'})['correct_object']


def test_premature_release_and_unstable_post_release_are_failures():
    spec = {'minimum_transport_displacement_m': 0.04, 'release_distance_m': 0.025,
        'target_distance_m': 0.025, 'maximum_linear_speed_mps': 0.02,
        'maximum_angular_speed_radps': 0.2, 'consecutive_samples': 10}
    frozen = SimpleNamespace(values={}, success=True)
    early = PlaceEvidence('block_small', 'setup_slot_b', spec, (0.1, 0., 0.))
    assert early.update(World(held=False, x=0.08), frozen)['premature_release']
    assert not early.update(World(held=False, x=0., supported='table'), frozen)['success']
    good = PlaceEvidence('block_small', 'setup_slot_b', spec, (0.1, 0., 0.))
    good.update(World(held=True, x=0.01), frozen)
    assert good.update(World(held=False, x=0., supported='table', speed=0.1), frozen)['settle_steps'] == 0
    for _ in range(9):
        assert not good.update(World(held=False, x=0., supported='table'), frozen)['success']
    assert good.update(World(held=False, x=0., supported='table'), frozen)['success']
    assert good.update(World(held=False, x=0., supported=None), frozen)['bounced']
    assert not good.update(World(held=False, x=0., supported='table'), frozen)['success']


def test_full_matrix_summary_keeps_premature_release_failures():
    _, inputs = load_place_spec('configs/level5/place_policy_v1.yaml')
    results = []
    for index, (row, _, _, condition) in enumerate(place_scenarios(inputs, 'validation')):
        early = index < 14
        results.append({'episode_id': row['episode_id'], 'render_condition': condition,
            'source': row['source'], 'session': row['recording_session_id'],
            'object_family': row['entity_id'].split('_')[0], 'object_instance': row['entity_id'],
            'target_id': row['typed_goal']['target_id'], 'target_type': 'placement_slot',
            'cell_id': row['cell_id'], 'boundary_class': 'nominal',
            'terminal_result': 'failed' if early else 'succeeded',
            'terminal_reason': 'premature_release' if early else 'physical_settled_placement_passed',
            'premature_release': early, 'nominally_placed': not early, 'settled': not early,
            'bounced': False, 'transported': not early, 'released': True,
            'final_metrics': {'unwanted_contact_count': 0},
            'safety_counters': dict.fromkeys(('invalid_actions', 'workspace_violations',
                'joint_limit_violations', 'object_workspace_violations'), 0),
            'mean_action_jerk': 0., 'normalized_terminal_error': 0.})
    report = summarize_place(results, inputs, 'validation')
    assert report['rollout_count'] == 120
    assert report['premature_release_rate'] > 0.1
    assert not report['gates']['premature_release_rate']


def test_frozen_place_entry_replays_to_a_verified_held_object():
    _, inputs = load_place_spec('configs/level5/place_policy_v1.yaml')
    row, seed, frame, condition = place_scenarios(inputs, 'validation')[0]
    metadata, prefix, phases = read_place_entries(inputs, [(row, seed, frame, condition)])[
        row['episode_id']]
    with create_place_task(inputs, row, metadata, prefix, phases) as task:
        world = physical_evidence_world(task)
        assert all(place_preconditions(world, row).values())
        assert task._phase == 'transport'
