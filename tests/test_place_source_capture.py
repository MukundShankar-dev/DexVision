"""Exact place replay and pre-action recording for additive 5.4A data."""
import json

import numpy as np

from dexvision.evaluation.place_repair import load_place_teacher_spec, run_place_teacher_attempt
from dexvision.evaluation.place_source_audit import audit_place_payload
from dexvision.learning.training_manifest import content_digest
from dexvision.logging.level4_collection import sample_level4_procedural_variation
from dexvision.logging.place_source_capture import capture_causal_place_trace


def test_causal_place_capture_replays_every_source_state(tmp_path):
    recipe, place, pick = load_place_teacher_spec('configs/level5/place_teacher_v4.yaml')
    seed = 830260915
    cell = 'pp_block_large_return_bin_left'
    variation = sample_level4_procedural_variation(
        place[1].requirements['level4_5b_procedural_expansion'],
        skill_name='pick_place_sequence', seed=seed, boundary_case=False)
    goal = {'object_id': 'block_large', 'target_id': 'return_bin_left'}
    row = {'episode_id': 'test_place_capture', 'cell_id': cell, 'entity_id': 'block_large',
           'typed_goal': goal, 'episode_digest': content_digest(
               {'cell': cell, 'seed': seed, 'variation': variation})}
    metadata = {'random_seed': seed, 'task_config': {'procedural_variation': variation}}
    source = tmp_path / 'source'
    result = run_place_teacher_attempt(recipe, place, pick, row, metadata, source)
    assert result['terminal_result'] == 'succeeded'
    output = tmp_path / 'capture'
    capture = capture_causal_place_trace(place, row, metadata, source, output,
        session_id='test_session')
    assert capture['source_replay_exact']
    with np.load(output / 'trajectory.npz', allow_pickle=False) as data:
        assert data['observations'].shape == (result['place_actions'], 195)
        assert data['actions'].shape == (result['place_actions'], 27)
        assert len(data['qpos']) == len(data['actions']) + 1
        assert np.array_equal(data['actions'][:-1], data['prior_applied_actions'][1:])
        assert not np.any(data['safety_masks'])
        assert set(data['phases']) <= {'transport', 'place', 'release', 'settle', 'retract'}
    saved = json.loads((output / 'result.json').read_text())
    assert saved['source_replay_exact'] and saved['session_id'] == 'test_session'
    audit = audit_place_payload(place, output)
    assert audit['passed'] and audit['settle_steps'] >= 10
    assert audit['pre_release_hand_table_samples'] == 0
    assert audit['post_release_hand_table_samples'] > 0
    assert audit['before_release_command_hand_table_samples'] == 0
