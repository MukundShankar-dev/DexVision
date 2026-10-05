"""Focused checks for the versioned 5.4A place teacher."""
from pathlib import Path

import numpy as np
import pytest

from dexvision.evaluation.place_repair import (
    load_place_teacher_spec,
    place_teacher_action,
    release_fingers,
    run_place_teacher_attempt,
)
from dexvision.logging.level4_collection import sample_level4_procedural_variation
from dexvision.learning.training_manifest import content_digest


def test_recipe_binds_both_prior_sources_and_frozen_release_gate():
    recipe, (place, _), _ = load_place_teacher_spec('configs/level5/place_teacher_v1.yaml')
    assert recipe['release_distance_by_family_m']['cuboid'] < place['terminal']['release_distance_m']
    assert recipe['release_distance_by_family_m']['flat_puck'] < place['terminal']['release_distance_m']
    revision, _, _ = load_place_teacher_spec('configs/level5/place_teacher_v2.yaml')
    assert revision['release_step_scale'] == 0.5
    repair, _, _ = load_place_teacher_spec('configs/level5/place_teacher_v3.yaml')
    assert repair['post_pick_setup']['tighten_excess_fraction_by_object']['block_large'] == 0.10
    revision, _, _ = load_place_teacher_spec('configs/level5/place_teacher_v4.yaml')
    assert revision['post_pick_setup']['tighten_translate_x_m_by_object'] == {'block_large': 0.002}
    assert revision['release_step_scale_by_family']['flat_puck'] == 0.25


def test_rank_one_release_holds_inactive_channels_and_stops_at_open_pose():
    opened = np.zeros(20)
    closed = np.ones(20)
    closed[[0, 1, 4]] = 0
    result = release_fingers(closed, opened, closed, 0.05)
    assert np.array_equal(result[[0, 1, 4]], closed[[0, 1, 4]])
    assert np.allclose(result[2], 0.95)
    assert np.array_equal(release_fingers(opened, opened, closed, 0.05), opened)
    with pytest.raises(ValueError, match='opening direction'):
        release_fingers(closed, np.ones(20), np.zeros(20), 0.05)


def test_causal_release_latch_holds_hand_while_fingers_open():
    recipe, (spec, _), _ = load_place_teacher_spec('configs/level5/place_teacher_v1.yaml')
    prior = np.r_[np.array([0., 0., 0.1, 1., 0., 0., 0.]), np.ones(20)]
    requested = prior.copy()
    requested[2] += 0.01
    opened, closed = np.zeros(20), np.ones(20)
    raw, latch, saturated = place_teacher_action(requested, prior, held=True,
        error=np.array([0., 0., -0.02]), family='cuboid', latch=None, recipe=recipe,
        place_spec=spec, open_targets=opened, closed_targets=closed, release_step=0.05)
    assert latch is not None and np.array_equal(raw[:7], prior[:7])
    assert np.allclose(raw[7:], 0.95) and not saturated
    next_raw, next_latch, _ = place_teacher_action(requested, raw, held=False,
        error=np.array([0., 0., -0.018]), family='cuboid', latch=latch, recipe=recipe,
        place_spec=spec, open_targets=opened, closed_targets=closed, release_step=0.05)
    assert np.array_equal(next_latch, latch)
    assert np.array_equal(next_raw[:7], prior[:7])
    assert np.allclose(next_raw[7:], 0.9)


def test_grasp_stays_closed_while_measured_object_is_far_from_target():
    recipe, (spec, _), _ = load_place_teacher_spec('configs/level5/place_teacher_v1.yaml')
    prior = np.r_[np.array([0., 0., 0.1, 1., 0., 0., 0.]), np.ones(20)]
    requested = prior.copy()
    requested[:3] = [0.001, 0.002, 0.102]
    requested[7:] = 0.9
    raw, latch, saturated = place_teacher_action(requested, prior, held=True,
        error=np.array([0.05, 0., -0.03]), family='cuboid', latch=None, recipe=recipe,
        place_spec=spec, open_targets=np.zeros(20), closed_targets=np.ones(20), release_step=0.05)
    assert latch is None and not saturated
    assert np.array_equal(raw[7:], prior[7:])
    assert np.array_equal(raw[:3], requested[:3])


@pytest.mark.parametrize(('seed', 'cell', 'object_id', 'target_id', 'tighten', 'shift'), [
    (820260915, 'pp_block_large_return_bin_left', 'block_large', 'return_bin_left', True, 0.0),
    (820260926, 'pp_puck_heavy_inspection_pad', 'puck_heavy', 'inspection_pad', False, 0.005),
])
def test_v3_setup_and_place_retain_physical_evidence(tmp_path, seed, cell, object_id,
                                                       target_id, tighten, shift):
    recipe, place, pick = load_place_teacher_spec('configs/level5/place_teacher_v3.yaml')
    _, inputs = place
    variation = sample_level4_procedural_variation(
        inputs.requirements['level4_5b_procedural_expansion'],
        skill_name='pick_place_sequence', seed=seed, boundary_case=False)
    goal = {'object_id': object_id, 'target_id': target_id}
    row = {'episode_id': f'place_teacher_coverage_{seed}', 'cell_id': cell,
        'entity_id': object_id, 'typed_goal': goal,
        'episode_digest': content_digest({'cell': cell, 'seed': seed, 'variation': variation})}
    metadata = {'random_seed': seed, 'task_config': {'procedural_variation': variation}}
    output = Path(tmp_path) / 'attempt'
    result = run_place_teacher_attempt(recipe, place, pick, row, metadata, output)
    assert result['terminal_reason'] == 'physical_settled_placement_passed'
    assert result['place_setup']['success']
    assert bool(result['place_setup']['tighten_actions']) == tighten
    assert result['inboard_raise_shift_m'] == shift
    assert not any(result['safety_counters'].values())
    with np.load(output / 'place_setup_trajectory.npz') as setup:
        assert len(setup['qpos']) == len(setup['actions']) + 1
        assert len(setup['qvel']) == len(setup['qpos'])
        assert len(setup['timestamps']) == len(setup['qpos'])
    assert (output / 'pick_setup_trajectory.npz').is_file()
    assert (output / 'trajectory.npz').is_file()


def test_coverage_v2_cli_uses_all_cell_runner(tmp_path, monkeypatch):
    from dexvision.apps import qualify_place_teacher as app

    config = tmp_path / 'coverage.yaml'
    config.write_text('version: level5/place-teacher-coverage-v2\n')
    monkeypatch.setattr(app, 'qualify_place_teacher',
        lambda *_: pytest.fail('Coverage was sent to the single-target runner'))
    monkeypatch.setattr(app, 'qualify_place_teacher_coverage',
        lambda *_: {'teacher_feasibility_passed': True})
    assert app.main(['--config', str(config), '--output-dir', str(tmp_path / 'out')]) == 0


def test_v4_coupled_inboard_grip_keeps_edge_block_held(tmp_path):
    recipe, place, pick = load_place_teacher_spec('configs/level5/place_teacher_v4.yaml')
    _, inputs = place
    seed = 830260915
    cell = 'pp_block_large_return_bin_left'
    variation = sample_level4_procedural_variation(
        inputs.requirements['level4_5b_procedural_expansion'],
        skill_name='pick_place_sequence', seed=seed, boundary_case=False)
    goal = {'object_id': 'block_large', 'target_id': 'return_bin_left'}
    row = {'episode_id': f'place_teacher_coverage_{seed}', 'cell_id': cell,
        'entity_id': 'block_large', 'typed_goal': goal,
        'episode_digest': content_digest({'cell': cell, 'seed': seed, 'variation': variation})}
    metadata = {'random_seed': seed, 'task_config': {'procedural_variation': variation}}
    output = tmp_path / 'attempt'
    result = run_place_teacher_attempt(recipe, place, pick, row, metadata, output)
    assert result['terminal_reason'] == 'physical_settled_placement_passed'
    assert result['place_setup']['tighten_actions'] > 0
    assert not any(result['safety_counters'].values())
    with np.load(output / 'place_setup_trajectory.npz') as setup:
        tighten = setup['actions'][setup['phases'] == 'tighten']
        assert np.all(np.diff(tighten[:, 0]) > 0)
        assert len(setup['qpos']) == len(setup['actions']) + 1


def test_coverage_v3_cli_uses_all_cell_runner(tmp_path, monkeypatch):
    from dexvision.apps import qualify_place_teacher as app

    config = tmp_path / 'coverage.yaml'
    config.write_text('version: level5/place-teacher-coverage-v3\n')
    monkeypatch.setattr(app, 'qualify_place_teacher',
        lambda *_: pytest.fail('Coverage was sent to the single-target runner'))
    monkeypatch.setattr(app, 'qualify_place_teacher_coverage',
        lambda *_: {'teacher_feasibility_passed': True})
    assert app.main(['--config', str(config), '--output-dir', str(tmp_path / 'out')]) == 0
