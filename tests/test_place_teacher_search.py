"""Exact-reset candidate search evidence for the 5.4A scripted source."""
import pytest

from dexvision.evaluation.place_teacher_search import load_place_search_spec, run_place_search_attempt
from dexvision.learning.training_manifest import content_digest
from dexvision.logging.level4_collection import sample_level4_procedural_variation


def test_search_recipe_retains_frozen_place_gates():
    search, base, place, _ = load_place_search_spec('configs/level5/place_teacher_search_v1.yaml')
    assert search['gates'] == base['gates']
    assert max(search['candidate_cuboid_release_distance_m']) < place[0]['terminal']['release_distance_m']
    revision, _, _, _ = load_place_search_spec('configs/level5/place_teacher_search_v2.yaml')
    assert 'first_zero_safety_failure' in revision['selection']


def test_exact_reset_search_reproduces_selected_live_place(tmp_path):
    search, base, place, pick = load_place_search_spec('configs/level5/place_teacher_search_v1.yaml')
    seed = 840260916
    cell = 'pp_block_large_inspection_pad'
    variation = sample_level4_procedural_variation(
        place[1].requirements['level4_5b_procedural_expansion'],
        skill_name='pick_place_sequence', seed=seed, boundary_case=False)
    goal = {'object_id': 'block_large', 'target_id': 'inspection_pad'}
    row = {'episode_id': f'place_teacher_coverage_{seed}', 'cell_id': cell,
           'entity_id': 'block_large', 'typed_goal': goal,
           'episode_digest': content_digest({'cell': cell, 'seed': seed, 'variation': variation})}
    metadata = {'random_seed': seed, 'task_config': {'procedural_variation': variation}}
    result = run_place_search_attempt(search, base, place, pick, row, metadata, tmp_path / 'attempt')
    assert result['terminal_reason'] == 'physical_settled_placement_passed'
    assert result['selected_candidate_index'] == 3
    assert len(result['preflight_candidates']) == 4
    assert not any(result['safety_counters'].values())
    assert (tmp_path / 'attempt' / 'preflight' / '03' / 'trajectory.npz').is_file()
    assert (tmp_path / 'attempt' / 'live' / 'trajectory.npz').is_file()


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_search_coverage_cli_dispatch(tmp_path, monkeypatch, version):
    from dexvision.apps import qualify_place_teacher as app

    config = tmp_path / 'coverage.yaml'
    config.write_text(f'version: level5/place-teacher-search-coverage-{version}\n')
    monkeypatch.setattr(app, 'qualify_place_teacher',
        lambda *_: pytest.fail('Search coverage was sent to the single-target runner'))
    monkeypatch.setattr(app, 'qualify_place_teacher_coverage',
        lambda *_: pytest.fail('Search coverage was sent to the static coverage runner'))
    monkeypatch.setattr(app, 'qualify_place_search_coverage',
        lambda *_: {'teacher_feasibility_passed': True})
    assert app.main(['--config', str(config), '--output-dir', str(tmp_path / 'out')]) == 0


@pytest.mark.parametrize('any_safe', [True, False])
def test_v2_search_never_runs_unsafe_fallback(tmp_path, monkeypatch, any_safe):
    from dexvision.evaluation import place_teacher_search as source

    search, base, place, pick = load_place_search_spec('configs/level5/place_teacher_search_v2.yaml')
    row = {'episode_id': 'synthetic', 'cell_id': 'pp_block_small_return_bin_left',
           'entity_id': 'block_small',
           'typed_goal': {'object_id': 'block_small', 'target_id': 'return_bin_left'},
           'episode_digest': 'synthetic'}
    calls = []

    def failing_candidate(*args):
        calls.append(args[-1])
        safe = any_safe and len(calls) != 1
        return {'episode_id': 'synthetic', 'terminal_result': 'failed',
                'terminal_reason': 'premature_release' if safe else 'object_workspace_violation',
                'safety_counters': {'object_workspace_violations': 0 if safe else 1},
                'place_actions': 1}

    monkeypatch.setattr(source, 'run_place_teacher_attempt', failing_candidate)
    result = run_place_search_attempt(search, base, place, pick, row,
        {'random_seed': 1}, tmp_path / 'attempt')
    if any_safe:
        assert result['selected_candidate_index'] == 1
        assert result['terminal_reason'] == 'premature_release'
        assert not any(result['safety_counters'].values())
        assert len(calls) == len(result['preflight_candidates']) + 1
    else:
        assert result['selected_candidate_index'] is None
        assert result['terminal_reason'] == 'no_safe_candidate'
        assert not any(result['safety_counters'].values())
        assert len(calls) == len(result['preflight_candidates'])
