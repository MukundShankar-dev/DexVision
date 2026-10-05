"""Exact copied-reset candidate search for the 5.4A scripted place source."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml

from dexvision.evaluation.place_repair import (
    load_place_teacher_spec,
    run_place_teacher_attempt,
    summarize_teacher_results,
)
from dexvision.learning.train_skill import _checksum_index, _json
from dexvision.learning.training_manifest import content_digest, digest_file
from dexvision.logging.level4_collection import sample_level4_procedural_variation


def load_place_search_spec(path):
    """Load a locked candidate set and its locked V4 physical controller."""
    path = Path(path)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place search recipe checksum mismatch')
    search = yaml.safe_load(path.read_text())
    if search.get('version') not in {'level5/place-teacher-search-v1',
                                     'level5/place-teacher-search-v2'} \
            or search.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place teacher search recipe')
    base_path = Path(search['base_teacher_recipe'])
    if digest_file(base_path) != search['base_teacher_recipe_sha256']:
        raise ValueError('Place search base teacher changed')
    base, place, pick = load_place_teacher_spec(base_path)
    if base['version'] != 'level5/place-teacher-v4' or search['gates'] != base['gates']:
        raise ValueError('Place search must retain the frozen V4 gates')
    yaw = search['candidate_yaw_degrees']
    if not yaw or len(yaw) != len(set(yaw)) or any(not -25 <= value <= 0 for value in yaw):
        raise ValueError('Invalid place search yaw candidates')
    triggers = search['candidate_cuboid_release_distance_m']
    if not triggers or len(triggers) != len(set(triggers)) or any(
            not 0 < value < place[0]['terminal']['release_distance_m'] for value in triggers):
        raise ValueError('Invalid place search release distances')
    scales = search['candidate_release_step_scales']
    if set(scales) != {'cuboid', 'cylinder', 'flat_puck'} or any(
            not values or len(values) != len(set(values)) or any(not 0 < value <= 1 for value in values)
            for values in scales.values()):
        raise ValueError('Invalid place search opening candidates')
    required_selection = ('first_exact_same_reset_physical_pass_zero_safety'
        if search['version'].endswith('v1') else
        'first_exact_same_reset_physical_pass_else_first_zero_safety_failure')
    if search['selection'] != required_selection:
        raise ValueError('Place search must retain exact physical candidate selection')
    return search, base, place, pick


def _candidate_recipes(search, base, row, family):
    """Yield a deterministic bounded grid, with the frozen base candidate first."""
    object_id = row['entity_id']
    target_id = row['typed_goal']['target_id']
    base_yaw = base['post_pick_setup']['yaw_degrees_by_object_target'].get(object_id, {}).get(target_id, 0)
    yaw_values = [base_yaw, *search['candidate_yaw_degrees']] if family == 'cuboid' else [base_yaw]
    triggers = ([base['release_distance_by_family_m']['cuboid'],
                 *search['candidate_cuboid_release_distance_m']] if family == 'cuboid' else
                [base['release_distance_by_family_m'][family]])
    scales = [base['release_step_scale_by_family'][family], *search['candidate_release_step_scales'][family]]
    seen = set()
    for yaw in yaw_values:
        for trigger in triggers:
            for scale in scales:
                key = (yaw, trigger, scale)
                if key in seen:
                    continue
                seen.add(key)
                candidate = deepcopy(base)
                if family == 'cuboid':
                    candidate['post_pick_setup']['yaw_degrees_by_object_target'].setdefault(object_id, {})[
                        target_id] = yaw
                    candidate['release_distance_by_family_m']['cuboid'] = trigger
                candidate['release_step_scale_by_family'][family] = scale
                yield {'yaw_degrees': yaw, 'release_distance_m': trigger,
                       'release_step_scale': scale}, candidate


def run_place_search_attempt(search, base, place, pick, row, metadata, output_dir):
    """Preview candidates on identical resets, then reproduce one live attempt."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    object_id = row['entity_id']
    # Use the workcell's declared family, never an outcome-derived label.
    family = place[1].requirements['workcell']['objects'][object_id]['family']
    candidates = list(_candidate_recipes(search, base, row, family))
    audits, selected = [], None
    for index, (parameters, candidate) in enumerate(candidates):
        result = run_place_teacher_attempt(candidate, place, pick, row, metadata,
            directory / 'preflight' / f'{index:02d}')
        passed = result['terminal_result'] == 'succeeded' and not any(result['safety_counters'].values())
        receipt_keys = ('terminal_result', 'terminal_reason', 'safety_counters', 'place_actions',
                        'pick_setup', 'place_setup', 'final_metrics', 'final_physical_context')
        receipt = content_digest({key: result.get(key) for key in receipt_keys})
        audits.append({'index': index, **parameters, 'terminal_result': result['terminal_result'],
                       'terminal_reason': result['terminal_reason'],
                       'safety_counters': result['safety_counters'], 'receipt_sha256': receipt,
                       'passed': passed})
        if passed:
            selected = index
            break
    if selected is None and search['version'].endswith('v2'):
        selected = next((item['index'] for item in audits
                         if not any(item['safety_counters'].values())), None)
        if selected is None:
            result = {'episode_id': row['episode_id'], 'cell_id': row['cell_id'],
                'reset_seed': metadata['random_seed'], 'source_digest': row['episode_digest'],
                'object_family': family, 'target_id': row['typed_goal']['target_id'],
                'terminal_result': 'rejected_before_execution', 'terminal_reason': 'no_safe_candidate',
                'safety_counters': dict.fromkeys(next(iter(audits))['safety_counters'], 0),
                'place_actions': 0, 'source_search': search['version'],
                'selected_candidate': None, 'selected_candidate_index': None,
                'preflight_candidates': audits, 'preflight_passed': False,
                'disclosure': 'All exact-reset candidates retained; no live place actions when no safe candidate'}
            _json(directory / 'result.json', result)
            _checksum_index(directory)
            return result
    if selected is None:
        selected = 0  # Preserve the historical V1 fallback and its negative evidence.
    parameters, candidate = candidates[selected]
    live = run_place_teacher_attempt(candidate, place, pick, row, metadata, directory / 'live')
    preview = audits[selected]
    if content_digest({key: live.get(key) for key in receipt_keys}) != preview['receipt_sha256']:
        raise ValueError('Place search live attempt differs from exact-reset preflight')
    result = {**live, 'source_search': search['version'], 'selected_candidate': parameters,
              'selected_candidate_index': selected, 'preflight_candidates': audits,
              'preflight_passed': preview['passed'],
              'disclosure': 'Exact same-reset ordinary-physics candidate search; all previews retained; '
                            'one selected live attempt; no post-reset object-state writes'}
    _json(directory / 'result.json', result)
    _checksum_index(directory)
    return result


def qualify_place_search_coverage(config, output_dir):
    """Run all 30 locked object/target cells on one new seed each."""
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place search coverage checksum mismatch')
    coverage = yaml.safe_load(path.read_text())
    if coverage.get('version') not in {'level5/place-teacher-search-coverage-v1',
                                      'level5/place-teacher-search-coverage-v2'} \
            or coverage.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place search coverage recipe')
    search_path = Path(coverage['search_recipe'])
    if digest_file(search_path) != coverage['search_recipe_sha256']:
        raise ValueError('Place search coverage recipe changed')
    search, base, place, pick = load_place_search_spec(search_path)
    _, inputs = place
    config_path = inputs.root / inputs.plan['inputs']['dataset_config']['path']
    dataset = yaml.safe_load(config_path.read_text())
    cells = {cell['id']: cell for cell in dataset['coverage_cells']
             if cell.get('data_group') == 'pick_place'}
    if set(coverage['cells']) != set(cells) or len(coverage['cells']) != 30:
        raise ValueError('Place search coverage must contain every cell once')
    boundary_cases = coverage.get('boundary_cases', [coverage['boundary_case']])
    if boundary_cases not in ([False], [True], [False, True]):
        raise ValueError('Place search coverage has invalid boundary-case matrix')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    results, reset_manifest = [], []
    for index, cell_id in enumerate(coverage['cells']):
        cell = cells[cell_id]
        for offset, boundary in enumerate(boundary_cases):
            seed = coverage['seed_start'] + index * len(boundary_cases) + offset
            variation = sample_level4_procedural_variation(
                inputs.requirements['level4_5b_procedural_expansion'],
                skill_name='pick_place_sequence', seed=seed, boundary_case=boundary)
            goal = {'object_id': cell['object_id'], 'target_id': cell['target_id']}
            metadata = {'random_seed': seed, 'task_config': {'procedural_variation': variation}}
            row = {'episode_id': f'place_teacher_search_coverage_{seed}', 'cell_id': cell_id,
                   'entity_id': cell['object_id'], 'typed_goal': goal,
                   'episode_digest': content_digest({'cell': cell_id, 'seed': seed, 'variation': variation})}
            reset_manifest.append({'episode_id': row['episode_id'], 'cell_id': cell_id,
                                   'seed': seed, 'goal': goal, 'variation': variation,
                                   'boundary_case': boundary})
            name = (f"{index:02d}_{row['episode_id']}" if len(boundary_cases) == 1
                    else f"{index:02d}_{offset}_{row['episode_id']}")
            result = run_place_search_attempt(search, base, place, pick, row, metadata,
                directory / name)
            results.append(result)
            print(f"{row['episode_id']} {cell_id}: {result['terminal_reason']} "
                  f"candidate={result['selected_candidate_index']}", flush=True)
    summary = summarize_teacher_results(search, results, search_path,
        'New development seeds, exact same-reset source preflight; no learned-policy qualification')
    summary['coverage_recipe_sha256'] = digest_file(path)
    summary['target_counts'] = dict(Counter(r['target_id'] for r in results))
    summary['target_successes'] = dict(Counter(r['target_id'] for r in results
                                               if r['terminal_result'] == 'succeeded'))
    summary['preflight_candidate_count'] = sum(len(r['preflight_candidates']) for r in results)
    _json(directory / 'resets.json', {'version': coverage['version'], 'resets': reset_manifest})
    _json(directory / 'summary.json', summary)
    _checksum_index(directory)
    return summary
