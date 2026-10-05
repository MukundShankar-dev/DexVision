"""Closed-loop validation and fresh confirmation for the 5.4A place repair."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dexvision.evaluation.benchmark_retargeters import mean_action_jerk
from dexvision.evaluation.evaluate_skill import SAFETY_KEYS, apply_action, reach_features, safe_action
from dexvision.evaluation.pick_policy import physical_evidence_world
from dexvision.evaluation.pick_teacher import build_teacher, execute_teacher
from dexvision.evaluation.place_policy import PlaceEvidence, place_preconditions
from dexvision.evaluation.place_repair import _place_pick_recipe, _post_pick_setup, load_place_teacher_spec
from dexvision.learning.place_policy import place_action_plan
from dexvision.learning.place_repair_policy import RepairPolicy, load_repair_spec
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, select_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file
from dexvision.logging.level4_collection import WorkcellPilotTask, sample_level4_procedural_variation


def repair_matrix(spec, inputs, split):
    """Freeze all assigned validation resets or a distinct fresh all-cell matrix."""
    manifest = json.loads(Path(spec['additive_data']['manifest']).read_text())
    cells = {cell['id']: cell for cell in inputs.requirements['coverage_cells']
             if cell.get('data_group') == 'pick_place'}
    if split == 'validation':
        assignments = manifest['reports']['validation']['results']
        rows = [(row['cell'], row['seed'], row['boundary'], row['episode_id'], row['session_id'])
                for row in assignments]
        if len(rows) != 48:
            raise ValueError('Place validation matrix lost a frozen reset')
    elif split == 'confirmation':
        names = sorted(cells)
        if len(names) != 30:
            raise ValueError('Place confirmation needs all 30 source cells')
        rows = []
        for index, cell in enumerate(names):
            for offset in range(spec['confirmation']['resets_per_cell']):
                seed = spec['confirmation']['seed_start'] + index * 100 + offset
                rows.append((cell, seed, bool(offset % 2), f'place_repair_confirm_{seed}',
                             f'place_repair_confirm_{cell}'))
    else:
        raise ValueError('Place repair split must be validation or confirmation')
    result = []
    for cell_id, seed, boundary, eid, session in rows:
        cell = cells[cell_id]
        variation = sample_level4_procedural_variation(
            inputs.requirements['level4_5b_procedural_expansion'],
            skill_name='pick_place_sequence', seed=seed, boundary_case=boundary)
        result.append({'episode_id': eid, 'cell_id': cell_id, 'seed': seed,
            'boundary': boundary, 'session': session, 'entity_id': cell['object_id'],
            'typed_goal': {'object_id': cell['object_id'], 'target_id': cell['target_id']},
            'variation': variation})
    if len({r['seed'] for r in result}) != len(result):
        raise ValueError('Duplicate place repair reset')
    return result


def _setup(task, row, teacher, pick, inputs):
    """Unassisted scripted pick and fixed, predeclared grip/yaw setup only."""
    recipe, _, _ = teacher
    pick_recipe, pick_spec, pick_inputs = pick
    runtime = _place_pick_recipe(task, pick_recipe, recipe)
    expert, preflight, _ = build_teacher(task, runtime, pick_spec, pick_inputs)
    pick_result, pick_arrays = execute_teacher(task, expert, preflight, runtime, pick_spec, pick_inputs)
    if not pick_result['success']:
        return {'success': False, 'reason': 'pick_setup_failed', 'pick': pick_result,
                'safety_counters': pick_result['safety_counters']}, None
    source = physical_evidence_world(task)
    relation = source.relation_for(row['entity_id'])
    if relation.held_by != 'rh_palm' or relation.supported_by is not None:
        return {'success': False, 'reason': 'held_precondition_failed', 'pick': pick_result,
                'safety_counters': pick_result['safety_counters']}, None
    counters = dict.fromkeys(SAFETY_KEYS, 0)
    counters['unwanted_contact'] = 0
    setup, prior, _ = _post_pick_setup(task, pick_arrays, row, recipe, pick_spec,
                                       inputs[0], inputs[1], counters)
    for key, value in pick_result['safety_counters'].items():
        counters[key] = counters.get(key, 0) + value
    return {'success': bool(setup['success']), 'reason': 'ready' if setup['success']
            else 'place_setup_failed', 'pick': pick_result, 'place_setup': setup,
            'safety_counters': counters}, prior


def run_repair_rollout(spec, inputs, teacher, pick, row, policy, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    fields = inputs.plan['action']['named_layout']
    plan = place_action_plan(spec, inputs)
    counters = dict.fromkeys(SAFETY_KEYS, 0)
    counters.update(object_workspace_violations=0, unwanted_contact=0)
    arrays = defaultdict(list)
    terminal, reason, metrics = 'timed_out', 'unsettled', {}
    with WorkcellPilotTask(
            workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
            dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
            skill_name='pick_place_sequence', goal_condition_id=row['cell_id'], seed=row['seed'],
            procedural_variation=row['variation']) as task:
        if task.goal != row['typed_goal']:
            raise ValueError('Place repair reset goal mismatch')
        setup, prior = _setup(task, row, teacher, pick, (spec, inputs))
        _json(directory / 'setup.json', setup)
        for key, value in setup.get('safety_counters', {}).items():
            counters[key] = counters.get(key, 0) + value
        world = physical_evidence_world(task)
        object_id, target_id = row['entity_id'], row['typed_goal']['target_id']
        family = next(o.family for o in task.workcell.config.objects if o.object_id == object_id)
        evidence = PlaceEvidence(object_id, target_id, spec['terminal'], world.require_entity(object_id).position)
        checks = place_preconditions(world, row)
        for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                           ('state_timestamps', task.env.data.time)):
            arrays[key].append(np.asarray(value).copy())
        if not setup['success'] or not all(checks.values()):
            terminal, reason = 'rejected_before_execution', setup['reason'] if not setup['success'] \
                else 'rejected_precondition'
        else:
            policy.reset()
            commanded, safety = prior.copy(), np.zeros(27)
            for _ in range(teacher[0]['maximum_place_actions']):
                world = physical_evidence_world(task)
                source = reach_features(task, world, row, inputs, commanded, prior, safety, task._phase)
                raw, prediction = policy(source, prior, task._phase)
                applied, mask, count, unsafe = safe_action(raw, prior, plan)
                for key in SAFETY_KEYS:
                    counters[key] += count[key]
                for key, value in (('observations', source), ('raw_actions', raw),
                    ('applied_actions', applied), ('policy_predictions', prediction),
                    ('prior_applied_actions', prior), ('safety_masks', mask)):
                    arrays[key].append(np.asarray(value).copy())
                apply_action(task, applied, fields)
                task.step(n_steps=inputs.plan['action']['simulation_steps_per_action'])
                world = physical_evidence_world(task)
                frozen = SimpleNamespace(values=task.current_state.task_values,
                                         success=task.current_state.success)
                metrics = evidence.update(world, frozen)
                for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                                   ('state_timestamps', task.env.data.time)):
                    arrays[key].append(np.asarray(value).copy())
                board = inputs.requirements['workcell']['board_workspace']
                xy = np.asarray(world.require_entity(object_id).position[:2])
                off_board = bool(np.any(xy < np.asarray(board['min_xy_m']) + board['safe_edge_margin_m'])
                    or np.any(xy > np.asarray(board['max_xy_m']) - board['safe_edge_margin_m']))
                counters['object_workspace_violations'] += int(off_board)
                entities = set(task.workcell.config.object_ids) | set(task.workcell.config.fixture_ids)
                unwanted = sum(any(name.startswith('rh_') for name in pair) and
                    any(name in entities and name not in {object_id, target_id} for name in pair)
                    for pair in world.contacts)
                counters['unwanted_contact'] += unwanted
                metrics['unwanted_contact_count'] = unwanted
                failure = unsafe or ('object_workspace_violation' if off_board else
                    'unwanted_contact' if unwanted else
                    'premature_release' if evidence.premature_release else
                    'bounce' if evidence.bounced else None)
                prior, commanded, safety = applied, raw, mask
                if failure:
                    terminal, reason = 'failed', failure
                    break
                if metrics['success']:
                    terminal, reason = 'succeeded', 'physical_settled_placement_passed'
                    break
        end = float(task.env.data.time)
    np.savez_compressed(directory / 'trajectory.npz',
        **{key: np.asarray(value) for key, value in arrays.items()})
    bounds = np.asarray([f['upper'] - f['lower'] for f in fields])
    actions = np.asarray(arrays['applied_actions']) if arrays['applied_actions'] else np.empty((0, 27))
    result = {'episode_id': row['episode_id'], 'reset_seed': row['seed'], 'cell_id': row['cell_id'],
        'session': row['session'], 'object_family': family, 'object_instance': object_id,
        'target_id': target_id, 'target_type': inputs.skill['target_type_mapping'][target_id],
        'boundary_class': 'boundary' if row['boundary'] else 'nominal',
        'render_condition': row['render_condition'],
        'terminal_result': terminal, 'terminal_reason': reason, 'final_metrics': metrics,
        'preconditions': checks, 'setup': setup, 'safety_counters': counters,
        'premature_release': evidence.premature_release, 'nominally_placed': evidence.nominally_placed,
        'bounced': evidence.bounced, 'transported': evidence.transported,
        'released': evidence.released,
        'settled': evidence.settle_dwell >= spec['terminal']['consecutive_samples'] and not evidence.bounced,
        'normalized_terminal_error': metrics['object_to_target_distance_m'] /
            spec['terminal']['target_distance_m'] if metrics else None,
        'mean_action_jerk': mean_action_jerk(actions / bounds) if len(actions) else 0.,
        'end_timestamp': end, 'trajectory_path': 'trajectory.npz'}
    _json(directory / 'result.json', result)
    _checksum_index(directory)
    return result


def summarize_repair(results, inputs):
    if not results or len({(r['episode_id'], r['render_condition']) for r in results}) != len(results):
        raise ValueError('Place repair report has missing/duplicate resets')
    success = sum(r['terminal_result'] == 'succeeded' for r in results) / len(results)
    groups = defaultdict(list)
    for result in results:
        groups[result['object_family']].append(result)
    worst = min(sum(r['terminal_result'] == 'succeeded' for r in group) / len(group)
                for group in groups.values())
    cells = defaultdict(list)
    for result in results:
        cells[result['cell_id']].append(result)
    worst_cell = min(sum(r['terminal_result'] == 'succeeded' for r in group) / len(group)
                     for group in cells.values())
    premature = sum(r['premature_release'] for r in results) / len(results)
    nominal = [r for r in results if r['nominally_placed']]
    settled = sum(r['settled'] for r in nominal) / len(nominal) if nominal else 0.
    counters = {key: sum(r['safety_counters'][key] for r in results)
                for key in results[0]['safety_counters']}
    jerk = float(np.mean([r['mean_action_jerk'] for r in results]))
    gates = inputs.skill['gates']
    passed = {'aggregate_success': success >= gates['state_success_min'],
        'worst_family_success': worst >= gates['state_worst_family_success_min'],
        'premature_release_rate': premature <= gates['premature_release_rate_max'],
        'settled_among_nominally_placed': settled >= gates['settled_among_nominally_placed_min'],
        'mean_action_jerk': jerk <= gates['mean_action_jerk_max'],
        **{key: value == 0 for key, value in counters.items()}}
    errors = [r['normalized_terminal_error'] for r in results
              if r['normalized_terminal_error'] is not None]
    return {'rollout_count': len(results),
        'unique_reset_count': len({r['episode_id'] for r in results}),
        'success_rate': success, 'worst_family_success_rate': worst,
        'worst_cell_success_rate': worst_cell, 'premature_release_rate': premature,
        'settled_among_nominally_placed': settled, 'mean_action_jerk': jerk,
        'safety_counters': counters, 'safety_violation_count': sum(counters.values()),
        'normalized_terminal_error': float(np.mean(errors)) if errors else None,
        'terminal_error_unavailable_reason': None if errors else 'rejected_precondition',
        'terminal_reasons': dict(Counter(r['terminal_reason'] for r in results)),
        'gates': passed, 'state_numerical_gates_passed': all(passed.values())}


def evaluate_repair(config, checkpoint_path, output_dir, *, split='validation'):
    spec, inputs = load_repair_spec(config)
    policy = RepairPolicy(checkpoint_path, config)
    checkpoint = load_checkpoint(checkpoint_path)
    teacher = load_place_teacher_spec(spec['setup_teacher_recipe'])
    pick = teacher[2]
    resets = repair_matrix(spec, inputs, split)
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    matrix = [{**row, 'render_condition': condition} for row in resets for condition in conditions]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': 'level5/place-repair-evaluation-v2', 'split': split,
        'seed': checkpoint['manifest']['seed'],
        'epoch': checkpoint['training_state']['completed_epochs'],
        'checkpoint_sha256': digest_file(checkpoint_path),
        'manifest_digest': checkpoint['manifest']['manifest_digest'],
        'validation_matrix_sha256': content_digest(repair_matrix(spec, inputs, 'validation')),
        'matrix_sha256': content_digest(matrix), 'recipe_sha256': digest_file(Path(config)),
        'source_exposure': spec['confirmation']['source_exposure']}
    if split == 'confirmation':
        run = Path(checkpoint_path).parent
        _verify_run(run)
        selected = json.loads((run / 'selection.json').read_text())
        if (Path(checkpoint_path).name != 'selected.pt'
                or selected['status'] != 'selected_from_validation'
                or (run / 'confirmation_attempt.json').exists()):
            raise ValueError('Fresh place confirmation needs untested selected weights')
        _json(run / 'confirmation_attempt.json', provenance)
    _json(directory / 'provenance.json', provenance)
    results = []
    for index, row in enumerate(matrix):
        if index % 12 == 0:
            print(f'Place {split} seed {provenance["seed"]} {index}/{len(matrix)}', flush=True)
        result = run_repair_rollout(spec, inputs, teacher, pick, row, policy,
            directory / f'{index:03d}_{row["episode_id"]}_{row["render_condition"]}')
        result['result_path'] = f'{index:03d}_{row["episode_id"]}_{row["render_condition"]}/result.json'
        results.append(result)
    report = {**provenance, **summarize_repair(results, inputs), 'results': results,
              'render_condition_effect': 'state and physics only; no RGB claim'}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    return report


def select_repair(config, run_dir, output_dir):
    run = Path(run_dir)
    _verify_run(run)
    checkpoint = load_checkpoint(run / 'last.pt')
    if not checkpoint['training_state']['finished'] or (run / 'selected.pt').exists():
        raise ValueError('Place repair selection needs a complete unselected run')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    reports = []
    for epoch, item in sorted(checkpoint['training_state']['candidates'].items(),
                               key=lambda value: int(value[0])):
        print(f'Place repair validation candidate {epoch}', flush=True)
        reports.append(evaluate_repair(config, run / item['path'],
            directory / f'epoch_{int(epoch):04d}', split='validation'))
    _checksum_index(run)
    selected = select_checkpoint(run, reports)
    _checksum_index(directory)
    return selected
