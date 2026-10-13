"""Frozen mixed validation and fresh confirmation for the 5.5 distance repair."""
from __future__ import annotations

import copy
import json
import math
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dexvision.evaluation.contact_policy import (
    contact_scenarios, create_contact_task, read_contact_entries, run_contact_rollout, summarize_contact,
)
from dexvision.evaluation.evaluate_skill import apply_action
from dexvision.evaluation.push_repair import freeze_push_confirmation
from dexvision.evaluation.push_settled import settled_prefix
from dexvision.learning.push_guarded import (
    PushPolicy, audit_decoder_events, execution_pins, load_push_spec, runtime_spec,
)
from dexvision.learning.push_predicates import PushPolicy as ParentPolicy
from dexvision.learning.push_predicates import load_push_spec as load_parent_spec
from dexvision.learning.skill_datasets import feature_schema
from dexvision.evaluation.pick_policy import physical_evidence_world
from dexvision.evaluation.contact_policy import physical_metrics
from dexvision.learning.train_skill import _checkpoint, _checksum_index, _json, _verify_run, load_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file


def prepare_guarded_run(config, seed):
    """Preserve exact full-budget V7 candidates in a separately pinned overlay."""
    spec, inputs = load_push_spec(config)
    parent, _ = load_parent_spec(spec['parent_learning_recipe'])
    source = Path(spec['source_runs']) / f'seed_{seed}'
    _verify_run(source)
    last = load_checkpoint(source / 'last.pt')
    state = last['training_state']
    if seed not in spec['seeds'] or not state['finished'] or state['completed_epochs'] != 200 or state['global_step'] != 8400:
        raise ValueError('Guarded push requires complete V7 training')
    ParentPolicy(source / 'last.pt', parent, inputs)
    previous = Path(parent['output_dir']) / f'validation_{seed}'
    _verify_run(previous)
    for epoch in state['candidates']:
        report = json.loads((previous / f'epoch_{int(epoch):04d}' / 'report.json').read_text())
        if report['rollout_count'] != 184:
            raise ValueError('Incomplete V7 candidate evidence')
    previous_test = Path(parent['output_dir']) / f'confirmation_{seed}'
    _verify_run(previous_test)
    if json.loads((previous_test / 'report.json').read_text())['rollout_count'] != 128:
        raise ValueError('Incomplete preserved V7 confirmation')
    validation, confirmation = (read_entry_matrix(spec, inputs, split) for split in ('validation', 'test'))
    run = Path(spec['output_dir']) / f'seed_{seed}'
    run.mkdir(parents=True, exist_ok=False)
    for name in ('last.pt', 'last.pt.sha256', 'best_offline.pt', 'best_offline.pt.sha256',
                 'training_manifest.json', 'normalization.json', 'environment.json', 'config_snapshot.yaml'):
        shutil.copyfile(source / name, run / name)
    shutil.copytree(source / 'validation_candidates', run / 'validation_candidates')
    _json(run / 'selection.json', {'status': 'pending_validation_rollouts', 'candidates': state['candidates'], 'test_used': False})
    _json(run / 'entry_overlay.json', {'recipe': spec, 'recipe_sha256': content_digest(spec),
        'source_run': str(source), 'source_run_sha256': digest_file(source / 'SHA256SUMS'),
        'validation_matrix_sha256': content_digest(validation['matrix']),
        'validation_resets_sha256': digest_file(Path(spec['output_dir']) / 'validation_resets/resets.json'),
        'confirmation_resets_sha256': digest_file(Path(spec['output_dir']) / 'confirmation_resets/resets.json'),
        'confirmation_matrix_sha256': content_digest(confirmation['matrix']), 'execution_sha256': execution_pins(inputs)})
    _checksum_index(run)
    return run


def audit_phase_guards(inputs, record, child, events):
    """Check every engineered settling hold against separate saved-state physics."""
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    latch, holds, aliases = False, 0, 0
    with np.load(child / 'trajectory.npz', allow_pickle=False) as arrays, create_contact_task(
            inputs, record['row'], record['metadata'], np.asarray(record['prefix_actions']).reshape(-1, 27)) as task:
        observations = arrays['observations'] if 'observations' in arrays else np.empty((0, len(names)))
        for index, (event, observation) in enumerate(zip(events, observations, strict=True)):
            alias = observation[names.index('phase.approach')] > 0 and not latch
            latch |= bool(observation[names.index('phase.retract')] > 0)
            hold = observation[names.index('phase.settle')] > 0 and not latch
            if alias != event['entry_phase_alias'] or hold != event['settle_hold']:
                raise ValueError('Guarded push phase decisions disagree with causal observations')
            aliases += int(alias)
            if hold:
                if index:
                    apply_action(task, arrays['applied_actions'][index - 1], inputs.plan['action']['named_layout'])
                task.env.data.qpos[:] = arrays['qpos'][index]
                task.env.data.qvel[:] = arrays['qvel'][index]
                task.env.data.time = arrays['state_timestamps'][index]
                task.env._mujoco.mj_forward(task.env.model, task.env.data)
                _, qualifies = physical_metrics(task, physical_evidence_world(task))
                if not qualifies:
                    raise ValueError('Settling hold lacks independently qualifying physical state')
                holds += 1
    return {'passed': True, 'independently_qualified_hold_samples': holds, 'causal_alias_samples': aliases}


def freeze_entry_matrices(config, release_root=None):
    """Predeclare every setup before overlay selection; no policy outcome filtering."""
    spec, inputs = load_push_spec(config)
    root = Path(spec['output_dir'])
    if list(root.glob('seed_*/last.pt')):
        raise ValueError('Freeze distance repair matrices before training')
    for key in ('selection', 'setup'):
        path = Path(spec[f'{key}_recipe'])
        if digest_file(path) != spec[f'{key}_recipe_sha256']:
            raise ValueError('Distance repair setup recipe changed')
    freeze_push_confirmation(spec['selection_recipe'], root / 'selection_raw', release_root)
    freeze_push_confirmation(spec['setup_recipe'], root / 'confirmation_raw', release_root)
    original = contact_scenarios(inputs, 'validation')
    unique = {}
    for scenario in original:
        unique.setdefault(scenario[0]['episode_id'], scenario)
    entries = read_contact_entries(inputs, list(unique.values()), release_root)
    original_records = [{'row': row, 'metadata': entries[row['episode_id']][0],
        'prefix_actions': entries[row['episode_id']][1].tolist(), 'setup_reason': 'ready',
        'role': 'original_validation'} for row, _, _, _ in unique.values()]
    for split, raw, records in (
        ('validation', 'selection_raw', original_records), ('test', 'confirmation_raw', []),
    ):
        frozen = json.loads((root / raw / 'resets.json').read_text())
        records = [*records, *[{**r, 'role': 'fresh_selection' if split == 'validation' else 'fresh_confirmation'}
            for r in frozen['records']]]
        groups = defaultdict(list)
        for r in records:
            r['original_prefix_length'] = len(r['prefix_actions'])
            if r['prefix_actions']:
                r['prefix_actions'] = settled_prefix(r['prefix_actions'], spec['setup_hold_samples']).tolist()
            groups[r['row']['cell_id']].append(r)
        cells = [{'cell_id': key, 'episode_ids': [r['row']['episode_id'] for r in rows],
            'reset_seeds': [r['metadata']['random_seed'] for r in rows],
            'start_frames': [len(r['prefix_actions']) for r in rows]} for key, rows in sorted(groups.items())]
        matrix = {'cells': cells, 'unique_reset_count': len(records),
            'rollouts_per_track_per_training_seed': len(records) * len(
                inputs.config_snapshot['evaluation']['rollouts']['perception_conditions'])}
        expected = 62 if split == 'validation' else 32
        if len(records) != expected:
            raise ValueError('Incomplete frozen distance repair matrix')
        directory = root / ('validation_resets' if split == 'validation' else 'confirmation_resets')
        directory.mkdir(parents=True, exist_ok=False)
        _json(directory / 'resets.json', {'recipe_sha256': content_digest(spec), 'input_digests': inputs.digests,
            'split': split, 'records': records, 'matrix': matrix, 'raw_resets_sha256': digest_file(root / raw / 'resets.json'),
            'exposure': spec['confirmation']['disclosure']})
        _checksum_index(directory)
    return {'validation_unique_resets': 62, 'confirmation_unique_resets': 32}


def read_entry_matrix(spec, inputs, split):
    if split not in {'validation', 'test'}:
        raise ValueError('Distance repair needs validation or test')
    directory = Path(spec['output_dir']) / ('validation_resets' if split == 'validation' else 'confirmation_resets')
    _verify_run(directory)
    frozen = json.loads((directory / 'resets.json').read_text())
    if (frozen['recipe_sha256'] != content_digest(spec) or frozen['input_digests'] != inputs.digests
            or frozen['split'] != split):
        raise ValueError('Distance repair frozen matrix mismatch')
    return frozen


def matrix_inputs(inputs, frozen):
    """Synthetic one-frame enumeration intervals are evaluation-only, never BC targets."""
    split = frozen['split']
    skill, splits = copy.deepcopy(inputs.skill), copy.deepcopy(inputs.splits)
    skill['evaluation'][split] = frozen['matrix']
    rows = []
    for record in frozen['records']:
        row = copy.deepcopy(record['row'])
        start = len(record['prefix_actions'])
        row['segments'] = [{'skill_name': 'push_object_to_target', 'start_frame': start, 'end_frame': start + 1}]
        row['phase_intervals'] = [{'phase': 'push_contact', 'start_frame': start, 'end_frame': start + 1}]
        row['training_target_interval'] = [start, start + 1]
        row['frames'] = start + 1
        rows.append(row)
    splits[split]['episodes'] = rows
    return replace(inputs, skill=skill, splits=splits)


def evaluate_distance(config, checkpoint_path, output_dir, *, split='test', release_root=None):
    spec, original_inputs = load_push_spec(config)
    path, directory = Path(checkpoint_path), Path(output_dir)
    policy = PushPolicy(path, spec, original_inputs)
    checkpoint = load_checkpoint(path)
    manifest = checkpoint['manifest']
    run = path.parent.parent if path.parent.name == 'validation_candidates' else path.parent
    _verify_run(run)
    if split == 'test':
        selection = json.loads((run / 'selection.json').read_text())
        if path.name != 'selected.pt' or selection['status'] != 'selected_from_validation':
            raise ValueError('Distance confirmation requires validation-selected weights')
        candidate_path = run / 'validation_candidates' / f"epoch_{selection['epoch']:04d}.pt"
        candidate = load_checkpoint(candidate_path)
        if digest_file(candidate_path) != selection['source_checkpoint_sha256'] or any(
                not torch.equal(v, candidate['model'][k]) for k, v in checkpoint['model'].items()):
            raise ValueError('Distance selected weights differ from validation candidate')
        if (run / 'test_state_attempt.json').exists():
            raise ValueError('Distance confirmation already reserved')
    frozen = read_entry_matrix(spec, original_inputs, split)
    validation = read_entry_matrix(spec, original_inputs, 'validation')
    overlay = json.loads((run / 'entry_overlay.json').read_text())
    if (overlay['validation_matrix_sha256'] != content_digest(validation['matrix'])
            or overlay['validation_resets_sha256'] != digest_file(
                Path(spec['output_dir']) / 'validation_resets/resets.json')
            or overlay['confirmation_resets_sha256'] != digest_file(
                Path(spec['output_dir']) / 'confirmation_resets/resets.json')):
        raise ValueError('Distance matrices changed after training freeze')
    inputs = matrix_inputs(original_inputs, frozen)
    scenarios = contact_scenarios(inputs, split)
    records = {r['row']['episode_id']: r for r in frozen['records']}
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': spec['version'], 'skill': inputs.skill['skill'], 'track': 'state', 'split': split,
        'seed': manifest['seed'], 'epoch': checkpoint['training_state']['completed_epochs'],
        'checkpoint_sha256': digest_file(path), 'checkpoint_path': str(path.resolve()),
        'output_dir': str(directory.resolve()), 'manifest_digest': manifest['manifest_digest'],
        'validation_matrix_sha256': overlay['validation_matrix_sha256'],
        'entry_overlay_sha256': digest_file(run / 'entry_overlay.json'),
        'matrix_sha256': content_digest(frozen['matrix']), 'recipe_sha256': content_digest(spec),
        'input_digests': inputs.digests, 'selection_uses_test': False,
        'frozen_entries_sha256': digest_file(Path(spec['output_dir']) / (
            'validation_resets/resets.json' if split == 'validation' else 'confirmation_resets/resets.json'))}
    _json(directory / 'provenance.json', provenance)
    if split == 'test':
        with (run / 'test_state_attempt.json').open('x') as stream:
            json.dump(provenance, stream, indent=2)
    results = []
    for index, (row, _, _, condition) in enumerate(scenarios):
        if index % 20 == 0:
            print(f'Distance push {split} seed {manifest["seed"]}: {index}/{len(scenarios)}', flush=True)
        r = records[row['episode_id']]
        policy.reset()
        current_policy = policy if r['setup_reason'] == 'ready' else (
            lambda source, prior, **kwargs: (np.full(27, np.nan), np.zeros(26)))
        child = directory / f"{index:03d}_{row['episode_id']}_{condition}"
        result = run_contact_rollout(inputs, runtime_spec(spec), row, r['metadata'],
            np.asarray(r['prefix_actions']).reshape(-1, 27), current_policy, condition=condition, output_dir=child)
        _json(child / 'decoder_events.json', {'events': policy.events})
        result['decoder_audit'] = audit_decoder_events(policy.events, child / 'trajectory.npz', spec)
        result['phase_guard_audit'] = audit_phase_guards(inputs, r, child, policy.events)
        _json(child / 'result.json', result)
        _checksum_index(child)
        result['result_path'] = f'{child.name}/result.json'
        results.append(result)
    report = {**provenance, **summarize_contact(results, inputs, split), 'results': results,
        'training_history': checkpoint['training_state']['history']}
    report['engineered_execution_audits_passed'] = all(
        r['decoder_audit']['passed'] and r['phase_guard_audit']['passed'] for r in results)
    report['engineering_counters'] = {key: sum(r['decoder_audit'][key] for r in results)
        for key in ('entry_phase_alias_samples', 'settle_hold_samples', 'workspace_bound_samples')}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    if split == 'test':
        _checksum_index(run)
    return report


def select_guarded_checkpoint(run, reports, spec, inputs):
    """Apply the original safety-first ranking to the separately bound overlay matrix."""
    checkpoint = load_checkpoint(run / 'last.pt')
    candidates = checkpoint['training_state']['candidates']
    overlay = json.loads((run / 'entry_overlay.json').read_text())
    matrix = read_entry_matrix(spec, inputs, 'validation')['matrix']
    if len(reports) != len(candidates) or {str(r['epoch']) for r in reports} != set(candidates):
        raise ValueError('Guarded selection requires every candidate exactly once')
    for report in reports:
        candidate = candidates[str(report['epoch'])]
        if (report['split'] != 'validation' or report['checkpoint_sha256'] != candidate['checkpoint_sha256']
                or report['manifest_digest'] != checkpoint['manifest']['manifest_digest']
                or report['seed'] != checkpoint['manifest']['seed']
                or report['validation_matrix_sha256'] != overlay['validation_matrix_sha256']
                or report['rollout_count'] != matrix['rollouts_per_track_per_training_seed']
                or report['entry_overlay_sha256'] != digest_file(run / 'entry_overlay.json')
                or not report['engineered_execution_audits_passed']):
            raise ValueError('Guarded selection has incompatible execution provenance')
        for key in ('safety_violation_count', 'success_rate', 'worst_cell_success_rate'):
            if not isinstance(report[key], (int, float)) or not math.isfinite(report[key]) or report[key] < 0:
                raise ValueError('Invalid guarded selection metric')
        if report['success_rate'] > 1 or report['worst_cell_success_rate'] > 1:
            raise ValueError('Invalid guarded selection success rate')
        error = report['normalized_terminal_error']
        if error is None:
            if report.get('terminal_error_unavailable_reason') not in {'rejected_precondition', 'missing_terminal_metric'}:
                raise ValueError('Unavailable guarded terminal error requires a reason')
        elif not isinstance(error, (int, float)) or not math.isfinite(error) or error < 0:
            raise ValueError('Invalid guarded terminal error')
    chosen = min(reports, key=lambda r: (r['safety_violation_count'], -r['success_rate'],
        -r['worst_cell_success_rate'], r['normalized_terminal_error'] if r['normalized_terminal_error'] is not None else math.inf,
        candidates[str(r['epoch'])]['offline_validation_loss'], r['epoch']))
    _checkpoint(run / 'selected.pt', load_checkpoint(run / candidates[str(chosen['epoch'])]['path']))
    result = {'status': 'selected_from_validation', 'epoch': chosen['epoch'],
        'source_checkpoint_sha256': chosen['checkpoint_sha256'], 'reports': reports, 'test_used': False,
        'qualification_claim': None, 'entry_overlay_sha256': digest_file(run / 'entry_overlay.json')}
    _json(run / 'selection.json', result)
    _checksum_index(run)
    return result


def select_distance(config, run_dir, output_dir, release_root=None):
    run, directory = Path(run_dir), Path(output_dir)
    if not run.exists():
        spec, _ = load_push_spec(config)
        seed = int(run.name.removeprefix('seed_'))
        if run != Path(spec['output_dir']) / f'seed_{seed}':
            raise ValueError('Guarded selection requires the frozen run directory')
        prepare_guarded_run(config, seed)
    _verify_run(run)
    checkpoint = load_checkpoint(run / 'last.pt')
    if not checkpoint['training_state']['finished'] or (run / 'selected.pt').exists():
        raise ValueError('Distance selection requires finished unselected training')
    directory.mkdir(parents=True, exist_ok=False)
    reports = []
    for epoch, item in sorted(checkpoint['training_state']['candidates'].items(), key=lambda x: int(x[0])):
        print(f'Distance validation candidate {epoch}', flush=True)
        reports.append(evaluate_distance(config, run / item['path'], directory / f'epoch_{int(epoch):04d}',
            split='validation', release_root=release_root))
    spec, inputs = load_push_spec(config)
    result = select_guarded_checkpoint(run, reports, spec, inputs)
    _checksum_index(directory)
    return result


def review_distance(config, checkpoint_path):
    import time
    import mujoco.viewer
    spec, inputs = load_push_spec(config)
    path = Path(checkpoint_path)
    PushPolicy(path, spec, inputs)
    run = path.parent
    _verify_run(run)
    root = Path(spec['output_dir'])
    for seed in (5001, 5002, 5003):
        other = root / f'seed_{seed}'
        _verify_run(other)
        attempt = json.loads((other / 'test_state_attempt.json').read_text())
        directory = Path(attempt['output_dir'])
        _verify_run(directory)
        report = json.loads((directory / 'report.json').read_text())
        if (not report['state_numerical_gates_passed'] or not report['engineered_execution_audits_passed']
                or report['checkpoint_sha256'] != digest_file(other / 'selected.pt')
                or report['recipe_sha256'] != content_digest(spec)):
            raise ValueError('Settled viewer requires all three passing fresh confirmations')
    attempt = json.loads((run / 'test_state_attempt.json').read_text())
    directory = Path(attempt['output_dir'])
    report = json.loads((directory / 'report.json').read_text())
    frozen = read_entry_matrix(spec, inputs, 'test')
    inputs = matrix_inputs(inputs, frozen)
    rows = {r['row']['episode_id']: r['row'] for r in frozen['records']}
    chosen = {}
    for result in report['results']:
        if result['terminal_result'] == 'succeeded':
            chosen.setdefault(result['cell_id'], result)
    if set(chosen) != set(report['breakdowns']['cell_id']):
        raise ValueError('Settled viewer needs success in every confirmation cell')
    for result in chosen.values():
        child = directory / Path(result['result_path']).parent
        _verify_run(child)
        saved = json.loads((child / 'setup.json').read_text())
        with create_contact_task(inputs, rows[result['episode_id']], saved['metadata'],
                np.asarray(saved['prefix_actions'])) as task:
            print(f"Push review {result['cell_id']}: {result['terminal_reason']}", flush=True)
            with np.load(child / 'trajectory.npz', allow_pickle=False) as data, \
                    mujoco.viewer.launch_passive(task.env.model, task.env.data) as viewer:
                for index, (qpos, qvel) in enumerate(zip(data['qpos'], data['qvel'], strict=True)):
                    if not viewer.is_running():
                        return {'manual_acceptance': 'pending_owner_confirmation', 'cancelled': True}
                    if index:
                        apply_action(task, data['applied_actions'][index - 1], inputs.plan['action']['named_layout'])
                    task.env.data.qpos[:] = qpos
                    task.env.data.qvel[:] = qvel
                    task.env.data.time = data['state_timestamps'][index]
                    task.env._mujoco.mj_forward(task.env.model, task.env.data)
                    viewer.sync()
                    time.sleep(inputs.plan['action']['control_interval_s'])
    return {'manual_acceptance': 'pending_owner_confirmation', 'reviewed': len(chosen)}



def run_distance_evaluation(config, *, checkpoint, output_dir, release_root=None, track='state', split='test',
                           select_candidates=None, viewer=False, scenario_set=None):
    if track != 'state':
        raise ValueError('Settled push implements state only')
    if viewer:
        if scenario_set != 'manual_stratified' or checkpoint is None or select_candidates:
            raise ValueError('Settled viewer requires selected checkpoint and manual_stratified')
        return review_distance(config, checkpoint)
    if scenario_set is not None or output_dir is None:
        raise ValueError('Settled push requires new output directory')
    if select_candidates is not None:
        if checkpoint is not None:
            raise ValueError('Choose checkpoint or candidate selection')
        return select_distance(config, select_candidates, output_dir, release_root)
    if checkpoint is None:
        raise ValueError('Settled push requires checkpoint')
    return evaluate_distance(config, checkpoint, output_dir, split=split, release_root=release_root)
