"""Full validation and one-time fresh confirmation for checkpoint 5.5 push."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dexvision.evaluation.contact_policy import (
    contact_scenarios, create_contact_task, read_contact_entries, run_contact_rollout,
    summarize_contact,
)
from dexvision.evaluation.evaluate_skill import apply_action, safe_action
from dexvision.learning.contact_policy import contact_action_plan
from dexvision.learning.push_repair import PushPolicy, load_push_spec, runtime_spec
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, select_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file
from dexvision.logging.level4_collection import WorkcellPilotTask, sample_level4_procedural_variation
from dexvision.sim.level4_expert import DeterministicPushConfig, DeterministicPushExpert
from dexvision.sim.scripted_expert_settings import resolve_scripted_expert_settings


def freeze_push_confirmation(config, output_dir, release_root=None):
    """Freeze all fresh resets without querying any learned controller or filtering failures."""
    spec, inputs = load_push_spec(config)
    if list(Path(spec['output_dir']).glob('seed_*/selected.pt')):
        raise ValueError('Freeze push confirmation before selection')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    scenarios = contact_scenarios(inputs, 'test')
    by_cell = {}
    for row, _, _, _ in scenarios:
        by_cell.setdefault(row['cell_id'], row)
    first_scenarios = {}
    for scenario in scenarios:
        first_scenarios.setdefault(scenario[0]['cell_id'], scenario)
    entries = read_contact_entries(inputs, list(first_scenarios.values()), release_root)
    records, cells = [], []
    plan = contact_action_plan(spec, inputs)
    for index, (cell_id, source_row) in enumerate(by_cell.items()):
        source_metadata, source_prefix = entries[source_row['episode_id']]
        cell = {'cell_id': cell_id, 'episode_ids': [], 'reset_seeds': [], 'start_frames': []}
        for repetition in range(spec['confirmation']['resets_per_source_cell']):
            seed = spec['confirmation']['fresh_reset_seed_start'] + index * 100 + repetition
            variation = sample_level4_procedural_variation(inputs.requirements['level4_5b_procedural_expansion'],
                skill_name='push_object_to_target', seed=seed, boundary_case=cell_id.endswith('boundary'))
            with WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
                    dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
                    skill_name='push_object_to_target', goal_condition_id=cell_id, seed=seed,
                    procedural_variation=variation) as task:
                metadata = copy.deepcopy(source_metadata)
                eid = f'push_v2_confirmation_{seed}'
                metadata.update(episode_id=eid, random_seed=seed, typed_goal=task.goal,
                    recording_session_id=f'evaluation_only_{cell_id}', task_config=task.metadata_task_config(),
                    initial_online_phase='approach', source='fresh_reset_scripted_preentry_only')
                row = {**source_row, 'episode_id': eid, 'random_seed': seed, 'typed_goal': task.goal,
                    'recording_session_id': metadata['recording_session_id'], 'episode_digest': content_digest(metadata)}
                fingers = dict(zip(source_metadata['finger_target_names'], source_prefix[-1, 7:], strict=True))
                expert = DeterministicPushExpert(finger_targets=fingers,
                    config=DeterministicPushConfig.from_mapping(resolve_scripted_expert_settings(task, 'scripted_push')))
                expert.reset(task, task.workcell.get_world_state())
                prefix, prior, reason = [], np.asarray(metadata['initial_applied_action']), 'setup_timeout'
                for _ in range(500):
                    requested, phase, done, error = expert.step(task.workcell.get_world_state())
                    if error:
                        reason = error
                        break
                    # Strictly exclude the first contact-phase expert command.
                    if phase == 'push_contact':
                        reason = 'ready'
                        break
                    action, _, _, unsafe = safe_action(requested.as_array(), prior, plan)
                    if unsafe:
                        reason = unsafe
                        break
                    apply_action(task, action, plan['action']['named_layout'])
                    task.step(n_steps=inputs.plan['action']['simulation_steps_per_action'])
                    prefix.append(action.copy())
                    prior = action
                    if done:
                        reason = 'setup_terminated'
                        break
            prefix = np.asarray(prefix).reshape(-1, 27)
            records.append({'row': row, 'metadata': metadata, 'prefix_actions': prefix.tolist(),
                'setup_reason': reason, 'prefix_source_episode': source_row['episode_id'],
                'prefix_source_digest': source_row['episode_digest']})
            cell['episode_ids'].append(eid)
            cell['reset_seeds'].append(seed)
            cell['start_frames'].append(len(prefix))
            print(f'Frozen fresh push reset {seed}: {reason}, {len(prefix)} prefix actions', flush=True)
        cells.append(cell)
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    _json(directory / 'resets.json', {'recipe_sha256': content_digest(spec), 'input_digests': inputs.digests,
        'records': records, 'disclosure': spec['confirmation']['disclosure'],
        'matrix': {'cells': cells, 'unique_reset_count': len(records),
            'rollouts_per_track_per_training_seed': len(records) * len(conditions)}})
    _checksum_index(directory)
    return {'frozen': len(records), 'setup_reasons': [r['setup_reason'] for r in records]}


def confirmation_inputs(spec, inputs):
    directory = Path(spec['output_dir']) / 'confirmation_resets'
    _verify_run(directory)
    frozen = json.loads((directory / 'resets.json').read_text())
    if frozen['recipe_sha256'] != content_digest(spec) or frozen['input_digests'] != inputs.digests:
        raise ValueError('Fresh push resets recipe/source mismatch')
    skill, splits = copy.deepcopy(inputs.skill), copy.deepcopy(inputs.splits)
    skill['evaluation']['test'] = frozen['matrix']
    splits['test']['episodes'] = [r['row'] for r in frozen['records']]
    replaced = replace(inputs, skill=skill, splits=splits)
    scenarios = contact_scenarios_fresh(replaced, frozen)
    entries = {r['row']['episode_id']: (r['metadata'], np.asarray(r['prefix_actions']).reshape(-1, 27))
               for r in frozen['records']}
    return replaced, scenarios, entries, frozen


def contact_scenarios_fresh(inputs, frozen):
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    scenarios = [(r['row'], r['row']['random_seed'], len(r['prefix_actions']), c)
        for r in frozen['records'] for c in conditions]
    if len(scenarios) != frozen['matrix']['rollouts_per_track_per_training_seed']:
        raise ValueError('Fresh push matrix incomplete')
    return scenarios


def evaluate_push(config, checkpoint_path, output_dir, *, split='test', release_root=None):
    spec, inputs = load_push_spec(config)
    path, directory = Path(checkpoint_path), Path(output_dir)
    policy = PushPolicy(path, spec, inputs)
    checkpoint = load_checkpoint(path)
    manifest = checkpoint['manifest']
    run = path.parent.parent if path.parent.name == 'validation_candidates' else path.parent
    if split == 'test':
        _verify_run(run)
        selection = json.loads((run / 'selection.json').read_text())
        if path.name != 'selected.pt' or selection['status'] != 'selected_from_validation':
            raise ValueError('Fresh push confirmation requires selected weights')
        candidate_path = run / 'validation_candidates' / f"epoch_{selection['epoch']:04d}.pt"
        candidate = load_checkpoint(candidate_path)
        if digest_file(candidate_path) != selection['source_checkpoint_sha256'] or any(
                not torch.equal(v, candidate['model'][k]) for k, v in checkpoint['model'].items()):
            raise ValueError('Selected push weights differ from validation candidate')
        if (run / 'test_state_attempt.json').exists():
            raise ValueError('Fresh push confirmation already reserved')
        inputs, scenarios, entries, frozen = confirmation_inputs(spec, inputs)
        if manifest['confirmation_resets_sha256'] != digest_file(
                Path(spec['output_dir']) / 'confirmation_resets' / 'resets.json'):
            raise ValueError('Push confirmation changed after training freeze')
    else:
        scenarios = contact_scenarios(inputs, split)
        entries = read_contact_entries(inputs, scenarios, release_root)
        frozen = None
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': 'level5/push-repair-evaluation-v2', 'skill': inputs.skill['skill'], 'track': 'state',
        'split': split, 'seed': manifest['seed'], 'epoch': checkpoint['training_state']['completed_epochs'],
        'checkpoint_sha256': digest_file(path), 'checkpoint_path': str(path.resolve()),
        'output_dir': str(directory.resolve()), 'manifest_digest': manifest['manifest_digest'],
        'validation_matrix_sha256': manifest['validation_matrix_sha256'],
        'matrix_sha256': content_digest(inputs.skill['evaluation'][split]), 'input_digests': inputs.digests,
        'recipe_sha256': content_digest(spec), 'selection_uses_test': False,
        'confirmation_resets_sha256': digest_file(Path(spec['output_dir']) / 'confirmation_resets' / 'resets.json')
            if frozen else None}
    _json(directory / 'provenance.json', provenance)
    if split == 'test':
        with (run / 'test_state_attempt.json').open('x') as stream:
            json.dump(provenance, stream, indent=2)
    results = []
    # The physical evaluator remains the accepted V1 code; only the policy changes.
    for index, (row, _, _, condition) in enumerate(scenarios):
        if index % 20 == 0:
            print(f'Push V2 {split} seed {manifest["seed"]}: {index}/{len(scenarios)}', flush=True)
        child = directory / f"{index:03d}_{row['episode_id']}_{condition}"
        result = run_contact_rollout(inputs, runtime_spec(spec), row, *entries[row['episode_id']], policy,
            condition=condition, output_dir=child)
        result['result_path'] = f'{child.name}/result.json'
        results.append(result)
    # summarize_contact calls the historical interval-aware matrix enumerator.
    # Fresh rows retain an explicit synthetic contact interval of length one solely
    # for matrix enumeration, never for supervision or action execution.
    if frozen:
        expected = {(r['episode_id'], c) for r, _, _, c in scenarios}
        if len(results) != len(expected) or {(r['episode_id'], r['render_condition']) for r in results} != expected:
            raise ValueError('Incomplete fresh push attempts')
        summary_inputs = _summary_inputs(inputs, frozen)
    else:
        summary_inputs = inputs
    report = {**provenance, **summarize_contact(results, summary_inputs, split), 'results': results,
        'training_history': checkpoint['training_state']['history']}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    if split == 'test':
        _checksum_index(run)
    return report


def _summary_inputs(inputs, frozen):
    skill, splits = copy.deepcopy(inputs.skill), copy.deepcopy(inputs.splits)
    for row in splits['test']['episodes']:
        start = next(len(r['prefix_actions']) for r in frozen['records'] if r['row']['episode_id'] == row['episode_id'])
        row['segments'] = [{'skill_name': 'push_object_to_target', 'start_frame': start, 'end_frame': start + 1}]
        row['phase_intervals'] = [{'phase': 'push_contact', 'start_frame': start, 'end_frame': start + 1}]
        row['training_target_interval'] = [start, start + 1]
        row['frames'] = start + 1
    return replace(inputs, skill=skill, splits=splits)


def select_push(config, run_dir, output_dir, release_root=None):
    run, directory = Path(run_dir), Path(output_dir)
    _verify_run(run)
    checkpoint = load_checkpoint(run / 'last.pt')
    if not checkpoint['training_state']['finished'] or (run / 'selected.pt').exists():
        raise ValueError('Push selection requires completed unselected training')
    directory.mkdir(parents=True, exist_ok=False)
    reports = []
    for epoch, item in sorted(checkpoint['training_state']['candidates'].items(), key=lambda x: int(x[0])):
        print(f'Push V2 validation candidate {epoch}', flush=True)
        reports.append(evaluate_push(config, run / item['path'], directory / f'epoch_{int(epoch):04d}',
            split='validation', release_root=release_root))
    result = select_checkpoint(run, reports)
    _checksum_index(directory)
    return result


def review_push(config, checkpoint_path):
    import time
    import mujoco.viewer
    spec, inputs = load_push_spec(config)
    path = Path(checkpoint_path)
    PushPolicy(path, spec, inputs)
    _verify_run(path.parent)
    attempt = json.loads((path.parent / 'test_state_attempt.json').read_text())
    directory = Path(attempt['output_dir'])
    _verify_run(directory)
    report = json.loads((directory / 'report.json').read_text())
    if report['checkpoint_sha256'] != digest_file(path) or not report['state_numerical_gates_passed']:
        raise ValueError('Push viewer requires passing fresh numerical and physical confirmation')
    inputs, _, _, frozen = confirmation_inputs(spec, inputs)
    rows = {r['row']['episode_id']: r['row'] for r in frozen['records']}
    chosen = {}
    for r in report['results']:
        if r['terminal_result'] == 'succeeded':
            chosen.setdefault(r['cell_id'], r)
    if set(chosen) != set(report['breakdowns']['cell_id']):
        raise ValueError('Push viewer requires a saved success in every confirmation cell')
    for result in chosen.values():
        child = directory / Path(result['result_path']).parent
        _verify_run(child)
        setup = json.loads((child / 'setup.json').read_text())
        with create_contact_task(inputs, rows[result['episode_id']], setup['metadata'], np.asarray(setup['prefix_actions'])) as task:
            print(f"Push V2 review {result['cell_id']}: {result['terminal_reason']}", flush=True)
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


def run_push_evaluation(config, *, checkpoint, output_dir, release_root=None, track='state', split='test',
                        select_candidates=None, viewer=False, scenario_set=None):
    if track != 'state':
        raise ValueError('Push V2 implements state only')
    if viewer:
        if scenario_set != 'manual_stratified' or checkpoint is None or select_candidates:
            raise ValueError('Push viewer requires selected checkpoint and manual_stratified')
        return review_push(config, checkpoint)
    if scenario_set is not None or output_dir is None:
        raise ValueError('Push V2 requires new output directory')
    if select_candidates is not None:
        if checkpoint is not None:
            raise ValueError('Choose selected checkpoint or candidate selection')
        return select_push(config, select_candidates, output_dir, release_root)
    if checkpoint is None:
        raise ValueError('Push evaluation requires checkpoint')
    return evaluate_push(config, checkpoint, output_dir, split=split, release_root=release_root)
