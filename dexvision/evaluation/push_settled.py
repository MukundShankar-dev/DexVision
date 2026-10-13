"""Versioned ordinary-physics entry hold around exact V2 push weights; checkpoint 5.5."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from dexvision.evaluation.contact_policy import (
    contact_scenarios, create_contact_task, read_contact_entries, run_contact_rollout, summarize_contact,
)
from dexvision.evaluation.evaluate_skill import apply_action
from dexvision.evaluation.push_repair import (
    _summary_inputs, confirmation_inputs, freeze_push_confirmation,
)
from dexvision.learning.push_repair import MODULES, PushPolicy, load_push_spec, runtime_spec
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, select_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file

CONFIG = 'configs/level5/push_policy_v3.yaml'


def load_settled_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Settled push recipe checksum mismatch')
    spec = yaml.safe_load(path.read_text())
    parent, inputs = load_push_spec(spec['parent_learning_recipe'])
    setup, setup_inputs = load_push_spec(spec['setup_recipe'])
    if (spec['version'] != 'level5/push-policy-v3' or spec['checkpoint'] != '5.5'
            or spec['setup_hold_samples'] != 120 or setup_inputs.digests != inputs.digests
            or spec['parent_learning_recipe_sha256'] != digest_file(Path(spec['parent_learning_recipe']))
            or spec['setup_recipe_sha256'] != digest_file(Path(spec['setup_recipe']))
            or setup['output_dir'] != spec['output_dir']):
        raise ValueError('Settled push parent, setup or immutable hold mismatch')
    return spec, parent, setup, inputs


def settled_prefix(prefix, count):
    prefix = np.asarray(prefix, float)
    if prefix.ndim != 2 or prefix.shape[1] != 27 or len(prefix) == 0 or not np.isfinite(prefix).all():
        raise ValueError('Entry hold needs a finite nonempty prefix')
    return np.concatenate((prefix, np.repeat(prefix[-1:], count, axis=0)))


def freeze_settled_confirmation(config, release_root=None):
    spec, _, setup, _ = load_settled_spec(config)
    root = Path(spec['output_dir'])
    if list(root.glob('seed_*/selected.pt')):
        raise ValueError('Freeze settled confirmation before selection')
    raw = root / 'confirmation_raw'
    freeze_push_confirmation(spec['setup_recipe'], raw, release_root)
    frozen = json.loads((raw / 'resets.json').read_text())
    for record in frozen['records']:
        record['original_prefix_length'] = len(record['prefix_actions'])
        if record['prefix_actions']:
            record['prefix_actions'] = settled_prefix(record['prefix_actions'], spec['setup_hold_samples']).tolist()
        # All setup failures are retained; no search or replacement of a seed.
    for cell in frozen['matrix']['cells']:
        by_id = {r['row']['episode_id']: len(r['prefix_actions']) for r in frozen['records']}
        cell['start_frames'] = [by_id[eid] for eid in cell['episode_ids']]
    frozen.update(overlay_recipe_sha256=content_digest(spec), setup_hold_samples=spec['setup_hold_samples'],
        raw_resets_sha256=digest_file(raw / 'resets.json'), recipe_sha256=content_digest(setup))
    directory = root / 'confirmation_resets'
    directory.mkdir(parents=True, exist_ok=False)
    _json(directory / 'resets.json', frozen)
    _checksum_index(directory)
    return {'frozen': len(frozen['records']), 'hold_samples': spec['setup_hold_samples']}


def execution_pins(inputs):
    return {name: digest_file(inputs.root / name)
        for name in (*MODULES, 'dexvision/evaluation/push_settled.py')}


def prepare_settled_run(config, seed):
    """Copy exact learned artifacts into a separate setup/selection bundle; preserve V2."""
    spec, parent, setup, inputs = load_settled_spec(config)
    source = Path(spec['source_runs']) / f'seed_{seed}'
    _verify_run(source)
    last = load_checkpoint(source / 'last.pt')
    if (seed not in parent['seeds'] or not last['training_state']['finished']
            or last['training_state']['completed_epochs'] != 200
            or last['training_state']['global_step'] != 8400
            or last['manifest']['contact_recipe'] != parent
            or (source / 'test_state_attempt.json').exists()):
        raise ValueError('Settled push requires untouched full-budget V2 source')
    PushPolicy(source / 'last.pt', parent, inputs)
    # Require all V2 negative candidate evidence before starting the new selector.
    previous = Path(parent['output_dir']) / f'validation_{seed}'
    _verify_run(previous)
    for epoch in last['training_state']['candidates']:
        report = json.loads((previous / f'epoch_{int(epoch):04d}' / 'report.json').read_text())
        if report['rollout_count'] != 120:
            raise ValueError('Incomplete V2 candidate evidence')
    _, _, _, frozen = confirmation_inputs(setup, inputs)
    if frozen['overlay_recipe_sha256'] != content_digest(spec):
        raise ValueError('Settled confirmation belongs to another overlay')
    run = Path(spec['output_dir']) / f'seed_{seed}'
    run.mkdir(parents=True, exist_ok=False)
    for name in ('last.pt', 'last.pt.sha256', 'best_offline.pt', 'best_offline.pt.sha256',
                 'training_manifest.json', 'normalization.json', 'environment.json', 'config_snapshot.yaml'):
        shutil.copyfile(source / name, run / name)
    shutil.copytree(source / 'validation_candidates', run / 'validation_candidates')
    _json(run / 'selection.json', {'status': 'pending_validation_rollouts',
        'candidates': last['training_state']['candidates'], 'test_used': False})
    _json(run / 'entry_overlay.json', {'recipe': spec, 'recipe_sha256': content_digest(spec),
        'source_run': str(source), 'source_run_sha256': digest_file(source / 'SHA256SUMS'),
        'source_last_sha256': digest_file(source / 'last.pt'),
        'confirmation_resets_sha256': digest_file(Path(spec['output_dir']) / 'confirmation_resets/resets.json'),
        'execution_sha256': execution_pins(inputs)})
    _checksum_index(run)
    return run


def verify_settled_bundle(config, checkpoint_path):
    spec, parent, setup, inputs = load_settled_spec(config)
    path = Path(checkpoint_path)
    run = path.parent.parent if path.parent.name == 'validation_candidates' else path.parent
    _verify_run(run)
    overlay = json.loads((run / 'entry_overlay.json').read_text())
    if (overlay['recipe'] != spec or overlay['recipe_sha256'] != content_digest(spec)
            or overlay['execution_sha256'] != execution_pins(inputs)
            or overlay['confirmation_resets_sha256'] != digest_file(
                Path(spec['output_dir']) / 'confirmation_resets/resets.json')
            or digest_file(Path(overlay['source_run']) / 'last.pt') != overlay['source_last_sha256']):
        raise ValueError('Settled push execution/source bundle mismatch')
    return spec, parent, setup, inputs, run, PushPolicy(path, parent, inputs), overlay


def evaluate_settled(config, checkpoint_path, output_dir, *, split='test', release_root=None):
    spec, parent, setup, inputs, run, policy, overlay = verify_settled_bundle(config, checkpoint_path)
    path, directory = Path(checkpoint_path), Path(output_dir)
    checkpoint = load_checkpoint(path)
    manifest = checkpoint['manifest']
    if split == 'test':
        selection = json.loads((run / 'selection.json').read_text())
        if path.name != 'selected.pt' or selection['status'] != 'selected_from_validation':
            raise ValueError('Settled confirmation requires selected weights')
        candidate_path = run / 'validation_candidates' / f"epoch_{selection['epoch']:04d}.pt"
        candidate = load_checkpoint(candidate_path)
        if digest_file(candidate_path) != selection['source_checkpoint_sha256'] or any(
                not torch.equal(v, candidate['model'][k]) for k, v in checkpoint['model'].items()):
            raise ValueError('Settled selected weights differ from candidate')
        if (run / 'test_state_attempt.json').exists():
            raise ValueError('Settled confirmation already reserved')
        inputs, scenarios, entries, frozen = confirmation_inputs(setup, inputs)
        summary_inputs = _summary_inputs(inputs, frozen)
        setup_reasons = {r['row']['episode_id']: r['setup_reason'] for r in frozen['records']}
    else:
        scenarios = contact_scenarios(inputs, split)
        entries = read_contact_entries(inputs, scenarios, release_root)
        entries = {eid: (metadata, settled_prefix(prefix, spec['setup_hold_samples']))
            for eid, (metadata, prefix) in entries.items()}
        summary_inputs = inputs
        setup_reasons = {row['episode_id']: 'ready' for row, _, _, _ in scenarios}
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': spec['version'], 'skill': inputs.skill['skill'], 'track': 'state', 'split': split,
        'seed': manifest['seed'], 'epoch': checkpoint['training_state']['completed_epochs'],
        'checkpoint_sha256': digest_file(path), 'checkpoint_path': str(path.resolve()),
        'output_dir': str(directory.resolve()), 'manifest_digest': manifest['manifest_digest'],
        'validation_matrix_sha256': manifest['validation_matrix_sha256'],
        'parent_matrix_sha256': content_digest(inputs.skill['evaluation'][split]),
        'execution_matrix_sha256': content_digest([{'episode_id': row['episode_id'], 'condition': condition,
            'prefix_sha256': content_digest(entries[row['episode_id']][1].tolist())}
            for row, _, _, condition in scenarios]),
        'recipe_sha256': content_digest(spec), 'entry_overlay_sha256': digest_file(run / 'entry_overlay.json'),
        'setup_hold_samples': spec['setup_hold_samples'], 'input_digests': inputs.digests,
        'confirmation_resets_sha256': overlay['confirmation_resets_sha256'], 'selection_uses_test': False}
    _json(directory / 'provenance.json', provenance)
    if split == 'test':
        with (run / 'test_state_attempt.json').open('x') as stream:
            json.dump(provenance, stream, indent=2)
    results = []
    for index, (row, _, _, condition) in enumerate(scenarios):
        if index % 20 == 0:
            print(f'Settled push {split} seed {manifest["seed"]}: {index}/{len(scenarios)}', flush=True)
        child = directory / f"{index:03d}_{row['episode_id']}_{condition}"
        current_policy = policy if setup_reasons[row['episode_id']] == 'ready' else (
            lambda source, prior, **kwargs: (np.full(27, np.nan), np.zeros(26)))
        result = run_contact_rollout(inputs, runtime_spec(parent), row, *entries[row['episode_id']], current_policy,
            condition=condition, output_dir=child)
        result['result_path'] = f'{child.name}/result.json'
        results.append(result)
    report = {**provenance, **summarize_contact(results, summary_inputs, split), 'results': results,
        'training_history': checkpoint['training_state']['history']}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    if split == 'test':
        _checksum_index(run)
    return report


def select_settled(config, source_run, output_dir, release_root=None):
    spec, _, _, _ = load_settled_spec(config)
    source = Path(source_run)
    last = load_checkpoint(source / 'last.pt')
    seed = last['manifest']['seed']
    if source.resolve() != (Path(spec['source_runs']) / f'seed_{seed}').resolve():
        raise ValueError('Settled selection uses the declared exact source run')
    run = prepare_settled_run(config, seed)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    reports = []
    for epoch, item in sorted(last['training_state']['candidates'].items(), key=lambda x: int(x[0])):
        print(f'Settled push validation candidate {epoch}', flush=True)
        reports.append(evaluate_settled(config, run / item['path'], directory / f'epoch_{int(epoch):04d}',
            split='validation', release_root=release_root))
    result = select_checkpoint(run, reports)
    _checksum_index(directory)
    return result


def review_settled(config, checkpoint_path):
    import time
    import mujoco.viewer
    spec, _, setup, inputs, run, _, _ = verify_settled_bundle(config, checkpoint_path)
    root = Path(spec['output_dir'])
    for seed in (5001, 5002, 5003):
        other = root / f'seed_{seed}'
        _verify_run(other)
        attempt = json.loads((other / 'test_state_attempt.json').read_text())
        directory = Path(attempt['output_dir'])
        _verify_run(directory)
        report = json.loads((directory / 'report.json').read_text())
        if (not report['state_numerical_gates_passed']
                or report['checkpoint_sha256'] != digest_file(other / 'selected.pt')
                or report['recipe_sha256'] != content_digest(spec)):
            raise ValueError('Settled viewer requires all three passing fresh confirmations')
    attempt = json.loads((run / 'test_state_attempt.json').read_text())
    directory = Path(attempt['output_dir'])
    report = json.loads((directory / 'report.json').read_text())
    inputs, _, _, frozen = confirmation_inputs(setup, inputs)
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


def run_settled_evaluation(config, *, checkpoint, output_dir, release_root=None, track='state', split='test',
                           select_candidates=None, viewer=False, scenario_set=None):
    if track != 'state':
        raise ValueError('Settled push implements state only')
    if viewer:
        if scenario_set != 'manual_stratified' or checkpoint is None or select_candidates:
            raise ValueError('Settled viewer requires selected checkpoint and manual_stratified')
        return review_settled(config, checkpoint)
    if scenario_set is not None or output_dir is None:
        raise ValueError('Settled push requires new output directory')
    if select_candidates is not None:
        if checkpoint is not None:
            raise ValueError('Choose checkpoint or candidate selection')
        return select_settled(config, select_candidates, output_dir, release_root)
    if checkpoint is None:
        raise ValueError('Settled push requires checkpoint')
    return evaluate_settled(config, checkpoint, output_dir, split=split, release_root=release_root)
