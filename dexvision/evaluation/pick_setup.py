"""Preregistered setup correction using immutable V4 weights and new reset seeds."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from dexvision.evaluation.pick_policy import (
    create_pick_task, pick_preconditions, pick_scenarios, read_pick_entries, run_pick_rollout, summarize_pick,
)
from dexvision.evaluation.pick_repair import conditioned_pick_prefix, expert_pick_prefix, physical_pick_audit, verify_pick_selection
from dexvision.learning.pick_policy import PickPolicy, load_pick_spec
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run
from dexvision.learning.training_manifest import content_digest, digest_file
from dexvision.logging.level4_collection import WorkcellPilotTask, sample_level4_procedural_variation


def load_setup_spec(path):
    path = Path(path)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Pick setup protocol checksum mismatch')
    protocol = yaml.safe_load(path.read_text())
    if protocol['version'] not in {'level5/pick-setup-confirmation-v1', 'level5/pick-setup-confirmation-v2', 'level5/pick-setup-confirmation-v3', 'level5/pick-setup-confirmation-v4'} or protocol['checkpoint'] != '5.3A':
        raise ValueError('Unsupported pick setup protocol')
    if digest_file(Path(protocol['policy_recipe'])) != protocol['policy_recipe_sha256']:
        raise ValueError('Setup correction cannot change the frozen policy recipe')
    spec, inputs = load_pick_spec(protocol['policy_recipe'])
    if protocol['seeds'] != spec['seeds']:
        raise ValueError('Setup study must include all original training seeds')
    if protocol.get('setup_mode') == 'unassisted_exact_teacher':
        from dexvision.evaluation.pick_teacher import load_teacher_spec
        teacher, _, teacher_inputs = load_teacher_spec(protocol['teacher_recipe'])
        if (digest_file(Path(protocol['teacher_recipe'])) != protocol['teacher_recipe_sha256']
                or teacher_inputs.digests != inputs.digests or not teacher.get('exact_controller_preflight')):
            raise ValueError('Unassisted setup must bind the frozen exact teacher recipe and inputs')
    for seed in protocol['seeds']:
        item = protocol['selected_weights'][str(seed)]
        if digest_file(Path(item['path'])) != item['sha256']:
            raise ValueError('Setup correction cannot change or reselect policy weights')
    return protocol, spec, inputs


def _load_resets(protocol, inputs, reset_dir):
    reset_dir = Path(reset_dir)
    _verify_run(reset_dir)
    frozen = json.loads((reset_dir / 'resets.json').read_text())
    if frozen['protocol_sha256'] != content_digest(protocol) or frozen['input_digests'] != inputs.digests:
        raise ValueError('Conditioned reset manifest belongs to another protocol')
    for split, parent in (('validation', 'validation'), ('confirmation', 'test')):
        expected_cells = {c['cell_id'] for c in inputs.skill['evaluation'][parent]['cells']}
        records = frozen[split]
        ids = [r['row']['episode_id'] for r in records]
        if (len(ids) != len(set(ids)) or len(ids) != len(expected_cells) * protocol[split]['resets_per_cell']
                or {r['row']['cell_id'] for r in records} != expected_cells):
            raise ValueError('Incomplete conditioned reset matrix')
        for index, cell in enumerate(inputs.skill['evaluation'][parent]['cells']):
            seeds = {protocol[split]['seed_start'] + index * 100 + j for j in range(protocol[split]['resets_per_cell'])}
            if {r['metadata']['random_seed'] for r in records if r['row']['cell_id'] == cell['cell_id']} != seeds:
                raise ValueError('Conditioned reset seeds differ from preregistration')
    return frozen


def freeze_setup(config, output_dir, release_root=None):
    protocol, spec, inputs = load_setup_spec(config)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    result = {'version': protocol['version'], 'protocol_sha256': content_digest(protocol),
              'input_digests': inputs.digests, 'weights_frozen_before_setup_audit': protocol['selected_weights'],
              'disclosure': protocol['disclosure']}
    for split, parent in (('validation', 'validation'), ('confirmation', 'test')):
        scenarios = pick_scenarios(inputs, parent)
        entries = read_pick_entries(inputs, scenarios, release_root)
        rows = {row['episode_id']: row for row, _, _, _ in scenarios}
        records = []
        for cell_index, cell in enumerate(inputs.skill['evaluation'][parent]['cells']):
            source_row = rows[cell['episode_ids'][0]]
            source, source_prefix = entries[source_row['episode_id']]
            for repetition in range(protocol[split]['resets_per_cell']):
                seed = protocol[split]['seed_start'] + cell_index * 100 + repetition
                variation = sample_level4_procedural_variation(inputs.requirements['level4_5b_procedural_expansion'],
                    skill_name='pick_place_sequence', seed=seed, boundary_case=bool(repetition % 2))
                with WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
                        dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
                        skill_name='pick_place_sequence', goal_condition_id=cell['cell_id'], seed=seed,
                        procedural_variation=variation) as task:
                    metadata = copy.deepcopy(source)
                    metadata.update(episode_id=f'pick_conditioned_{seed}', random_seed=seed, typed_goal=task.goal,
                        task_config=task.metadata_task_config(), recording_session_id='evaluation_only_no_recording_session',
                        initial_online_phase='approach', source='fresh_conditioned_approach_not_demonstration')
                    row = {**source_row, 'episode_id': metadata['episode_id'], 'typed_goal': task.goal,
                           'episode_digest': content_digest(metadata), 'recording_session_id': metadata['recording_session_id']}
                    if protocol.get('setup_mode') == 'unassisted_exact_teacher':
                        from dexvision.evaluation.pick_teacher import load_teacher_spec, unassisted_pick_prefix
                        teacher, _, _ = load_teacher_spec(protocol['teacher_recipe'])
                        prefix, audit = unassisted_pick_prefix(task, teacher, spec, inputs)
                    elif protocol.get('setup_mode') == 'frozen_level4_expert':
                        prefix, audit = expert_pick_prefix(task, spec, inputs)
                    else:
                        prefix, audit = conditioned_pick_prefix(task, source, source_prefix, inputs,
                            allow_legacy_yaw=protocol.get('allow_legacy_source_yaw', False))
                    checks = pick_preconditions(task.workcell.get_world_state(), row, spec)
                records.append({'row': row, 'metadata': metadata, 'prefix_actions': prefix.tolist(),
                    'prefix_source_episode': source_row['episode_id'], 'prefix_source_digest': source_row['episode_digest'],
                    'entry_checks': checks, 'setup_audit': audit})
                print(f'Frozen {split} {seed}: entry={all(checks.values())}', flush=True)
        result[split] = records
    _json(directory / 'resets.json', result)
    _checksum_index(directory)
    _load_resets(protocol, inputs, directory)
    return {split: {'unique_resets': len(result[split]), 'valid_entries': sum(all(r['entry_checks'].values()) for r in result[split])}
            for split in ('validation', 'confirmation')}


def _expected_attempts(records, inputs):
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    return {(record['row']['episode_id'], condition) for record in records for condition in conditions}


def _verify_report(report, protocol, inputs, frozen, split, seed, reset_dir):
    item = protocol['selected_weights'][str(seed)]
    if (report['protocol_sha256'] != content_digest(protocol) or report['seed'] != seed or report['split'] != split
            or report['checkpoint_sha256'] != item['sha256']
            or report['reset_manifest_sha256'] != digest_file(Path(reset_dir) / 'resets.json')):
        raise ValueError('Conditioned report bindings differ from the preregistered study')
    summary = summarize_pick(report['results'], inputs, 'validation',
        expected_attempts=_expected_attempts(frozen[split], inputs))
    if any(report.get(k) != summary[k] for k in ('gates', 'state_numerical_gates_passed', 'rollout_count')):
        raise ValueError('Conditioned report does not contain the complete physical attempts')
    return summary


def evaluate_setup(config, reset_dir, output_dir, *, seed, split):
    protocol, spec, inputs = load_setup_spec(config)
    if seed not in protocol['seeds'] or split not in {'validation', 'confirmation'}:
        raise ValueError('Unknown setup study seed or split')
    frozen = _load_resets(protocol, inputs, reset_dir)
    for other in protocol['seeds']:
        verify_pick_selection(Path(protocol['selected_weights'][str(other)]['path']).parent, spec, inputs)
    root = Path(protocol['output_dir'])
    if split == 'confirmation':
        for other in protocol['seeds']:
            directory = root / f'validation_{other}'
            _verify_run(directory)
            report = json.loads((directory / 'report.json').read_text())
            summary = _verify_report(report, protocol, inputs, frozen, 'validation', other, reset_dir)
            if not summary['state_numerical_gates_passed']:
                raise ValueError('Every frozen model must pass conditioned validation before confirmation')
    checkpoint = Path(protocol['selected_weights'][str(seed)]['path'])
    directory = Path(output_dir)
    provenance = {'version': protocol['version'], 'protocol_sha256': content_digest(protocol),
        'seed': seed, 'split': split, 'skill': 'pick_object', 'track': 'state',
        'checkpoint_sha256': digest_file(checkpoint), 'checkpoint_path': str(checkpoint.resolve()),
        'policy_recipe_sha256': protocol['policy_recipe_sha256'], 'input_digests': inputs.digests,
        'reset_manifest_sha256': digest_file(Path(reset_dir) / 'resets.json'),
        'output_dir': str(directory.resolve()), 'disclosure': protocol['disclosure'],
        'weight_selection_uses_this_study': False,
        'execution_implementation_sha256': {p.relative_to(inputs.root).as_posix(): digest_file(p)
            for p in sorted((inputs.root / 'dexvision').rglob('*.py'))}}
    # Different protocol; original failing V4 reservation and reports are immutable.
    receipt = root / f'{split}_attempt_{seed}.json'
    if receipt.exists():
        raise ValueError('Conditioned protocol attempt already reserved; do not repeat')
    directory.mkdir(parents=True, exist_ok=False)
    with receipt.open('x') as stream:
        json.dump(provenance, stream, indent=2)
    _json(directory / 'provenance.json', provenance)
    policy = PickPolicy(checkpoint, spec, inputs)
    results = []
    for record in frozen[split]:
        row = record['row']
        for condition in inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']:
            child = directory / f"{len(results):03d}_{row['episode_id']}_{condition}"
            result = run_pick_rollout(inputs, spec, row, record['metadata'], np.asarray(record['prefix_actions']),
                policy, condition=condition, output_dir=child)
            result['result_path'] = f'{child.name}/result.json'
            results.append(result)
            if len(results) % 20 == 0:
                print(f'Conditioned pick {split} {seed}: {len(results)}', flush=True)
    report = {**provenance, **summarize_pick(results, inputs, 'validation',
        expected_attempts=_expected_attempts(frozen[split], inputs)), 'results': results}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    return {k: v for k, v in report.items() if k not in {'results', 'breakdowns', 'input_digests', 'execution_implementation_sha256'}}


def audit_setup(config, reset_dir, report_dir, output_dir, seed):
    protocol, spec, inputs = load_setup_spec(config)
    frozen = _load_resets(protocol, inputs, reset_dir)
    directory = Path(report_dir)
    _verify_run(directory)
    report = json.loads((directory / 'report.json').read_text())
    summary = _verify_report(report, protocol, inputs, frozen, 'confirmation', seed, reset_dir)
    if not summary['state_numerical_gates_passed']:
        raise ValueError('Physical qualification audit requires passing numerical confirmation')
    records = {r['row']['episode_id']: r for r in frozen['confirmation']}
    audits = []
    for result in report['results']:
        record = records[result['episode_id']]
        child = directory / Path(result['result_path']).parent
        with create_pick_task(inputs, record['row'], record['metadata'], np.asarray(record['prefix_actions'])) as task:
            with np.load(child / 'trajectory.npz', allow_pickle=False) as values:
                audit = physical_pick_audit(task, values, result, spec)
        audits.append({'episode_id': result['episode_id'], 'condition': result['render_condition'], **audit})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {'passed': all(a['passed'] for a in audits), 'attempts': len(audits), 'audits': audits,
              'report_sha256': digest_file(directory / 'report.json'), 'checkpoint_sha256': report['checkpoint_sha256'],
              'owner_acceptance': 'pending'}
    _json(output / 'audit.json', report)
    _checksum_index(output)
    return {k: v for k, v in report.items() if k != 'audits'}
