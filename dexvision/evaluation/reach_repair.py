"""Preregistered reach repair screening and validation; original reports stay read-only."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

from dexvision.evaluation.evaluate_skill import frozen_scenarios, run_reach_rollout, summarize_rollouts
from dexvision.learning.reach_repair import RepairPolicy, load_repair_spec
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, select_checkpoint
from dexvision.learning.training_manifest import ReleaseSource, content_digest, digest_file


def evaluate_repair_candidate(spec_path, checkpoint_path, output_dir, *, screen=False, release_root=None):
    spec, inputs = load_repair_spec(spec_path)
    policy = RepairPolicy(checkpoint_path, spec, inputs)
    checkpoint = load_checkpoint(checkpoint_path)
    scenarios = frozen_scenarios(inputs, 'validation')
    if screen:
        scenarios = [(row, seed, condition) for row, seed, condition in scenarios if condition in spec['screen']['conditions']]
    unique = {row['episode_id']: row for row, _, _ in scenarios}
    names = [f"data/demos/level4/{row['source_path']}/metadata.json" for row in unique.values()]
    blobs = ReleaseSource(inputs, release_root).read_many(names)
    metadata = {eid: json.loads(blobs[f"data/demos/level4/{row['source_path']}/metadata.json"]) for eid, row in unique.items()}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = checkpoint['manifest']
    provenance = {'version': 'level5/reach-repair-evaluation-v1', 'split': 'validation', 'track': 'state',
                  'screen_only': screen, 'seed': manifest['seed'], 'repair_arm': manifest['repair_arm'],
                  'epoch': checkpoint['training_state']['completed_epochs'], 'checkpoint_sha256': digest_file(checkpoint_path),
                  'manifest_digest': manifest['manifest_digest'], 'repair_spec_sha256': content_digest(spec),
                  'input_digests': inputs.digests, 'validation_matrix_sha256': manifest['validation_matrix_sha256'],
                  'execution_code_sha256': {p.relative_to(inputs.root).as_posix(): digest_file(p) for p in sorted((inputs.root / 'dexvision').rglob('*.py'))}}
    _json(output_dir / 'provenance.json', provenance)
    results = []
    for index, (row, seed, condition) in enumerate(scenarios):
        if metadata[row['episode_id']]['random_seed'] != seed:
            raise ValueError('Frozen validation reset mismatch')
        directory = output_dir / f"{index:03d}_{row['episode_id']}_{condition}"
        result = run_reach_rollout(inputs, row, metadata[row['episode_id']], policy, condition=condition, output_dir=directory)
        result['result_path'] = f'{directory.name}/result.json'
        results.append(result)
    summary_inputs = inputs
    if screen:
        skill = copy.deepcopy(inputs.skill)
        skill['evaluation']['validation']['rollouts_per_track_per_training_seed'] = len(scenarios)
        snapshot = copy.deepcopy(inputs.config_snapshot)
        snapshot['evaluation']['rollouts']['perception_conditions'] = spec['screen']['conditions']
        summary_inputs = replace(inputs, skill=skill, config_snapshot=snapshot)
    report = {**provenance, **summarize_rollouts(results, summary_inputs, 'validation'), 'results': results}
    if screen:
        report.update(state_qualified=False, qualification_claim=False, manual_acceptance='screen_only')
    _json(output_dir / 'report.json', report)
    _checksum_index(output_dir)
    return report


def evaluate_repair_run(spec_path, run_dir, output_dir, *, screen=False, release_root=None):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    _verify_run(run_dir)
    checkpoint = load_checkpoint(run_dir / 'last.pt')
    if not checkpoint['training_state']['finished']:
        raise ValueError('Repair training must finish before candidate evaluation')
    if not screen and (run_dir / 'selected.pt').exists():
        raise ValueError('This repair run has already selected a checkpoint')
    output_dir.mkdir(parents=True, exist_ok=False)
    reports = []
    for epoch, candidate in sorted(checkpoint['training_state']['candidates'].items(), key=lambda item: int(item[0])):
        print(f"{checkpoint['manifest']['repair_arm']} seed {checkpoint['manifest']['seed']}: {'screen' if screen else 'validation'} epoch {epoch}", flush=True)
        report = evaluate_repair_candidate(spec_path, run_dir / candidate['path'], output_dir / f'epoch_{int(epoch):04d}',
                                           screen=screen, release_root=release_root)
        reports.append(report)
        print(f"success={report['success_rate']:.3f}, safety={report['safety_violation_count']}, jerk={report['mean_action_jerk']:.6f}", flush=True)
    _checksum_index(run_dir)
    if not screen:
        result = select_checkpoint(run_dir, reports)
    else:
        candidates = checkpoint['training_state']['candidates']
        best = min(reports, key=lambda r: (r['safety_violation_count'], -r['success_rate'], -r['worst_cell_success_rate'],
            r['normalized_terminal_error'] if r['normalized_terminal_error'] is not None else float('inf'),
            candidates[str(r['epoch'])]['offline_validation_loss'], r['epoch']))
        result = {'screen_only': True, 'best_epoch': best['epoch'], 'reports': reports, 'test_used': False}
        _json(output_dir / 'screen.json', result)
    _checksum_index(output_dir)
    _checksum_index(run_dir)
    return result


def freeze_confirmation(spec_path, output_dir, *, release_root=None):
    """Freeze new reset metadata before selecting a repaired method; never record demos."""
    import numpy as np

    from dexvision.logging.level4_collection import WorkcellPilotTask, sample_level4_procedural_variation

    spec, inputs = load_repair_spec(spec_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    first = next(r for r in inputs.splits['train']['episodes'] if r['skill'] == 'reach_object' and r['stream'] == 'expert_success')
    name = f"data/demos/level4/{first['source_path']}/metadata.json"
    neutral = json.loads(ReleaseSource(inputs, release_root).read_many([name])[name])['initial_applied_action']
    records, cells = [], []
    settings = spec['confirmation']
    for family_index, cell_id in enumerate(settings['source_cells']):
        cell = {'cell_id': cell_id, 'episode_ids': [], 'reset_seeds': [], 'start_frames': []}
        for repetition in range(settings['fresh_reset_count_per_held_out_family']):
            seed = settings['fresh_reset_seed_start'] + family_index * 100 + repetition
            variation = sample_level4_procedural_variation(inputs.requirements['level4_5b_procedural_expansion'],
                skill_name='reach_object', seed=seed, boundary_case=bool(repetition % 2))
            with WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
                                  dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
                                  skill_name='reach_object', goal_condition_id=cell_id, seed=seed,
                                  procedural_variation=variation) as task:
                world = task.workcell.get_world_state()
                prior = np.asarray(neutral).copy()
                prior[:7] = [*world.robot.base_position, *world.robot.base_orientation_wxyz]
                eid = f'reach_confirmation_{seed}'
                metadata = {'episode_id': eid, 'recording_session_id': 'evaluation_only_no_recording_session',
                            'typed_goal': task.goal, 'random_seed': seed, 'initial_online_phase': 'approach',
                            'initial_applied_action': prior.tolist(), 'initial_commanded_action': prior.tolist(),
                            'task_config': task.metadata_task_config(), 'source': 'evaluation_reset_not_demonstration'}
                row = {'episode_id': eid, 'recording_session_id': metadata['recording_session_id'],
                       'typed_goal': task.goal, 'entity_id': task.goal['entity_id'], 'cell_id': cell_id,
                       'skill': 'reach_object', 'split': 'fresh_confirmation', 'episode_digest': content_digest(metadata)}
                records.append({'row': row, 'metadata': metadata})
                cell['episode_ids'].append(eid)
                cell['reset_seeds'].append(seed)
                cell['start_frames'].append(0)
        cells.append(cell)
    result = {'version': 'level5/reach-fresh-confirmation-v1', 'repair_spec_sha256': content_digest(spec),
              'input_digests': inputs.digests, 'neutral_action_source_episode': first['episode_id'],
              'disclosure': settings['disclosure'], 'records': records,
              'matrix': {'cells': cells, 'unique_reset_count': len(records),
                         'rollouts_per_track_per_training_seed': len(records) * len(settings['conditions'])}}
    _json(output_dir / 'resets.json', result)
    _checksum_index(output_dir)
    return {'frozen_unique_resets': len(records), 'path': str(output_dir / 'resets.json'), 'sha256': digest_file(output_dir / 'resets.json')}


def select_repair_method(spec_path, screen_dirs, output_dir, *, reset_dir):
    """Select a method on preregistered screening evidence, never confirmation data."""
    spec, _ = load_repair_spec(spec_path)
    output_dir = Path(output_dir)
    reset_dir = Path(reset_dir)
    _verify_run(reset_dir)
    frozen = json.loads((reset_dir / 'resets.json').read_text())
    if frozen['repair_spec_sha256'] != content_digest(spec):
        raise ValueError('Method selection requires confirmation resets frozen for this recipe')
    expected = set(spec['models'])
    evidence, seen = [], set()
    for directory in map(Path, screen_dirs):
        _verify_run(directory)
        screen = json.loads((directory / 'screen.json').read_text())
        for report in screen['reports']:
            if (report['repair_spec_sha256'] != content_digest(spec) or report['split'] != 'validation'
                    or not report['screen_only'] or report['seed'] != spec['screen']['seed']):
                raise ValueError('Invalid method selection evidence')
        arm = screen['reports'][0]['repair_arm']
        if arm in seen or {r['repair_arm'] for r in screen['reports']} != {arm}:
            raise ValueError('Duplicate or mixed repair arm')
        seen.add(arm)
        best = next(r for r in screen['reports'] if r['epoch'] == screen['best_epoch'])
        evidence.append({'arm': arm, 'best_epoch': screen['best_epoch'], 'report': best,
                         'screen_path': str(directory), 'screen_sha256': digest_file(directory / 'screen.json')})
    if seen != expected:
        raise ValueError('Method selection requires all preregistered arms')
    # Cross-arm offline losses have different active dimensions, so do not rank them.
    chosen = min(evidence, key=lambda e: (e['report']['safety_violation_count'], -e['report']['success_rate'],
        -e['report']['worst_cell_success_rate'], e['report']['normalized_terminal_error'] if e['report']['normalized_terminal_error'] is not None else float('inf'),
        e['report']['mean_action_jerk'], e['arm']))
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {'version': 'level5/reach-method-selection-v1', 'repair_spec_sha256': content_digest(spec),
              'reset_manifest_sha256': digest_file(reset_dir / 'resets.json'),
              'selected_arm': chosen['arm'], 'screen_numerical_gates_passed': chosen['report']['state_numerical_gates_passed'],
              'ranking': ['safety_ascending', 'success_descending', 'worst_cell_descending', 'error_ascending', 'jerk_ascending', 'arm_name'],
              'evidence': evidence, 'confirmation_used': False, 'test_used': False}
    _json(output_dir / 'method_selection.json', result)
    _checksum_index(output_dir)
    return result


def confirm_repair(spec_path, run_dir, reset_dir, decision_dir, output_dir):
    """One fresh confirmation per selected seed, after all three validation gates pass."""
    spec, inputs = load_repair_spec(spec_path)
    run_dir, reset_dir, decision_dir, output_dir = map(Path, (run_dir, reset_dir, decision_dir, output_dir))
    for directory in (run_dir, reset_dir, decision_dir):
        _verify_run(directory)
    decision = json.loads((decision_dir / 'method_selection.json').read_text())
    frozen = json.loads((reset_dir / 'resets.json').read_text())
    if (decision['repair_spec_sha256'] != content_digest(spec) or frozen['repair_spec_sha256'] != content_digest(spec)
            or frozen['input_digests'] != inputs.digests
            or decision['reset_manifest_sha256'] != digest_file(reset_dir / 'resets.json')):
        raise ValueError('Repair confirmation specification/input mismatch')
    arm_root = run_dir.parent.parent
    for seed in spec['seeds']:
        sibling = arm_root / f'seed_{seed}' / run_dir.name
        _verify_run(sibling)
        selected = json.loads((sibling / 'selection.json').read_text())
        report = next(r for r in selected['reports'] if r['epoch'] == selected['epoch'])
        _verify_selected_binding(sibling, selected, report, spec, inputs)
        if (report['repair_arm'] != decision['selected_arm'] or not report['state_numerical_gates_passed']
                or report['seed'] != seed or report['screen_only']):
            raise ValueError('All three selected seeds must pass full validation before confirmation')
    checkpoint_path = run_dir / 'selected.pt'
    checkpoint = load_checkpoint(checkpoint_path)
    policy = RepairPolicy(checkpoint_path, spec, inputs)
    reservation = run_dir / 'fresh_confirmation_attempt.json'
    if reservation.exists():
        raise ValueError('This selected run already reserved its single fresh confirmation')
    skill, splits = copy.deepcopy(inputs.skill), copy.deepcopy(inputs.splits)
    skill['evaluation']['test'] = frozen['matrix']
    splits['test']['episodes'] = [r['row'] for r in frozen['records']]
    confirmation_inputs = replace(inputs, skill=skill, splits=splits)
    metadata = {r['row']['episode_id']: r['metadata'] for r in frozen['records']}
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = {'version': 'level5/reach-fresh-confirmation-report-v1', 'track': 'state', 'split': 'fresh_confirmation',
                  'repair_arm': checkpoint['manifest']['repair_arm'], 'seed': checkpoint['manifest']['seed'],
                  'checkpoint_sha256': digest_file(checkpoint_path), 'repair_spec_sha256': content_digest(spec),
                  'input_digests': inputs.digests, 'reset_manifest_sha256': digest_file(reset_dir / 'resets.json'),
                  'method_selection_sha256': digest_file(decision_dir / 'method_selection.json'),
                  'disclosure': frozen['disclosure'], 'output_dir': str(output_dir.resolve()),
                  'execution_code_sha256': {p.relative_to(inputs.root).as_posix(): digest_file(p) for p in sorted((inputs.root / 'dexvision').rglob('*.py'))}}
    with reservation.open('x') as stream:
        json.dump(provenance, stream, indent=2)
    _json(output_dir / 'provenance.json', provenance)
    results = []
    for index, (row, _, condition) in enumerate(frozen_scenarios(confirmation_inputs, 'test')):
        directory = output_dir / f"{index:03d}_{row['episode_id']}_{condition}"
        result = run_reach_rollout(inputs, row, metadata[row['episode_id']], policy, condition=condition, output_dir=directory)
        result['result_path'] = f'{directory.name}/result.json'
        results.append(result)
    report = {**summarize_rollouts(results, confirmation_inputs, 'test'), **provenance, 'results': results}
    _json(output_dir / 'report.json', report)
    _checksum_index(output_dir)
    _checksum_index(run_dir)
    return report


def _verify_selected_binding(run_dir, selection, report, spec, inputs):
    """Bind selected weights to the candidate actually evaluated on full validation."""
    import torch

    checkpoint = load_checkpoint(run_dir / 'selected.pt')
    last = load_checkpoint(run_dir / 'last.pt')
    candidate = last['training_state']['candidates'][str(selection['epoch'])]
    source = load_checkpoint(run_dir / candidate['path'])
    RepairPolicy(run_dir / 'selected.pt', spec, inputs)
    if (report['checkpoint_sha256'] != digest_file(run_dir / candidate['path'])
            or report['manifest_digest'] != checkpoint['manifest']['manifest_digest']
            or report['repair_spec_sha256'] != content_digest(spec)
            or report['rollout_count'] != checkpoint['manifest']['validation_rollout_count']
            or checkpoint['schema'] != source['schema']
            or checkpoint['manifest'] != source['manifest']
            or checkpoint['normalization'] != source['normalization']
            or checkpoint['model'].keys() != source['model'].keys()
            or any(not torch.equal(checkpoint['model'][k], source['model'][k]) for k in source['model'])):
        raise ValueError('Selected repair weights or validation binding changed')
