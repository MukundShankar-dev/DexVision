"""Fresh all-cell selection and saved-state audit for V24 place control."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import yaml

from dexvision.evaluation.place_full_source_selection import _audit_attempt
from dexvision.evaluation.place_wrist_hold_policy import wrist_hold_matrix
from dexvision.evaluation.place_repair import load_place_teacher_spec
from dexvision.evaluation.place_repair_policy import repair_matrix, summarize_repair
from dexvision.evaluation.place_wrist_hold_rollout import run_wrist_hold_rollout
from dexvision.learning.place_wrist_hold_policy import (
    WristHoldPlacePolicy, load_wrist_hold_spec,
)
from dexvision.learning.train_skill import _checkpoint, _checksum_index, _json, _verify_run, load_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file

VERSION = 'level5/place-wrist-hold-selection-v24'


def load_selection_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V24 selection recipe checksum mismatch')
    protocol = yaml.safe_load(path.read_text())
    if protocol.get('version') != VERSION or protocol.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place V24 selection recipe')
    recipe = Path(protocol['policy_recipe'])
    if digest_file(recipe) != protocol['policy_recipe_sha256']:
        raise ValueError('Place V24 selection policy recipe changed')
    overlay, spec, inputs = load_wrist_hold_spec(recipe)
    name = 'dexvision/evaluation/place_wrist_hold_selection.py'
    if digest_file(inputs.root / name) != protocol['selection_implementation_sha256']:
        raise ValueError('Place V24 selection implementation changed')
    if (protocol['seeds'] != overlay['seeds']
            or protocol['candidate_epochs'] != overlay['candidate_epochs']
            or protocol['training_output_dir'] != overlay['training_output_dir']
            or protocol['validation_output_dir'] != overlay['validation_output_dir']
            or protocol['output_dir'] != overlay['selection_output_dir']
            or protocol['selection_screen'] != {
                'seed_start': overlay['selection_seed_start'],
                'resets_per_cell': overlay['selection_resets_per_cell'],
                'render_condition': 'nominal'}
            or protocol['confirmation'] != {
                'seed_start': overlay['confirmation_seed_start'],
                'resets_per_cell': overlay['confirmation_resets_per_cell']}):
        raise ValueError('Place V24 selection changed frozen policy boundaries')
    return protocol, overlay, spec, inputs


def selection_matrix(protocol, overlay, spec, inputs):
    amended = copy.deepcopy(spec)
    amended['confirmation']['seed_start'] = overlay['selection_seed_start']
    rows = [{**row, 'render_condition': 'nominal'}
            for row in repair_matrix(amended, inputs, 'confirmation')]
    confirm = wrist_hold_matrix(overlay, spec, inputs, 'confirmation')
    validation = wrist_hold_matrix(overlay, spec, inputs, 'validation')
    if (len(rows) != 60 or len({row['cell_id'] for row in rows}) != 30
            or {row['seed'] for row in rows} & {row['seed'] for row in confirm + validation}):
        raise ValueError('Place V24 selection matrix incomplete or overlapping')
    return rows


def _validation_report(protocol, overlay, spec, inputs, seed):
    epoch = overlay['candidate_epochs'][str(seed)]
    run = Path(protocol['training_output_dir']) / f'seed_{seed}'
    _verify_run(run)
    checkpoint = load_checkpoint(run / 'last.pt')
    candidate = checkpoint['training_state']['candidates'].get(str(epoch))
    if (not checkpoint['training_state']['finished'] or candidate is None
            or checkpoint['manifest']['seed'] != seed):
        raise ValueError('Place V24 needs its completed V19 candidate weights')
    path = run / candidate['path']
    directory = Path(protocol['validation_output_dir']) / f'validation_seed{seed}' / f'epoch_{epoch:04d}'
    _verify_run(directory)
    report = json.loads((directory / 'report.json').read_text())
    expected = content_digest(wrist_hold_matrix(overlay, spec, inputs, 'validation'))
    if (report['version'] != 'level5/place-wrist-hold-evaluation-v24'
            or report['split'] != 'validation' or report['seed'] != seed
            or report['epoch'] != epoch
            or report['checkpoint_sha256'] != candidate['checkpoint_sha256']
            or digest_file(path) != candidate['checkpoint_sha256']
            or report['manifest_digest'] != checkpoint['manifest']['manifest_digest']
            or report['recipe_sha256'] != protocol['policy_recipe_sha256']
            or report['validation_matrix_sha256'] != expected
            or report['rollout_count'] != 240 or report['unique_reset_count'] != 60):
        raise ValueError('Place V24 validation provenance mismatch')
    if not report['state_numerical_gates_passed']:
        raise ValueError('Place V24 candidate failed full validation')
    return path, directory, report


def screen_candidate(config, seed):
    """One fresh all-cell matrix; independently audit every saved attempt."""
    protocol, overlay, spec, inputs = load_selection_spec(config)
    if seed not in protocol['seeds']:
        raise ValueError('Unknown place V24 selection seed')
    candidate, validation_dir, _ = _validation_report(protocol, overlay, spec, inputs, seed)
    matrix = selection_matrix(protocol, overlay, spec, inputs)
    epoch = overlay['candidate_epochs'][str(seed)]
    directory = Path(protocol['output_dir']) / 'screens' / f'seed_{seed}' / f'epoch_{epoch:04d}'
    directory.mkdir(parents=True, exist_ok=False)
    policy = WristHoldPlacePolicy(candidate, protocol['policy_recipe'])
    teacher = load_place_teacher_spec(spec['setup_teacher_recipe'])
    results, audits = [], []
    for index, row in enumerate(matrix):
        if index % 10 == 0:
            print(f'Place V24 selection seed {seed} {index}/{len(matrix)}', flush=True)
        policy.bind(row)
        child = directory / f'{index:03d}_{row["episode_id"]}'
        result = run_wrist_hold_rollout(spec, inputs, teacher, teacher[2], row, policy, child)
        result['result_path'] = f'{child.name}/result.json'
        results.append(result)
        audits.append(_audit_attempt(spec, inputs, row, result, child))
    report = {'version': VERSION, 'seed': seed, 'epoch': epoch,
        'selection_recipe_sha256': digest_file(Path(config)),
        'policy_recipe_sha256': protocol['policy_recipe_sha256'],
        'checkpoint_sha256': digest_file(candidate),
        'validation_report_sha256': digest_file(validation_dir / 'report.json'),
        'matrix_sha256': content_digest(matrix),
        'source_exposure': protocol['source_exposure'],
        'physical_audit_passed': all(item['passed'] for item in audits),
        'physical_audit_failures': sum(not item['passed'] for item in audits),
        'physical_audits': audits,
        **summarize_repair(results, inputs), 'results': results}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    return {key: value for key, value in report.items()
            if key not in {'results', 'physical_audits'}}


def select_screened(config, seed):
    """Package only the exact validation and physical-screen passing weights."""
    protocol, overlay, spec, inputs = load_selection_spec(config)
    if seed not in protocol['seeds']:
        raise ValueError('Unknown place V24 selection seed')
    epoch = overlay['candidate_epochs'][str(seed)]
    candidate, validation_dir, validation = _validation_report(
        protocol, overlay, spec, inputs, seed)
    matrix = selection_matrix(protocol, overlay, spec, inputs)
    screen_dir = Path(protocol['output_dir']) / 'screens' / f'seed_{seed}' / f'epoch_{epoch:04d}'
    _verify_run(screen_dir)
    screen = json.loads((screen_dir / 'report.json').read_text())
    expected = {(row['episode_id'], row['render_condition']) for row in matrix}
    if (screen['version'] != VERSION or screen['seed'] != seed or screen['epoch'] != epoch
            or screen['rollout_count'] != 60 or screen['unique_reset_count'] != 60
            or screen['selection_recipe_sha256'] != digest_file(Path(config))
            or screen['policy_recipe_sha256'] != protocol['policy_recipe_sha256']
            or screen['checkpoint_sha256'] != digest_file(candidate)
            or screen['validation_report_sha256'] != digest_file(validation_dir / 'report.json')
            or screen['matrix_sha256'] != content_digest(matrix)
            or {(row['episode_id'], row['render_condition']) for row in screen['results']} != expected
            or len(screen['results']) != len(expected)
            or {(row['episode_id'], row['render_condition']) for row in screen['physical_audits']} != expected
            or len(screen['physical_audits']) != len(expected)
            or screen['physical_audit_passed'] != all(
                row['passed'] for row in screen['physical_audits'])
            or screen['state_numerical_gates_passed'] != all(screen['gates'].values())):
        raise ValueError('Place V24 selection evidence mismatch')
    if not screen['state_numerical_gates_passed'] or not screen['physical_audit_passed']:
        raise ValueError('Place V24 all-target or physical selection gate failed')
    bundle = Path(protocol['output_dir']) / f'seed_{seed}'
    bundle.mkdir(parents=True, exist_ok=False)
    copied = bundle / 'validation_candidates' / f'epoch_{epoch:04d}.pt'
    copied.parent.mkdir(parents=True)
    shutil.copy2(candidate, copied)
    shutil.copy2(candidate.with_suffix('.pt.sha256'), copied.with_suffix('.pt.sha256'))
    _checkpoint(bundle / 'selected.pt', load_checkpoint(candidate))
    evidence = {'status': 'selected_from_validation', 'version': VERSION, 'epoch': epoch,
        'source_checkpoint_sha256': digest_file(candidate),
        'selection_recipe_sha256': digest_file(Path(config)),
        'validation_report_sha256': digest_file(validation_dir / 'report.json'),
        'all_target_report_sha256': digest_file(screen_dir / 'report.json'),
        'validation_gates_passed': validation['state_numerical_gates_passed'],
        'all_target_gates_passed': screen['state_numerical_gates_passed'],
        'all_target_physical_audit_passed': screen['physical_audit_passed'],
        'eligible_epochs': [epoch], 'passing_all_target_epochs': [epoch],
        'ranking': protocol['ranking'], 'test_used': False, 'qualification_claim': None}
    _json(bundle / 'selection.json', evidence)
    _checksum_index(bundle)
    return evidence
