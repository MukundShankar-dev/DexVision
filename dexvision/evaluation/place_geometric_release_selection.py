"""Fresh all-target and saved-state selection for the V18 geometric-release decoder."""
from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

import numpy as np
import yaml

from dexvision.evaluation.place_repair import load_place_teacher_spec
from dexvision.evaluation.place_repair_policy import repair_matrix, run_repair_rollout, summarize_repair
from dexvision.evaluation.place_repair_audit import audit_repair_rollout
from dexvision.learning.place_geometric_release_policy import GeometricReleasePlacePolicy, load_geometric_release_spec
from dexvision.learning.train_skill import _checkpoint, _checksum_index, _json, _verify_run, load_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file

VERSION = 'level5/place-geometric-release-selection-v18'


def _audit_attempt(spec, inputs, row, result, child):
    """Audit a place trace; retain a failed setup with no place actions as excluded."""
    with np.load(child / 'trajectory.npz', allow_pickle=False) as data:
        has_place_actions = 'applied_actions' in data.files
    if not has_place_actions:
        if (result['terminal_result'] != 'rejected_before_execution'
                or result['terminal_reason'] not in {
                    'pick_setup_failed', 'held_precondition_failed',
                    'place_setup_failed', 'rejected_precondition'}):
            raise ValueError('Only rejected setup can lack saved place actions')
        return {'episode_id': row['episode_id'], 'render_condition': row['render_condition'],
                'passed': not any(result['safety_counters'].values()),
                'reported_success': False, 'excluded_before_place': True,
                'trajectory_sha256': digest_file(child / 'trajectory.npz')}
    return audit_repair_rollout(spec, inputs, row, result, child)


def load_selection_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V18 selection recipe checksum mismatch')
    protocol = yaml.safe_load(path.read_text())
    if protocol['version'] != VERSION or protocol['checkpoint'] != '5.4A':
        raise ValueError('Unsupported place selection recipe')
    policy_recipe = Path(protocol['policy_recipe'])
    if digest_file(policy_recipe) != protocol['policy_recipe_sha256']:
        raise ValueError('Place selection policy recipe changed')
    overlay, spec, inputs = load_geometric_release_spec(policy_recipe)
    if digest_file(inputs.root / 'dexvision/evaluation/place_geometric_release_selection.py') != \
            protocol['selection_implementation_sha256']:
        raise ValueError('Place V18 selection implementation changed')
    if (protocol['seeds'] != spec['seeds']
            or protocol['training_output_dir'] != spec['output_dir']
            or protocol['validation_output_dir'] != overlay['validation_output_dir']
            or protocol['confirmation']['seed_start'] != overlay['confirmation_seed_start']
            or protocol['confirmation']['resets_per_cell'] != spec['confirmation']['resets_per_cell']
            or protocol['selection_screen']['resets_per_cell'] != 2
            or protocol['selection_screen']['render_condition'] != 'nominal'
            or protocol['selection_screen']['seed_start'] != 1300260900
            or protocol['candidate_epochs'] != {seed: [epoch] for seed, epoch in overlay['candidate_epochs'].items()}):
        raise ValueError('Place selection changed frozen policy or confirmation boundary')
    return protocol, spec, inputs


def selection_matrix(protocol, spec, inputs):
    amended = copy.deepcopy(spec)
    amended['confirmation']['seed_start'] = protocol['selection_screen']['seed_start']
    rows = [{**row, 'render_condition': protocol['selection_screen']['render_condition']}
            for row in repair_matrix(amended, inputs, 'confirmation')]
    confirmation = copy.deepcopy(spec)
    confirmation['confirmation']['seed_start'] = protocol['confirmation']['seed_start']
    if (len(rows) != 60 or len({row['cell_id'] for row in rows}) != 30
            or {row['seed'] for row in rows} &
            {row['seed'] for row in repair_matrix(confirmation, inputs, 'confirmation')}):
        raise ValueError('Place all-target selection matrix is incomplete or overlaps confirmation')
    return rows


def _validation_report(protocol, spec, inputs, seed, epoch):
    root = Path(protocol['training_output_dir'])
    run = root / f'seed_{seed}'
    _verify_run(run)
    checkpoint = load_checkpoint(run / 'last.pt')
    candidate = checkpoint['training_state']['candidates'].get(str(epoch))
    if (not checkpoint['training_state']['finished'] or candidate is None
            or checkpoint['manifest']['seed'] != seed):
        raise ValueError('Place selection needs a completed candidate training run')
    path = run / candidate['path']
    report_dir = Path(protocol['validation_output_dir']) / f'validation_seed{seed}' / f'epoch_{epoch:04d}'
    _verify_run(report_dir)
    report = json.loads((report_dir / 'report.json').read_text())
    expected_matrix = content_digest(repair_matrix(spec, inputs, 'validation'))
    if (report['split'] != 'validation' or report['seed'] != seed or report['epoch'] != epoch
            or report['checkpoint_sha256'] != candidate['checkpoint_sha256']
            or digest_file(path) != candidate['checkpoint_sha256']
            or report['manifest_digest'] != checkpoint['manifest']['manifest_digest']
            or report['recipe_sha256'] != protocol['policy_recipe_sha256']
            or report['validation_matrix_sha256'] != expected_matrix
            or report['rollout_count'] != 192 or report['unique_reset_count'] != 48):
        raise ValueError('Place selection validation report provenance mismatch')
    return path, report_dir, report


def eligible_candidates(config, seed):
    protocol, spec, inputs = load_selection_spec(config)
    if seed not in protocol['seeds']:
        raise ValueError('Unknown place selection seed')
    result = []
    for text_epoch in protocol['candidate_epochs'][str(seed)]:
        epoch = int(text_epoch)
        _, _, report = _validation_report(protocol, spec, inputs, seed, epoch)
        if report['state_numerical_gates_passed']:
            result.append(epoch)
    if not result:
        raise ValueError('No place candidate passes the full validation gates')
    return result


def screen_candidate(config, seed, epoch):
    """Run a fresh all-target matrix and audit every saved physical attempt."""
    protocol, spec, inputs = load_selection_spec(config)
    if epoch not in eligible_candidates(config, seed):
        raise ValueError('Place all-target screen requires a validation-qualified candidate')
    candidate, validation_dir, _ = _validation_report(protocol, spec, inputs, seed, epoch)
    matrix = selection_matrix(protocol, spec, inputs)
    root = Path(protocol['output_dir'])
    directory = root / 'screens' / f'seed_{seed}' / f'epoch_{epoch:04d}'
    directory.mkdir(parents=True, exist_ok=False)
    policy = GeometricReleasePlacePolicy(candidate, protocol['policy_recipe'])
    teacher = load_place_teacher_spec(spec['setup_teacher_recipe'])
    results, audits = [], []
    for index, row in enumerate(matrix):
        if index % 10 == 0:
            print(f'Place all-target selection seed {seed} epoch {epoch} {index}/{len(matrix)}', flush=True)
        policy.bind(row)
        child = directory / f'{index:03d}_{row["episode_id"]}'
        result = run_repair_rollout(spec, inputs, teacher, teacher[2], row, policy, child)
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
    """Select only among candidates that pass validation and all-target gates."""
    protocol, spec, inputs = load_selection_spec(config)
    epochs = eligible_candidates(config, seed)
    matrix = selection_matrix(protocol, spec, inputs)
    matrix_sha = content_digest(matrix)
    expected = {(row['episode_id'], row['render_condition']) for row in matrix}
    root = Path(protocol['output_dir'])
    bundle = root / f'seed_{seed}'
    if bundle.exists():
        raise ValueError('Place selection bundle already exists')
    candidates = []
    for epoch in epochs:
        path, validation_dir, validation = _validation_report(protocol, spec, inputs, seed, epoch)
        screen_dir = root / 'screens' / f'seed_{seed}' / f'epoch_{epoch:04d}'
        _verify_run(screen_dir)
        screen = json.loads((screen_dir / 'report.json').read_text())
        if (screen['seed'] != seed or screen['epoch'] != epoch or screen['rollout_count'] != 60
                or screen['unique_reset_count'] != 60
                or screen['selection_recipe_sha256'] != digest_file(Path(config))
                or screen['policy_recipe_sha256'] != protocol['policy_recipe_sha256']
                or screen['checkpoint_sha256'] != digest_file(path)
                or screen['validation_report_sha256'] != digest_file(validation_dir / 'report.json')
                or screen['matrix_sha256'] != matrix_sha
                or {(row['episode_id'], row['render_condition']) for row in screen['results']} != expected
                or len(screen['results']) != len(expected)
                or {(row['episode_id'], row['render_condition'])
                    for row in screen['physical_audits']} != expected
                or len(screen['physical_audits']) != len(expected)
                or screen['physical_audit_passed'] != all(
                    row['passed'] for row in screen['physical_audits'])
                or screen['state_numerical_gates_passed'] != all(screen['gates'].values())):
            raise ValueError('Place all-target selection evidence mismatch')
        candidates.append((epoch, path, validation_dir, validation, screen_dir, screen))
    passing = [item for item in candidates if item[5]['state_numerical_gates_passed']
               and item[5]['physical_audit_passed']]
    if not passing:
        raise ValueError('No place candidate passes validation, all-target and physical gates')

    def rank(item):
        epoch, _, _, validation, _, screen = item
        error = validation['normalized_terminal_error']
        return (screen['safety_violation_count'], -screen['success_rate'],
            -screen['worst_family_success_rate'], -screen['settled_among_nominally_placed'],
            -validation['success_rate'], error if error is not None else math.inf, epoch)

    epoch, candidate, validation_dir, validation, screen_dir, screen = min(passing, key=rank)
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
        'eligible_epochs': epochs, 'passing_all_target_epochs': [item[0] for item in passing],
        'ranking': protocol['ranking'], 'test_used': False, 'qualification_claim': None}
    _json(bundle / 'selection.json', evidence)
    _checksum_index(bundle)
    return evidence
