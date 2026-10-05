"""Full validation, selection and one-time confirmation for V6 place guarded decoder."""
from __future__ import annotations

import json
from pathlib import Path

from dexvision.evaluation.place_repair import load_place_teacher_spec
from dexvision.evaluation.place_repair_policy import repair_matrix, run_repair_rollout, summarize_repair
from dexvision.learning.place_guarded_policy import GuardedPolicy, load_guarded_spec
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, select_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file


def evaluate_guarded(config, checkpoint_path, output_dir, *, split='validation'):
    spec, inputs = load_guarded_spec(config)
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    policy = GuardedPolicy(checkpoint_path, config)
    teacher = load_place_teacher_spec(spec['setup_teacher_recipe'])
    resets = repair_matrix(spec, inputs, split)
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    matrix = [{**row, 'render_condition': condition} for row in resets for condition in conditions]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': 'level5/place-guarded-evaluation-v6', 'split': split,
        'seed': checkpoint['manifest']['seed'],
        'epoch': checkpoint['training_state']['completed_epochs'],
        'checkpoint_sha256': digest_file(checkpoint_path),
        'manifest_digest': checkpoint['manifest']['manifest_digest'],
        'validation_matrix_sha256': content_digest(repair_matrix(spec, inputs, 'validation')),
        'matrix_sha256': content_digest(matrix), 'recipe_sha256': digest_file(Path(config)),
        'source_exposure': spec['confirmation']['source_exposure']}
    if split == 'confirmation':
        run = checkpoint_path.parent
        _verify_run(run)
        selected = json.loads((run / 'selection.json').read_text())
        if (checkpoint_path.name != 'selected.pt' or selected['status'] != 'selected_from_validation'
                or (run / 'confirmation_attempt.json').exists()):
            raise ValueError('Fresh place confirmation needs untested selected weights')
        _json(run / 'confirmation_attempt.json', provenance)
    _json(directory / 'provenance.json', provenance)
    results = []
    for index, row in enumerate(matrix):
        if index % 12 == 0:
            print(f'Place guarded {split} seed {provenance["seed"]} {index}/{len(matrix)}', flush=True)
        policy.bind(row)
        child = f'{index:03d}_{row["episode_id"]}_{row["render_condition"]}'
        result = run_repair_rollout(spec, inputs, teacher, teacher[2], row, policy, directory / child)
        result['result_path'] = f'{child}/result.json'
        results.append(result)
    report = {**provenance, **summarize_repair(results, inputs), 'results': results,
              'render_condition_effect': 'independent state and physics runs; no RGB claim'}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    return report


def select_guarded(config, run_dir, output_dir):
    run = Path(run_dir)
    _verify_run(run)
    checkpoint = load_checkpoint(run / 'last.pt')
    if not checkpoint['training_state']['finished'] or (run / 'selected.pt').exists():
        raise ValueError('Place guarded selection needs a complete unselected run')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    reports = []
    for epoch, item in sorted(checkpoint['training_state']['candidates'].items(),
                               key=lambda value: int(value[0])):
        print(f'Place guarded validation candidate {epoch}', flush=True)
        reports.append(evaluate_guarded(config, run / item['path'],
            directory / f'epoch_{int(epoch):04d}', split='validation'))
    _checksum_index(run)
    selected = select_checkpoint(run, reports)
    _checksum_index(directory)
    return selected
