"""Full validation and fresh confirmation for the V19 full-source decoder."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from dexvision.evaluation.place_repair import load_place_teacher_spec
from dexvision.evaluation.place_repair_policy import repair_matrix, run_repair_rollout, summarize_repair
from dexvision.learning.place_full_source_policy import (
    FullSourcePlacePolicy, full_source_validation_matrix, load_full_source_spec,
)
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file


def full_source_matrix(overlay, spec, inputs, split):
    amended = copy.deepcopy(spec)
    if split == 'confirmation':
        amended['confirmation']['seed_start'] = overlay['confirmation_seed_start']
        return repair_matrix(amended, inputs, split)
    if split == 'validation':
        return full_source_validation_matrix(spec, inputs)
    else:
        raise ValueError('Place V19 evaluation requires validation or confirmation')


def evaluate_full_source(config, checkpoint_path, output_dir, *, split='validation'):
    spec, inputs = load_full_source_spec(config)
    overlay = spec
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    policy = FullSourcePlacePolicy(checkpoint_path, config)
    teacher = load_place_teacher_spec(spec['setup_teacher_recipe'])
    resets = full_source_matrix(overlay, spec, inputs, split)
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    matrix = [{**row, 'render_condition': condition} for row in resets for condition in conditions]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': 'level5/place-full-source-evaluation-v19', 'split': split,
        'seed': checkpoint['manifest']['seed'],
        'epoch': checkpoint['training_state']['completed_epochs'],
        'checkpoint_sha256': digest_file(checkpoint_path),
        'manifest_digest': checkpoint['manifest']['manifest_digest'],
        'validation_matrix_sha256': content_digest(full_source_matrix(overlay, spec, inputs, 'validation')),
        'matrix_sha256': content_digest(matrix), 'recipe_sha256': digest_file(Path(config)),
        'source_exposure': spec['confirmation']['source_exposure'],
        'parent_policy_recipe_sha256': overlay['parent_policy_recipe_sha256']}
    if split == 'confirmation':
        run = checkpoint_path.parent
        _verify_run(run)
        selected = json.loads((run / 'selection.json').read_text())
        if (checkpoint_path.name != 'selected.pt'
                or selected['status'] != 'selected_from_validation'
                or selected['version'] != 'level5/place-full-source-selection-v19'
                or not selected['all_target_gates_passed']
                or not selected['all_target_physical_audit_passed']
                or (run / 'confirmation_attempt.json').exists()):
            raise ValueError('Fresh V19 confirmation needs physically screened selected weights')
        _json(run / 'confirmation_attempt.json', provenance)
    _json(directory / 'provenance.json', provenance)
    results = []
    for index, row in enumerate(matrix):
        if index % 12 == 0:
            print(f'Place full source {split} seed {provenance["seed"]} {index}/{len(matrix)}', flush=True)
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
