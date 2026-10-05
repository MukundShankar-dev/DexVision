"""Full validation and fresh confirmation for the V18 geometric-release decoder."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from dexvision.evaluation.place_repair import load_place_teacher_spec
from dexvision.evaluation.place_repair_policy import repair_matrix, run_repair_rollout, summarize_repair
from dexvision.learning.place_geometric_release_policy import GeometricReleasePlacePolicy, load_geometric_release_spec
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file


def geometric_release_matrix(overlay, spec, inputs, split):
    amended = copy.deepcopy(spec)
    if split == 'confirmation':
        amended['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif split != 'validation':
        raise ValueError('Place V18 evaluation requires validation or confirmation')
    return repair_matrix(amended, inputs, split)


def evaluate_geometric_release(config, checkpoint_path, output_dir, *, split='validation'):
    overlay, spec, inputs = load_geometric_release_spec(config)
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    policy = GeometricReleasePlacePolicy(checkpoint_path, config)
    teacher = load_place_teacher_spec(spec['setup_teacher_recipe'])
    resets = geometric_release_matrix(overlay, spec, inputs, split)
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    matrix = [{**row, 'render_condition': condition} for row in resets for condition in conditions]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': 'level5/place-geometric-release-evaluation-v18', 'split': split,
        'seed': checkpoint['manifest']['seed'],
        'epoch': checkpoint['training_state']['completed_epochs'],
        'checkpoint_sha256': digest_file(checkpoint_path),
        'manifest_digest': checkpoint['manifest']['manifest_digest'],
        'validation_matrix_sha256': content_digest(geometric_release_matrix(overlay, spec, inputs, 'validation')),
        'matrix_sha256': content_digest(matrix), 'recipe_sha256': digest_file(Path(config)),
        'source_exposure': overlay['source_exposure'],
        'parent_policy_recipe_sha256': overlay['parent_policy_recipe_sha256']}
    if split == 'confirmation':
        run = checkpoint_path.parent
        _verify_run(run)
        selected = json.loads((run / 'selection.json').read_text())
        if (checkpoint_path.name != 'selected.pt'
                or selected['status'] != 'selected_from_validation'
                or selected['version'] != 'level5/place-geometric-release-selection-v18'
                or not selected['all_target_gates_passed']
                or not selected['all_target_physical_audit_passed']
                or (run / 'confirmation_attempt.json').exists()):
            raise ValueError('Fresh V18 confirmation needs physically screened selected weights')
        _json(run / 'confirmation_attempt.json', provenance)
    _json(directory / 'provenance.json', provenance)
    results = []
    for index, row in enumerate(matrix):
        if index % 12 == 0:
            print(f'Place geometric release {split} seed {provenance["seed"]} {index}/{len(matrix)}', flush=True)
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
