"""Full validation and one-time confirmation for the V25 place policy."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from dexvision.evaluation.place_repair import load_place_teacher_spec
from dexvision.evaluation.place_repair_policy import repair_matrix, summarize_repair
from dexvision.evaluation.place_wrist_hold_rollout import run_wrist_hold_rollout
from dexvision.learning.place_pad_release_policy import (
    PadReleasePlacePolicy, load_pad_release_spec,
)
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file

VERSION = 'level5/place-pad-release-evaluation-v25'


def pad_release_matrix(overlay, spec, inputs, split):
    amended = copy.deepcopy(spec)
    if split == 'validation':
        amended['confirmation']['seed_start'] = overlay['validation_seed_start']
    elif split == 'confirmation':
        amended['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    else:
        raise ValueError('Place V25 evaluation needs validation or confirmation')
    rows = repair_matrix(amended, inputs, 'confirmation')
    if len(rows) != 60 or len({row['cell_id'] for row in rows}) != 30:
        raise ValueError('Place V25 evaluation matrix is incomplete')
    return rows


def evaluate_pad_release(config, checkpoint_path, output_dir, *, split='validation'):
    overlay, spec, inputs = load_pad_release_spec(config)
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    seed = checkpoint['manifest']['seed']
    epoch = checkpoint['training_state']['completed_epochs']
    if epoch != overlay['candidate_epochs'][str(seed)]:
        raise ValueError('Place V25 evaluation requires predeclared V19 weight epoch')
    policy = PadReleasePlacePolicy(checkpoint_path, config)
    teacher = load_place_teacher_spec(spec['setup_teacher_recipe'])
    resets = pad_release_matrix(overlay, spec, inputs, split)
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    matrix = [{**row, 'render_condition': condition} for row in resets for condition in conditions]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': VERSION, 'split': split, 'seed': seed, 'epoch': epoch,
        'checkpoint_sha256': digest_file(checkpoint_path),
        'manifest_digest': checkpoint['manifest']['manifest_digest'],
        'validation_matrix_sha256': content_digest(pad_release_matrix(
            overlay, spec, inputs, 'validation')),
        'matrix_sha256': content_digest(matrix), 'recipe_sha256': digest_file(Path(config)),
        'source_exposure': overlay['source_exposure'],
        'parent_policy_recipe_sha256': overlay['parent_policy_recipe_sha256']}
    if split == 'confirmation':
        run = checkpoint_path.parent
        _verify_run(run)
        selected = json.loads((run / 'selection.json').read_text())
        if (checkpoint_path.name != 'selected.pt'
                or selected['status'] != 'selected_from_validation'
                or selected['version'] != 'level5/place-pad-release-selection-v25'
                or not selected['all_target_gates_passed']
                or not selected['all_target_physical_audit_passed']
                or (run / 'confirmation_attempt.json').exists()):
            raise ValueError('Fresh V25 confirmation needs physically screened selected weights')
        _json(run / 'confirmation_attempt.json', provenance)
    _json(directory / 'provenance.json', provenance)
    results = []
    for index, row in enumerate(matrix):
        if index % 12 == 0:
            print(f'Place V25 {split} seed {seed} {index}/{len(matrix)}', flush=True)
        policy.bind(row)
        child = f'{index:03d}_{row["episode_id"]}_{row["render_condition"]}'
        result = run_wrist_hold_rollout(spec, inputs, teacher, teacher[2], row, policy,
            directory / child)
        result['result_path'] = f'{child}/result.json'
        results.append(result)
    report = {**provenance, **summarize_repair(results, inputs), 'results': results,
              'render_condition_effect': 'independent state and physics runs; no RGB claim'}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    return report
