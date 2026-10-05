"""Full validation and one-time confirmation for the V11 slot-clearance overlay."""
from __future__ import annotations

import json
from pathlib import Path

from dexvision.evaluation.place_repair import load_place_teacher_spec
from dexvision.evaluation.place_repair_policy import repair_matrix, run_repair_rollout, summarize_repair
from dexvision.learning.place_slot_clearance_policy import SlotClearancePolicy, load_slot_clearance_spec
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file


def evaluate_slot_clearance(config, checkpoint_path, output_dir, *, split='validation'):
    overlay, spec, inputs = load_slot_clearance_spec(config)
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    policy = SlotClearancePolicy(checkpoint_path, config)
    teacher = load_place_teacher_spec(spec['setup_teacher_recipe'])
    resets = repair_matrix(spec, inputs, split)
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    matrix = [{**row, 'render_condition': condition} for row in resets for condition in conditions]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': 'level5/place-slot-clearance-evaluation-v11', 'split': split,
        'seed': checkpoint['manifest']['seed'],
        'epoch': checkpoint['training_state']['completed_epochs'],
        'checkpoint_sha256': digest_file(checkpoint_path),
        'manifest_digest': checkpoint['manifest']['manifest_digest'],
        'validation_matrix_sha256': content_digest(repair_matrix(spec, inputs, 'validation')),
        'matrix_sha256': content_digest(matrix), 'recipe_sha256': digest_file(Path(config)),
        'source_exposure': overlay['source_exposure'], 'parent_policy_recipe_sha256': overlay['parent_policy_recipe_sha256']}
    if split == 'confirmation':
        run = checkpoint_path.parent
        _verify_run(run)
        selected = json.loads((run / 'selection.json').read_text())
        if (checkpoint_path.name != 'selected.pt' or selected['status'] != 'selected_from_validation'
                or (run / 'confirmation_attempt.json').exists()):
            raise ValueError('Fresh place V11 confirmation needs untested selected weights')
        _json(run / 'confirmation_attempt.json', provenance)
    _json(directory / 'provenance.json', provenance)
    results = []
    for index, row in enumerate(matrix):
        if index % 12 == 0:
            print(f'Place slot clearance {split} seed {provenance["seed"]} {index}/{len(matrix)}', flush=True)
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
