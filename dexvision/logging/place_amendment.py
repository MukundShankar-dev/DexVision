"""Append-only ordinary-physics place collection and immutable release."""
from __future__ import annotations

from collections import defaultdict
import json
import tarfile
from pathlib import Path

import yaml

from dexvision.evaluation.place_teacher_search import load_place_search_spec, run_place_search_attempt
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run
from dexvision.learning.training_manifest import content_digest, digest_file
from dexvision.logging.level4_collection import sample_level4_procedural_variation
from dexvision.logging.place_source_capture import capture_causal_place_trace


def load_place_data_plan(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place data plan checksum mismatch')
    plan = yaml.safe_load(path.read_text())
    if plan.get('version') not in {'level5/place-unassisted-data-v1',
                                   'level5/place-unassisted-data-v2',
                                   'level5/place-unassisted-data-v3'} \
            or plan.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place data plan')
    search_path = Path(plan['teacher_search_recipe'])
    if digest_file(search_path) != plan['teacher_search_sha256']:
        raise ValueError('Place search changed after data plan freeze')
    search, base, place, pick = load_place_search_spec(search_path)
    report_path = Path(plan['teacher_feasibility_report'])
    if digest_file(report_path) != plan['teacher_feasibility_sha256']:
        raise ValueError('Place teacher feasibility evidence changed')
    report = json.loads(report_path.read_text())
    if (not report['teacher_feasibility_passed'] or report['version'] != search['version']
            or report['source_recipe_sha256'] != plan['teacher_search_sha256']):
        raise ValueError('Place data requires a passing locked all-cell source screen')
    _, inputs = place
    config_path = inputs.root / inputs.plan['inputs']['dataset_config']['path']
    dataset = yaml.safe_load(config_path.read_text())
    cells = {cell['id'] for cell in dataset['coverage_cells']
             if cell.get('data_group') == 'pick_place'}
    train, validation = (plan['splits'][split]['cells'] for split in ('train', 'validation'))
    if (not set(train + validation) <= cells
            or any(len(values) != len(set(values)) for values in (train, validation))):
        raise ValueError('Place data cells overlap or differ from declared workcell coverage')
    if plan['version'] == 'level5/place-unassisted-data-v3':
        if set(train) != cells or set(validation) != cells:
            raise ValueError('V3 place data requires all workcell cells in both splits')
    elif set(train) & set(validation):
        raise ValueError('Place data cells overlap or differ from declared workcell coverage')
    if plan['publication_success_min'] < search['gates']['success_min'] \
            or plan['publication_worst_family_success_min'] < search['gates']['worst_family_success_min']:
        raise ValueError('Place data publication gates weaken source feasibility')
    return plan, search, base, place, pick


def freeze_place_data_plan(config, output_dir):
    plan, _, _, place, _ = load_place_data_plan(config)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    assignments = {}
    for split, settings in plan['splits'].items():
        assignments[split] = []
        for i, cell in enumerate(settings['cells']):
            for j in range(settings['episodes_per_cell']):
                seed = settings['seed_start'] + i * 100 + j
                assignments[split].append({'cell': cell, 'seed': seed, 'boundary': bool(j % 2),
                    'episode_id': f"{plan['episode_namespace']}_{split}_{seed}",
                    'session_id': f"{plan['episode_namespace']}_{split}_{cell}"})
    seeds = [item['seed'] for values in assignments.values() for item in values]
    episodes = [item['episode_id'] for values in assignments.values() for item in values]
    if len(seeds) != len(set(seeds)) or len(episodes) != len(set(episodes)):
        raise ValueError('Place data plan has duplicate seeds or episodes')
    _json(directory / 'plan.json', {'plan': plan, 'plan_digest': content_digest(plan),
        'input_digests': place[1].digests, 'assignments': assignments})
    return {split: len(values) for split, values in assignments.items()}


def _frozen(config, output_dir):
    plan, search, base, place, pick = load_place_data_plan(config)
    frozen = json.loads((Path(output_dir) / 'plan.json').read_text())
    if frozen['plan_digest'] != content_digest(plan) or frozen['input_digests'] != place[1].digests:
        raise ValueError('Place data/source bindings changed after freezing assignments')
    return plan, search, base, place, pick, frozen


def collect_place_data_split(config, output_dir, split):
    plan, search, base, place, pick, frozen = _frozen(config, output_dir)
    if split not in {'train', 'validation'}:
        raise ValueError('Collect only the frozen place train or validation split')
    _, inputs = place
    config_path = inputs.root / inputs.plan['inputs']['dataset_config']['path']
    cells = {cell['id']: cell for cell in yaml.safe_load(config_path.read_text())['coverage_cells']
             if cell.get('data_group') == 'pick_place'}
    directory = Path(output_dir) / split
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'report.json').exists():
        _verify_run(directory)
        existing = json.loads((directory / 'report.json').read_text())
        return {'split': split, 'attempts': existing['attempts'], 'successes': existing['successes'],
                'resumed_completed_report': True}
    results = []
    for item in frozen['assignments'][split]:
        cell = cells[item['cell']]
        variation = sample_level4_procedural_variation(
            inputs.requirements['level4_5b_procedural_expansion'],
            skill_name='pick_place_sequence', seed=item['seed'], boundary_case=item['boundary'])
        goal = {'object_id': cell['object_id'], 'target_id': cell['target_id']}
        row = {'episode_id': item['episode_id'], 'cell_id': item['cell'],
               'entity_id': cell['object_id'], 'typed_goal': goal,
               'episode_digest': content_digest({'cell': item['cell'], 'seed': item['seed'],
                                                 'variation': variation})}
        metadata = {'random_seed': item['seed'], 'task_config': {'procedural_variation': variation}}
        child = directory / item['episode_id']
        source = child / 'source'
        if source.exists():
            _verify_run(source)
            result = json.loads((source / 'result.json').read_text())
        else:
            result = run_place_search_attempt(search, base, place, pick, row, metadata, source)
        if (result['episode_id'], result['reset_seed']) != (item['episode_id'], item['seed']):
            raise ValueError('Place collection source ownership changed')
        payload = child / 'payload'
        if result['terminal_result'] == 'succeeded' and not any(result['safety_counters'].values()):
            if payload.exists():
                _verify_run(payload)
            else:
                capture_causal_place_trace(place, row, metadata, source / 'live', payload,
                    session_id=item['session_id'])
        elif payload.exists():
            raise ValueError('Failed place attempt cannot have a training payload')
        results.append({**item, 'object_id': cell['object_id'], 'target_id': cell['target_id'],
            'object_family': result['object_family'], 'terminal_result': result['terminal_result'],
            'terminal_reason': result['terminal_reason'], 'safety_counters': result['safety_counters'],
            'selected_candidate_index': result['selected_candidate_index'],
            'source_result_sha256': digest_file(source / 'result.json'),
            'payload': payload.exists()})
        print(f"Place {split} {len(results)}/{len(frozen['assignments'][split])}: "
              f"{item['episode_id']} {result['terminal_reason']}", flush=True)
    report = {'plan_digest': content_digest(plan), 'split': split, 'attempts': len(results),
        'successes': sum(r['terminal_result'] == 'succeeded' for r in results), 'results': results,
        'failed_attempts_retained': True}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    return {'split': split, 'attempts': report['attempts'], 'successes': report['successes']}


def publish_place_data(config, output_dir, release_dir):
    """Publish only canonical successful payloads; retain every failed attempt in reports."""
    plan, _, _, place, _, frozen = _frozen(config, output_dir)
    destination = Path(release_dir)
    destination.mkdir(parents=True, exist_ok=True)
    name = plan['release_name']
    archive = destination / f'{name}.tar.gz'
    manifest_path = destination / f'{name}_manifest.json'
    checksum_path = destination / f'{name}.tar.gz.sha256'
    if any(path.exists() for path in (archive, manifest_path, checksum_path)):
        raise FileExistsError('Immutable place release already exists; create a new version')
    reports, payload, source_search_results = {}, {}, {}
    for split in ('train', 'validation'):
        directory = Path(output_dir) / split
        _verify_run(directory)
        report = json.loads((directory / 'report.json').read_text())
        expected = {item['episode_id']: item for item in frozen['assignments'][split]}
        if (report['plan_digest'] != content_digest(plan) or report['split'] != split
                or report['attempts'] != len(expected) or len(report['results']) != len(expected)
                or {item['episode_id'] for item in report['results']} != set(expected)):
            raise ValueError('Place collection has missing, duplicate or extra attempts')
        family = defaultdict(list)
        for item in report['results']:
            owner = expected[item['episode_id']]
            if any(item[key] != owner[key] for key in ('cell', 'seed', 'boundary', 'session_id')):
                raise ValueError('Place collection changed preregistered ownership')
            search_path = directory / item['episode_id'] / 'source' / 'result.json'
            if digest_file(search_path) != item['source_result_sha256']:
                raise ValueError('Place search outcome changed after collection')
            search_result = json.loads(search_path.read_text())
            if (search_result['episode_id'] != item['episode_id']
                    or search_result['terminal_result'] != item['terminal_result']
                    or search_result['safety_counters'] != item['safety_counters']):
                raise ValueError('Place search summary differs from frozen attempt')
            source_search_results[item['episode_id']] = search_result
            family[item['object_family']].append(item)
            if any(item['safety_counters'].values()):
                raise ValueError('Place collection has a live safety violation')
            if item['payload']:
                if item['terminal_result'] != 'succeeded':
                    raise ValueError('Failed place attempt entered source payload')
                for filename in ('metadata.json', 'result.json', 'trajectory.npz'):
                    source = directory / item['episode_id'] / 'payload' / filename
                    member = f'datasets/{name}/episodes/{item["episode_id"]}/{filename}'
                    payload[member] = {'source': source, 'sha256': digest_file(source),
                                       'bytes': source.stat().st_size}
        if (report['successes'] / report['attempts'] < plan['publication_success_min']
                or set(family) != {'cuboid', 'cylinder', 'flat_puck'}
                or any(sum(item['payload'] for item in rows) / len(rows)
                       < plan['publication_worst_family_success_min'] for rows in family.values())):
            raise ValueError(f'Place {split} source fails publication success/family gate')
        reports[split] = report
    audit_dir = Path(output_dir) / 'audit'
    _verify_run(audit_dir)
    audit = json.loads((audit_dir / 'report.json').read_text())
    expected_successes = {(split, item['episode_id']) for split, report in reports.items()
                          for item in report['results'] if item['payload']}
    audited = {(item['split'], item['episode_id']) for item in audit['audits']}
    excluded = {(item['split'], item['episode_id']) for item in audit['excluded_failures']}
    expected_failures = {(split, item['episode_id']) for split, report in reports.items()
                         for item in report['results'] if not item['payload']}
    if (not audit['passed'] or audit['plan_digest'] != content_digest(plan)
            or audited != expected_successes or len(audit['audits']) != len(expected_successes)
            or excluded != expected_failures or len(audit['excluded_failures']) != len(expected_failures)):
        raise ValueError('Place data requires complete passing saved-state source audit')
    for item in audit['audits']:
        payload_path = (Path(output_dir) / item['split'] / item['episode_id'] / 'payload' / 'trajectory.npz')
        if not item['passed'] or digest_file(payload_path) != item['payload_sha256']:
            raise ValueError('Place physical audit payload changed before release')
    with archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for member, item in sorted(payload.items()):
            tar.add(item['source'], arcname=member, recursive=False)
    manifest = {'version': plan['version'], 'archive': archive.as_posix(),
        'archive_sha256': digest_file(archive), 'plan': plan, 'plan_digest': content_digest(plan),
        'input_digests': place[1].digests, 'reports': reports,
        'files': {key: {name: value for name, value in item.items() if name != 'source'}
                  for key, item in payload.items()},
        'physical_audit_sha256': digest_file(audit_dir / 'report.json'),
        'physical_audit': audit,
        'source_search_results': source_search_results,
        'preflight_artifacts': 'Full candidate trajectories remain in ignored local collection staging; '
            'candidate outcomes and selected-live receipts are embedded in this manifest.',
        'session_provenance': plan['session_rule'], 'disclosure': plan['disclosure'],
        'training_stream': plan['training_stream'], 'source_alignment': plan['source_alignment']}
    _json(manifest_path, manifest)
    checksum_path.write_text(f'{manifest["archive_sha256"]}  {archive.as_posix()}\n')
    return {'archive': str(archive), 'manifest': str(manifest_path),
            'episodes': {split: {'attempts': report['attempts'], 'successes': report['successes']}
                         for split, report in reports.items()}}
