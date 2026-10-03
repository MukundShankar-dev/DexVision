"""Append-only, checksum-bound collection for the approved unassisted pick amendment."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import yaml

from dexvision.evaluation.pick_teacher import load_teacher_spec, teacher_attempt
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run
from dexvision.learning.training_manifest import content_digest, digest_file


def load_data_plan(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Pick data plan checksum mismatch')
    plan = yaml.safe_load(path.read_text())
    if plan['version'] not in {'level5/pick-unassisted-data-v1', 'level5/pick-unassisted-data-v2'} or plan['checkpoint'] != '5.3A':
        raise ValueError('Unknown pick data amendment')
    teacher, spec, inputs = load_teacher_spec(plan['teacher_recipe'])
    if digest_file(Path(plan['teacher_recipe'])) != plan['teacher_sha256']:
        raise ValueError('Teacher recipe changed after data preregistration')
    report_path = Path(plan['teacher_feasibility_report'])
    if digest_file(report_path) != plan['teacher_feasibility_sha256']:
        raise ValueError('Teacher feasibility report changed')
    report = json.loads(report_path.read_text())
    if not report['passed'] or not all(r['success'] for r in report['results']):
        raise ValueError('Qualify the unassisted teacher before collecting data')
    if teacher.get('orientation_reference') != 'skill_entry' or not teacher.get('record_causal_features'):
        raise ValueError('Dataset requires aligned causal entry context and feature recording')
    return plan, teacher, spec, inputs


def freeze_data_plan(config, output_dir):
    plan, _, _, inputs = load_data_plan(config)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    assignments = {}
    namespace = plan.get('episode_namespace', 'pick_unassisted_v1')
    for split, settings in plan['splits'].items():
        assignments[split] = []
        for i, cell in enumerate(settings['cells']):
            for j in range(settings['episodes_per_cell']):
                seed = settings['seed_start'] + i * 100 + j
                assignments[split].append({'cell': cell, 'seed': seed, 'boundary': bool(j % 2),
                    'episode_id': f'{namespace}_{split}_{seed}',
                    'session_id': f'{namespace}_{split}_{cell}'})
    seeds = [r['seed'] for rows in assignments.values() for r in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Data split seeds overlap')
    _json(directory / 'plan.json', {'plan': plan, 'plan_digest': content_digest(plan),
        'input_digests': inputs.digests, 'assignments': assignments})
    return {s: len(v) for s, v in assignments.items()}


def _frozen(config, directory):
    plan, teacher, spec, inputs = load_data_plan(config)
    frozen = json.loads((Path(directory) / 'plan.json').read_text())
    if frozen['plan_digest'] != content_digest(plan) or frozen['input_digests'] != inputs.digests:
        raise ValueError('Collection plan/source bindings changed')
    return plan, teacher, spec, inputs, frozen


def collect_data_split(config, output_dir, split):
    plan, teacher, spec, inputs, frozen = _frozen(config, output_dir)
    if split not in frozen['assignments']:
        raise ValueError('Collect only the frozen training or validation split')
    directory = Path(output_dir) / split
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for item in frozen['assignments'][split]:
        result = teacher_attempt(teacher, spec, inputs, item['cell'], item['seed'], item['boundary'],
            directory / item['episode_id'])
        results.append({**item, **result})
        print(f"Collected {split} {len(results)}/{len(frozen['assignments'][split])}: {result['reason']}", flush=True)
    _json(directory / 'report.json', {'plan_digest': content_digest(plan), 'split': split,
        'successes': sum(r['success'] for r in results), 'attempts': len(results), 'results': results})
    _checksum_index(directory)
    return {'split': split, 'successes': sum(r['success'] for r in results), 'attempts': len(results)}


def publish_data(config, output_dir, release_dir):
    """Archive only canonical episode payloads, not the staging directory or scratch runs."""
    plan, _, _, inputs, frozen = _frozen(config, output_dir)
    destination = Path(release_dir)
    destination.mkdir(parents=True, exist_ok=True)
    name = plan['release_name']
    archive = destination / f'{name}.tar.gz'
    manifest_path = destination / f'{name}_manifest.json'
    checksum_path = destination / f'{name}.tar.gz.sha256'
    if any(p.exists() for p in (archive, manifest_path, checksum_path)):
        raise FileExistsError('Immutable pick release already exists; create a new version')
    reports, payload = {}, {}
    for split in frozen['assignments']:
        directory = Path(output_dir) / split
        _verify_run(directory)
        report = json.loads((directory / 'report.json').read_text())
        expected = {r['episode_id']: r for r in frozen['assignments'][split]}
        if (report['plan_digest'] != content_digest(plan) or len(report['results']) != len(expected)
                or {r['episode_id'] for r in report['results']} != set(expected)):
            raise ValueError('Collection has missing, duplicate or extra attempts')
        if (report['successes'] / report['attempts'] < plan['publication_success_min']
                or any(any(r['safety_counters'].values()) for r in report['results'])):
            raise ValueError('Collection fails publication success/safety requirements')
        reports[split] = report
        for row in report['results']:
            if any(row[k] != expected[row['episode_id']][k] for k in ('cell', 'seed', 'boundary', 'session_id')):
                raise ValueError('Collection changed preregistered ownership')
            for filename in ('metadata.json', 'result.json', 'trajectory.npz'):
                source = directory / row['episode_id'] / filename
                member = f'datasets/{name}/episodes/{row["episode_id"]}/{filename}'
                payload[member] = {'source': source, 'sha256': digest_file(source), 'bytes': source.stat().st_size}
    # Exclusive archive creation and explicit canonical member names preserve old releases.
    with archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for member, item in sorted(payload.items()):
            tar.add(item['source'], arcname=member, recursive=False)
    manifest = {'version': plan['version'], 'archive': archive.as_posix(), 'archive_sha256': digest_file(archive),
        'plan': plan, 'plan_digest': content_digest(plan), 'input_digests': inputs.digests,
        'reports': reports, 'files': {k: {a: b for a, b in v.items() if a != 'source'} for k, v in payload.items()},
        'session_provenance': plan['session_rule'], 'disclosure': plan['disclosure'],
        'source_task': 'pick_place_sequence scene; trajectories terminate after standalone pick success, no place labels',
        'training_stream': 'unassisted_scripted_pick_success; failed attempts retained but excluded'}
    _json(manifest_path, manifest)
    checksum_path.write_text(f'{manifest["archive_sha256"]}  {archive.as_posix()}\n')
    return {'archive': str(archive), 'manifest': str(manifest_path),
        'episodes': {s: {'successes': r['successes'], 'attempts': r['attempts']} for s, r in reports.items()}}
