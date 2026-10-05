"""Frozen ownership and source gates for the additive 5.4A place data."""
import json
from pathlib import Path

import pytest
import yaml

from dexvision.logging.place_amendment import freeze_place_data_plan, load_place_data_plan
from dexvision.learning.training_manifest import digest_file


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_place_data_plan_freezes_disjoint_source_ownership(tmp_path, version):
    config = f'configs/level5/place_data_{version}.yaml'
    plan, _, _, _, _ = load_place_data_plan(config)
    assert plan['publication_success_min'] >= 0.95
    assert freeze_place_data_plan(config, tmp_path / 'frozen') == {
        'train': 144, 'validation': 48}
    frozen = json.loads((tmp_path / 'frozen' / 'plan.json').read_text())
    train, validation = (frozen['assignments'][split] for split in ('train', 'validation'))
    assert {item['episode_id'] for item in train}.isdisjoint(
        item['episode_id'] for item in validation)
    assert {item['seed'] for item in train}.isdisjoint(item['seed'] for item in validation)
    assert {item['session_id'] for item in train}.isdisjoint(
        item['session_id'] for item in validation)


def test_place_data_plan_rejects_changed_feasibility_pin(tmp_path):
    plan = yaml.safe_load(Path('configs/level5/place_data_v1.yaml').read_text())
    plan['teacher_feasibility_sha256'] = '0' * 64
    recipe = tmp_path / 'altered.yaml'
    recipe.write_text(yaml.safe_dump(plan))
    import hashlib
    recipe.with_suffix('.sha256').write_text(
        f'{hashlib.sha256(recipe.read_bytes()).hexdigest()}  {recipe.name}\n')
    with pytest.raises(ValueError, match='feasibility evidence changed'):
        load_place_data_plan(recipe)


def test_v3_place_data_plan_covers_all_cells_with_disjoint_resets(tmp_path):
    config = 'configs/level5/place_data_v3.yaml'
    plan, _, _, _, _ = load_place_data_plan(config)
    cells = {row['id'] for row in yaml.safe_load(
        Path('configs/level4_dataset.yaml').read_text())['coverage_cells']
        if row.get('data_group') == 'pick_place'}
    assert set(plan['splits']['train']['cells']) == cells
    assert set(plan['splits']['validation']['cells']) == cells
    assert freeze_place_data_plan(config, tmp_path / 'frozen') == {
        'train': 240, 'validation': 120}
    frozen = json.loads((tmp_path / 'frozen' / 'plan.json').read_text())
    train, validation = (frozen['assignments'][split] for split in ('train', 'validation'))
    for key in ('seed', 'episode_id', 'session_id'):
        assert {row[key] for row in train}.isdisjoint(row[key] for row in validation)


def test_v3_place_data_plan_rejects_missing_cell(tmp_path):
    plan = yaml.safe_load(Path('configs/level5/place_data_v3.yaml').read_text())
    plan['splits']['validation']['cells'].pop()
    recipe = tmp_path / 'missing.yaml'
    recipe.write_text(yaml.safe_dump(plan))
    import hashlib
    recipe.with_suffix('.sha256').write_text(
        f'{hashlib.sha256(recipe.read_bytes()).hexdigest()}  {recipe.name}\n')
    with pytest.raises(ValueError, match='all workcell cells'):
        load_place_data_plan(recipe)


def test_v3_release_retains_every_attempt_and_physical_audit():
    manifest = json.loads(Path(
        'datasets/level5-place-unassisted-v3_manifest.json').read_text())
    assert manifest['version'] == 'level5/place-unassisted-data-v3'
    assert manifest['archive_sha256'] == digest_file(Path(manifest['archive']))
    assert manifest['physical_audit']['passed']
    assert manifest['physical_audit']['audited_successes'] == 354
    assert len(manifest['physical_audit']['excluded_failures']) == 6
    assert len(manifest['files']) == 354 * 3
    for split, attempts, successes in (('train', 240, 237), ('validation', 120, 117)):
        report = manifest['reports'][split]
        assert (report['attempts'], report['successes']) == (attempts, successes)
        assert len(report['results']) == attempts
        assert all(not any(row['safety_counters'].values()) for row in report['results'])
        assert sum(row['payload'] for row in report['results']) == successes
    train, validation = (manifest['reports'][split]['results']
                         for split in ('train', 'validation'))
    for key in ('seed', 'episode_id', 'session_id'):
        assert {row[key] for row in train}.isdisjoint(row[key] for row in validation)
