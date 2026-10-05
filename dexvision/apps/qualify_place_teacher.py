"""Run the versioned place-teacher feasibility diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dexvision.evaluation.place_repair import qualify_place_teacher, qualify_place_teacher_coverage
from dexvision.evaluation.place_teacher_search import qualify_place_search_coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    print(f'Place teacher: {args.config}; output: {args.output_dir}', flush=True)
    try:
        version = yaml.safe_load(Path(args.config).read_text())['version']
        if version in {'level5/place-teacher-search-coverage-v1',
                       'level5/place-teacher-search-coverage-v2'}:
            run = qualify_place_search_coverage
        elif version in {'level5/place-teacher-coverage-v1', 'level5/place-teacher-coverage-v2',
                         'level5/place-teacher-coverage-v3'}:
            run = qualify_place_teacher_coverage
        else:
            run = qualify_place_teacher
        result = run(args.config, args.output_dir)
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.exit(2, f'Place teacher failed: {exc}\n')
    print(json.dumps(result, indent=2))
    return 0 if result['teacher_feasibility_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
