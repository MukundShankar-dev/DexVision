"""Independently audit a selected Level 5.4A place confirmation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/level5/place_policy_safety_v8.yaml'))
    parser.add_argument('--selection-config', type=Path, default=Path('configs/level5/place_safety_selection_v8.yaml'))
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args(argv)
    print(f'DexVision place physical audit: seed {args.seed}, recipe {args.config}', flush=True)
    try:
        from dexvision.evaluation.place_repair_audit import audit_cadence_confirmation
        import yaml

        version = yaml.safe_load(args.selection_config.read_text())['version']
        if version == 'level5/place-pad-release-selection-v25':
            from dexvision.evaluation.place_pad_release_selection import load_selection_spec
        elif version == 'level5/place-wrist-hold-selection-v24':
            from dexvision.evaluation.place_wrist_hold_selection import load_selection_spec
        elif version == 'level5/place-bin-opening-selection-v23':
            from dexvision.evaluation.place_bin_opening_selection import load_selection_spec
        elif version == 'level5/place-supported-descent-selection-v22':
            from dexvision.evaluation.place_supported_descent_selection import load_selection_spec
        elif version == 'level5/place-shadow-selection-v21':
            from dexvision.evaluation.place_shadow_selection import load_selection_spec
        elif version == 'level5/place-object-calibration-selection-v20':
            from dexvision.evaluation.place_object_calibration_selection import load_selection_spec
        elif version == 'level5/place-full-source-selection-v19':
            from dexvision.evaluation.place_full_source_selection import load_selection_spec
        elif version == 'level5/place-geometric-release-selection-v18':
            from dexvision.evaluation.place_geometric_release_selection import load_selection_spec
        elif version == 'level5/place-source-wrist-selection-v17':
            from dexvision.evaluation.place_source_wrist_selection import load_selection_spec
        elif version == 'level5/place-contact-selection-v16':
            from dexvision.evaluation.place_contact_selection import load_selection_spec
        elif version == 'level5/place-physical-selection-v15':
            from dexvision.evaluation.place_physical_selection import load_selection_spec
        elif version == 'level5/place-wrist-selection-v14':
            from dexvision.evaluation.place_wrist_selection import load_selection_spec
        elif version == 'level5/place-recovery-selection-v13':
            from dexvision.evaluation.place_recovery_selection import load_selection_spec
        elif version == 'level5/place-adaptive-selection-v12':
            from dexvision.evaluation.place_adaptive_selection import load_selection_spec
        elif version == 'level5/place-slot-clearance-selection-v11':
            from dexvision.evaluation.place_slot_clearance_selection import load_selection_spec
        elif version == 'level5/place-clearance-selection-v10':
            from dexvision.evaluation.place_clearance_selection import load_selection_spec
        elif version == 'level5/place-slow-selection-v9':
            from dexvision.evaluation.place_slow_selection import load_selection_spec
        elif version == 'level5/place-safety-selection-v8':
            from dexvision.evaluation.place_safety_selection import load_selection_spec
        else:
            raise ValueError('Unsupported place selection audit recipe')

        protocol = load_selection_spec(args.selection_config)[0]
        if args.seed not in protocol['seeds']:
            raise ValueError('Unknown place repair seed')
        root = Path(protocol['output_dir'])
        result = audit_cadence_confirmation(args.config,
            root / f'seed_{args.seed}' / 'selected.pt',
            root / f'confirmation_seed{args.seed}', root / f'audit_seed{args.seed}',
            selection_config=args.selection_config)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'ERROR: {exc}. Use the dexvision environment.', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
