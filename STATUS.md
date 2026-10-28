# Status

**Active checkpoint:** R7 — press (`docs/PLAN.md`)
**State:** press spec v3 (button anywhere on the wall): val baselines
done; gate needs owner re-confirmation before training.
**Done:** pick, reach, push ±90° (owner approved 2026-10-22). Tags:
`legacy-v1` (old pipeline), `push-45` (the first push spec).

## Shared protocol (every skill)

- **Method:** `configs/learning/method.yaml`, unchanged since R5. Diffusion
  and MLP-BC, 60k steps, checkpoints every 10k, desktop RTX 5070 Ti.
- **Selection:** the most val successes on the first 200 val seeds, ties to
  the later step. Then one ledgered 200-episode test run per method.
- **Platform:** results are evaluated on the desktop (Linux x86-64); MuJoCo
  is not bit-identical across arm64 and x86-64.
- **Test budget:** 3 runs per (skill, method, test seed range).

## Results (test, 200 episodes each; details in `results/` and git history)

| Skill | MLP-BC | Diffusion | Check with the key input hidden (val) |
|---|---|---|---|
| pick | 200 | 200 | object pose: 0 and 12 |
| reach | 200 | 200 | goal: 1 and 2 |
| push ±90°, seed 0 | 188 | 194 | goal: 17 and 23 (expert without goal: 26) |
| push ±90°, seed 1 | 180 | 194 | |

Diffusion beats MLP-BC only on rightward (back-of-hand) pushes: 106/114 vs
83/114 over both seeds (Fisher p = 0.0001). The MLP does not average the two
hand postures; it holds the right one and times out. Why is not yet known.

## Press (`configs/tasks/press.yaml`, `dexvision/policies/press_expert.py`)

- **Task:** the button's centre is placed uniformly on the wall face
  (y −0.11..0.11, z 0.12..0.28; set by the wall's extent) and the hand
  starts anywhere in an 8 × 12 × 8 cm box. Success (measured on the button
  joint): in ≥ 15 mm for 0.25 s, then out (≤ 3 mm) with no hand contact for
  0.5 s, within 7.5 s. Hand contact with the wall, trays or table fails.
  Bottoming out is allowed; peak hand-button force is reported.
- **Observed:** hand state, button depth, and `button_position` (privileged).
- **Earlier specs:** v1 (tag `press-depth-limit`) failed presses past 24 mm
  (about 6 N; dropped as unjustified): MLP-BC 33, Diffusion 141. v2 (tag
  `press-fixed-button`) had a fixed button: MLP-BC 107, Diffusion 196. All
  in the ledger. Test range now 220000–230000.
- **Expert:** a pointing fist that steers the measured fingertip in front
  of the measured button, presses servoing on depth (target 18 mm), and
  pulls back after 5 pressed steps.

| Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|
| val | zero | 0 / 100 | [0.0%, 3.7%] | 100 timeouts |
| val | random | 0 / 100 | [0.0%, 3.7%] | 71 collisions |
| val | expert | 200 / 200 | [98.1%, 100%] | |
| val | expert + DART noise | 199 / 200 | [97.2%, 99.9%] | 1 collision |
| val | expert, mean button position | 14 / 200 | [4.2%, 11.4%] | 186 collisions |
| val | open-loop expert replay | 6 / 1000 | [0.3%, 1.3%] | |
| train | demos kept (DART) | 989 / 1000 | [98.0%, 99.4%] | 96k transitions |

**Press gate (proposed for v3, before any training).** A learned policy
passes only if its single 200-episode test run has a Wilson 95% lower
bound ≥ 80% (≥ 170/200).

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` observations (pick, push) are privileged simulator state.
- Press: `button_position` is simulator ground truth, like `object_*`.
