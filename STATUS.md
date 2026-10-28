# Status

**Active checkpoint:** R7 — press (`docs/PLAN.md`)
**State:** spec revised (depth limit dropped), val baselines re-run; gate
re-confirmed; training both methods.
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

- **Task:** the wall button is fixed; the hand starts anywhere in an
  8 × 12 × 8 cm box. Success (measured on the button joint): pushed in
  ≥ 15 mm for 0.25 s, then back out (≤ 3 mm) with no hand contact for
  0.5 s, within 7.5 s. Hand contact with the wall fixture, trays or table
  fails. Bottoming out is allowed; peak hand-button force is reported.
- **Spec v1** (tag `press-depth-limit`) also failed presses past 24 mm,
  about 6 N into the end stop. Dropped as unjustified. Its test runs
  (MLP-BC 33/200, Diffusion 141/200, mostly "overpressed") stay in the
  ledger; the test range moved to 210000–220000.
- **Expert:** a pointing fist. It steers the measured index fingertip in
  front of the button, presses servoing on the measured depth (target
  18 mm), and pulls back after 5 measured pressed steps.

| Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|
| val | zero | 0 / 100 | [0.0%, 3.7%] | 100 timeouts |
| val | random | 0 / 100 | [0.0%, 3.7%] | 71 collisions |
| val | expert | 200 / 200 | [98.1%, 100%] | peak force ≤ 0.82 N |
| val | expert + DART noise | 194 / 200 | [93.6%, 98.6%] | peak force median 1.4 N, max 31 N |
| val | open-loop expert replay | 25 / 1000 | [1.7%, 3.7%] | 10 recordings × 100 seeds |
| train | demos kept (DART) | 981 / 1000 | [97.0%, 98.8%] | 103k transitions |

**Press gate (approved 2026-10-28, re-confirmed for spec v2 before training).** A learned
policy passes only if its single 200-episode test run has a Wilson 95%
lower bound ≥ 80% (≥ 170/200).

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` observations (pick, push) are privileged simulator state.
- Press: the button never moves, so its position is not an input. The
  button depth is also implied by the fingertip position, so hiding depth
  alone may not hurt a learned policy.
