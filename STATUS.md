# Status

**Active checkpoint:** R7 — press (`docs/PLAN.md`)
**State:** first test runs done; both methods FAIL the gate (below).
Diagnosing on val before any change; test budget used 1 of 3 per method.
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
| press (fails gate ≥170) | 33 | 141 | n/a (button never moves) |

Diffusion beats MLP-BC only on rightward (back-of-hand) pushes: 106/114 vs
83/114 over both seeds (Fisher p = 0.0001). The MLP does not average the two
hand postures; it holds the right one and times out. Why is not yet known.

## Press (`configs/tasks/press.yaml`, `dexvision/policies/press_expert.py`)

- **Task:** the wall button is fixed; the hand starts anywhere in an
  8 × 12 × 8 cm box. Success (measured on the button joint): pushed in
  ≥ 15 mm for 0.25 s, then back out (≤ 3 mm) with no hand contact for
  0.5 s, within 7.5 s. Pushing past 24 mm (into the soft end stop) fails,
  as does hand contact with the wall fixture, trays or table.
- **Expert:** a pointing fist. It steers the measured index fingertip in
  front of the button, then presses servoing on the measured depth (target
  18 mm), and pulls back after 5 measured pressed steps. Train seeds 0–299:
  300/300 clean, 287/300 with DART noise.

| Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|
| val | zero | 0 / 100 | [0.0%, 3.7%] | 100 timeouts |
| val | random | 0 / 100 | [0.0%, 3.7%] | 71 collisions |
| val | expert | 200 / 200 | [98.1%, 100%] | peak depth ≤ 20 mm |
| val | expert + DART noise | 187 / 200 | [89.2%, 96.2%] | 7 overpressed |
| val | open-loop expert replay | 8 / 1000 | [0.4%, 1.6%] | 10 recordings × 100 seeds |

Learned (val, selected step): MLP-BC 39/200 (step 50k), Diffusion 160/200
(step 60k, still rising). Test: MLP-BC 33/200 [12.0%, 22.3%] (94 never
release, 73 overpressed); Diffusion 141/200 [63.8%, 76.4%] (53 overpressed,
6 timeouts). Both press deeper than the expert (median peak 22 mm vs 19.7;
the noisy demos reach 20.7 median, 23.9 max, against the 24 mm limit).

The open-loop row shows a memorized motion does not work: the policy needs
feedback on where the hand is.

**Press gate (approved by the owner 2026-10-28, before any training).** A
learned policy passes only if its single 200-episode test run has a Wilson
95% lower bound ≥ 80% (≥ 170/200).

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` observations (pick, push) are privileged simulator state.
- Press: the button never moves, so its position is not an input. The
  button depth is also implied by the fingertip position, so hiding depth
  alone may not hurt a learned policy.
