# Status

**Active checkpoint:** R7 — place (`docs/PLAN.md`)
**State:** spec, expert and val baselines done; gate approved; data, training
and test running.
**Done:** pick, reach, push ±90°, press (button anywhere; owner approved
2026-10-28). Tags: `legacy-v1`, `push-45`, `press-depth-limit`,
`press-fixed-button` (earlier specs; their test runs stay in the ledger).

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
| press (button anywhere) | 158 | 197 | button: 0 and 6 (expert, mean button: 14) |

Diffusion beats MLP-BC only on rightward (back-of-hand) pushes: 106/114 vs
83/114 over both seeds (Fisher p = 0.0001). The MLP does not average the two
hand postures; it holds the right one and times out. Why is not yet known.

## Place (`configs/tasks/place.yaml`, `dexvision/policies/place_expert.py`)

- **Task:** one of six objects starts in the pick region; carry it to a goal
  in x −0.12..−0.05, y −0.10..0.10 (≥ 6 cm away) and leave it there upright
  (tilt ≤ 10°, resting), released, within 2 cm, for 0.5 s, within 20 s. It
  must be lifted ≥ 3 cm on the way. The goal's far edge is where the hand
  starts to hit the trays, wall and button (a goal sweep on train seeds).
  Landing speed is reported, not failed.
- **Expert:** the pick expert unchanged, then carry and lower by the measured
  object position, open thumb first, slide back, rise; place again if the
  object is off the goal. Train 0–299: 300/300 clean, 264/300 with DART.

| Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|
| val | zero | 0 / 100 | [0.0%, 3.7%] | timeouts |
| val | random | 0 / 100 | [0.0%, 3.7%] | |
| val | expert | 200 / 200 | [98.1%, 100%] | landing ≤ 0.06 m/s |
| val | expert + DART noise | 169 / 200 | [78.8%, 88.9%] | light puck 27/36 |
| val | expert, mean goal | 8 / 200 | [2.0%, 7.7%] | |

**Place gate (approved by the owner 2026-10-28, before any training).** A learned policy passes only
if its single 200-episode test run has a Wilson 95% lower bound ≥ 80%
(≥ 170/200) and every object succeeds ≥ 60%.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
