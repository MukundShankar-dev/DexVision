# Status

**Active checkpoint:** none — every planned skill is done (`docs/PLAN.md`)
**State:** the next step ("Later" in the plan) needs planning with the owner.
**Done:** pick, reach, push ±90°, press (button anywhere), place (owner
approved 2026-10-28). Tags: `legacy-v1`, `push-45`, `press-depth-limit`,
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
| place | 182 | 194 | goal: 6 and 13 (expert, mean goal: 8) |

Push: Diffusion beats MLP-BC only on back-of-hand pushes (106/114 vs 83/114,
p = 0.0001); why is not yet known.

## Place (`configs/tasks/place.yaml`, `dexvision/policies/place_expert.py`)

- **Task:** lift one of six objects ≥ 3 cm, carry it to a goal ≥ 6 cm away
  (x −0.12..−0.05, y −0.10..0.10) and leave it released and upright within
  2 cm for 0.5 s, within 20 s. Landing speed is reported, not failed.
- **Expert:** pick expert, then carry, lower, thumb-first release, slide
  back, rise; re-place if off the goal.

| Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|
| val | zero, random | 0 / 100 each | [0.0%, 3.7%] | |
| val | expert | 200 / 200 | [98.1%, 100%] | landing ≤ 0.06 m/s |
| val | expert + DART noise | 169 / 200 | [78.8%, 88.9%] | light puck 27/36 |
| val | expert, mean goal | 8 / 200 | [2.0%, 7.7%] | |
| val | MLP-BC, step 60k | 177 / 200 | [83.3%, 92.2%] | selected |
| val | Diffusion, step 60k | 193 / 200 | [93.0%, 98.3%] | selected |
| test | MLP-BC | 182 / 200 | [86.2%, 94.2%] | pucks 23/29, 20/28; **passes** |
| test | Diffusion | 194 / 200 | [93.6%, 98.6%] | worst object 29/31; **passes** |

Data: 1000 noisy expert demos, 878 kept (light puck 133/177). Every test
failure was lifted and ended off the goal (timeouts 15 and 4, fell off the
table 3 and 2). MLP-BC is slower (median 233 steps vs 164; expert 152).
Learned policies sometimes let go high: landing > 0.2 m/s (a fall of ≥ 2 cm)
in 7 (MLP) and 4 (Diffusion) test successes; the expert stays ≤ 0.06 m/s.

**Gate (approved 2026-10-28, before training):** test Wilson lower bound
≥ 80% (≥ 170/200) and every object ≥ 60%.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
