# Status

**Active checkpoint:** R7 — push, widened to ±90° (`docs/PLAN.md`)
**State:** push ±90° trained and tested; both methods pass. Awaiting owner
sign-off before press.
**Done:** pick, reach, push ±45° (git tag `push-45`). Legacy: tag `legacy-v1`.

## Shared protocol (every skill)

- **Method:** `configs/learning/method.yaml`, unchanged across skills.
  Diffusion and MLP-BC, seed 0, 60k steps, checkpoints every 10k, trained on
  the desktop RTX 5070 Ti.
- **Selection:** the most val successes on the first 200 val seeds, ties to
  the later step. Then one ledgered 200-episode test run per method.
- **Platform:** all results are evaluated on the desktop (Linux x86-64).
  MuJoCo is not bit-identical across arm64 and x86-64.
- **Test budget:** 3 runs per (skill, method, test seed range).

## Results so far (test, 200 episodes each)

| Skill | MLP-BC | Diffusion | Goal or object hidden (val) |
|---|---|---|---|
| pick | 200 | 200 | object pose hidden: 0 and 12 |
| reach | 200 | 200 | goal hidden: 1 and 2 |
| push ±45° | 198 | 199 | goal hidden: 65 and 70 (the expert without the goal: 69) |
| push ±90° seed 0 | 188 | **194** | goal hidden: 17 and 23 (the expert without the goal: 26) |
| push ±90° seed 1 | 180 | **194** | |

Push ±90° is the first task where diffusion beats MLP-BC, on rightward
(back-of-hand) pushes only. Pooled over both training seeds, test right
sector: diffusion 106/114 vs MLP 83/114 (Fisher p = 0.0001; each seed
alone favours diffusion). Forward and left: no difference.

## Push ±90° (`configs/tasks/push.yaml`, `dexvision/policies/push_expert.py`)

- **Task:** slide one of four objects 5–8 cm toward a goal in any direction
  not toward the arm (±90° of +x). Success: within 2 cm of the goal and
  upright for 0.5 s, within 10 s. Hand contact with trays, wall fixture or
  button fails. Test seeds are now 210000–220000.
- **Expert:** three pushing faces. Each approach picks the one needing the
  least hand turn:
  - forward: fingers bent at the knuckle into a wall;
  - left: the palm, thumb up;
  - right: the back of the hand, thumb up.
  The thumb-up faces lean 30° so they touch below the tall cylinder's
  tipping height. Train seeds 0–299: 300/300 clean, 297/300 with DART noise.
  First face used on those seeds: wall 150, back 80, palm 70.

| Split | Policy | Successes / n | Wilson 95% CI | right / forward / left |
|---|---|---|---|---|
| val | zero | 0 / 100 | [0.0%, 3.7%] | |
| val | random | 3 / 100 | [1.0%, 8.5%] | 1/30, 1/50, 1/20 |
| val | expert | 200 / 200 | [98.1%, 100%] | 60/60, 97/97, 43/43 |
| val | expert + DART noise | 199 / 200 | [97.2%, 99.9%] | 60/60, 97/97, 42/43 |
| val | expert, goal = training mean | 26 / 200 | [9.0%, 18.4%] | floor without the goal |

Sectors: right < −45° ≤ forward ≤ 45° < left (`push_angle_deg` metric).

**Gate** (approved before training): Wilson lower bound ≥ 80%, every object
≥ 60%, every direction sector ≥ 60%. **Both pass.**

| Test (seeds 210000–210199) | Successes | Wilson 95% CI | right / forward / left |
|---|---|---|---|
| MLP-BC seed 0, step 20k | 188 / 200 | [89.8%, 96.5%] | 45/57, 97/97, 46/46 |
| MLP-BC seed 1, step 30k | 180 / 200 | [85.1%, 93.4%] | 38/57, 96/97, 46/46 |
| Diffusion seed 0, step 40k | 194 / 200 | [93.6%, 98.6%] | 54/57, 96/97, 44/46 |
| Diffusion seed 1, step 40k | 194 / 200 | [93.6%, 98.6%] | 52/57, 97/97, 45/46 |

All four runs pass the gate; each method has used 2 of 3 test runs here.
- Data: 982/1000 kept; 200/200 replays exact. Val selection (seed 0 / 1):
  MLP 185 / 178, Diffusion 192 / 193 of 200.
- **The prediction was wrong about why.** I expected the MLP to average the
  two hand postures near the ±45° switch between faces. It doesn't: there is
  no gap at 30–60°, and its failed rightward pushes hold the back-of-hand roll
  (≈63°) cleanly, then time out. Why diffusion does better there is not yet
  understood.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in). The
  thumb-up faces use 60–120° of forearm roll, within a human range.
- `object_*` observations are privileged simulator state (pick and push).
