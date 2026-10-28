# Status

**Active checkpoint:** caveats (second seeds, MLP-BC diagnostics) done;
next is V1 perception (`docs/PLAN.md`), after owner sign-off.
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
| press, seed 0 | 158 | 197 | button: 0 and 6 (expert, mean button: 14) |
| press, seed 1 | 129 | 196 | |
| place, seed 0 | 182 | 194 | goal: 6 and 13 (expert, mean goal: 8) |
| place, seed 1 | 164 (fails: lower bound 76%, light puck 13/28) | 193 | |

Diffusion passes push, press and place on both seeds (pick and reach: one
seed). MLP-BC fails press on both seeds and place on seed 1. Diagnostics (val, scripts in `runs/` on the desktop):
- Press: MLP-BC presses, starts to pull back, and stalls at ~8 mm depth.
  There the demos push in (59%) or pull out (41%), told apart only by hand
  speed; slowed down, the regression averages them to ~0 and stays stuck.
- Push, back of hand (106/114 vs 83/114): 25 of 35 MLP-BC failures stop
  2.0-2.5 cm from the goal (tolerance 2 cm) and hover. Demos near the goal
  are equally mixed for every face, so that is not the cause. Open.

## Place (`configs/tasks/place.yaml`, `dexvision/policies/place_expert.py`)

- **Task:** lift one of six objects ≥ 3 cm, carry it to a goal ≥ 6 cm away
  (x −0.12..−0.05, y −0.10..0.10) and leave it released and upright within
  2 cm for 0.5 s, within 20 s. Landing speed is reported, not failed.
- **Expert:** pick expert, then carry, lower, thumb-first release, slide
  back, rise; re-place if off the goal.

Val baselines: zero and random 0/100; expert 200/200; expert + DART
169/200; expert with the mean goal 8/200. Demos: 878/1000 kept. Learned
policies sometimes let go higher than the expert (landing > 0.2 m/s in a
few successes; expert ≤ 0.06 m/s). MLP-BC fails mostly on the pucks.

**Gate (approved 2026-10-28, before training):** test Wilson lower bound
≥ 80% (≥ 170/200) and every object ≥ 60%.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
