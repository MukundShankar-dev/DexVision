# Status

**Active checkpoint:** V1 — cameras and image observations (`docs/PLAN.md`).
**State:** cameras, rig, tests and measurements done; awaiting owner review of
sample frames and the resolution choice.
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

Place details (spec, baselines, gate) are in `configs/tasks/place.yaml`,
`results/place/` and the commit history.

## V1 (`configs/cameras.yaml`, `dexvision/sim/cameras.py`)

- Two fixed cameras, `policy_right` and `policy_left`: a mirrored pair 0.52 m
  from the work area, 40° down. Adding them leaves every skill's expert
  rollout bit-identical. The goal marker and decals are not rendered.
- Desktop rendering is software (WSL: llvmpipe; D3D12 to the 5070 Ti was
  slower, 160 ms). Both cameras: 68 ms per step with a 4096 px shadow map,
  46 ms with 1024 (same image: 0.26/255 mean difference), 19 ms without
  shadows; 6 workers give about 63 steps/s. Laptop (Metal): 13 ms.
- 96 vs 128 px renders at the same speed; storage is a few GB per skill.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
