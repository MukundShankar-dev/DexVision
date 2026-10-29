# Status

**Active checkpoint:** V3 — pick from images (`docs/PLAN.md`). V1 and V2
approved. **State:** spec, diagnostics, data, baselines and probe done; gate
proposed, awaiting approval. No vision policy trained yet.
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

## V1–V2 (cameras, encoder)

- `policy_right`/`policy_left`, 96 px RGB, shadows on, goal marker hidden;
  physics bit-identical with cameras. Per-camera CNN, 32 spatial-softmax
  keypoints, 88 px random crop; both methods. 60k steps: 102 / 78 min.

## V3 — pick_vision (val, 200 episodes; details in `results/pick_vision/`)

Spec `configs/tasks/pick_vision.yaml`: pick without `object_*`, plus both
cameras; test seeds 300000–310000. Demos: 982/1000 kept (x86-64; the state
set was made on arm64, 13 seeds differ in kept/dropped, code reproduces).

| Baseline | Success |
|---|---|
| zero, random | 0/200, 0/200 [0, 1.9%] |
| expert (state, same seeds) | 200/200 |
| state Diffusion / MLP-BC (ceiling) | 197/200 / 200/200 |

- **Probe** (encoder trained alone, 8210 val frames from 98 val episodes):
  object position error 0.22 cm mean (0.35 p90; mean guess 4.9 cm), object
  identity 100%; with the hand near: 0.22 cm; object lifted: 0.26 cm. The
  images hold what pick needs at 96 px.
- **State failures:** Diffusion's 3 all `lifted_not_held` (pucks, 2 heavy).
- Diagnostics per run: phase reached, by object and start cell, metric
  medians, paired McNemar vs state, offline action error, blank-camera val
  runs, keypoint videos (`python -m dexvision.apps.diagnose|probe|keypoint_video`).

**Proposed gate (not yet approved).** A vision policy passes if its single
200-episode test run has Wilson lower bound ≥ 80% (≥ 170/200), every object
≥ 60%, and on val with both cameras blanked it succeeds ≤ 10% (otherwise it
is not using the images and the result does not count).

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
