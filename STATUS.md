# Status

**Active checkpoint:** V5 — push from images (`docs/PLAN.md`). **State:**
spec, phases, data, baselines and probe done; gate approved; training.
**Done:** pick, reach, push ±90°, press (button anywhere), place (owner
approved 2026-10-28); V1–V3, pick from images (approved 2026-10-29); V4, press from images
(approved 2026-10-30). Tags: `legacy-v1`, `push-45`, `press-depth-limit`,
`press-fixed-button` (earlier specs; their test runs stay in the ledger).

## Shared protocol (every skill)

- **Method:** `configs/learning/method.yaml`, unchanged since R5. Diffusion
  and MLP-BC, 60k steps, checkpoints every 10k, desktop RTX 5070 Ti.
- **Selection:** the most val successes on the first 200 val seeds, ties to
  the later step. Then one ledgered 200-episode test run per method.
- **Platform:** desktop (Linux x86-64); MuJoCo differs slightly on arm64.
  Test budget: 3 runs per (skill, method, test seed range).

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
| pick from images | 200 | 189 | both cameras blanked: 0 and 0 |
| press from images | 92 (fails: lower bound 39%) | 194 | both cameras blanked: 0 and 3 |

Diffusion passes every skill. MLP-BC fails press (state and images: it
averages push-in and pull-out demos) and place seed 1. Open: MLP-BC push
back-of-hand stalls 2.0–2.5 cm short. Vision: two 96 px cameras, per-camera
CNN with 32 keypoints; on pick, Diffusion relies on `policy_left` alone.

## V5 — push_vision (val, 200 episodes; details in `results/push_vision/`)

Spec: push without `object_*` (pose, velocity, shape, size), plus both
cameras; `goal_position` stays (the instruction; the marker is hidden from
the cameras). Test seeds 320000–330000. Demos 982/1000, replay 50/50; val
demos 100/100.

| Baseline (val) | Success | Phases of failures |
|---|---|---|
| zero | 0 [0, 1.9%] | all no contact |
| random | 5 [1.1, 5.7%] | 79 no contact, 66 not at goal, 35 tipped |
| expert (state) | 200 | |
| state Diffusion / MLP-BC (ceiling) | 192 / 185 | mostly at goal not held, not at goal |
| same, object_* hidden (R7) | 73 / 47 | |

- **Probe** (scratch encoder, 7722 val frames): object position error
  0.17 cm mean (mean guess 3.0 cm), 0.20 cm while pushed; object identity
  100%.
- The object starts within ±2 × ±3 cm of a fixed spot, so a policy that
  never sees it still succeeds 37% (state Diffusion, object hidden).

**Gate (approved by the owner 2026-11-02, before any training).** A vision policy passes if (1) its single
200-episode test run has Wilson lower bound ≥ 80% (≥ 170/200), and (2) on
val, blanking both cameras makes it significantly worse: exact McNemar
p < 0.01 on the 200 paired seeds. (V3/V4 used "blanked ≤ 10%"; here a
policy that ignores the object can reach 37%, so that bar could fail a
policy that does use its cameras.)

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
