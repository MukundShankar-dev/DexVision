# Status

**Active checkpoint:** V4 — press from images (`docs/PLAN.md`). **State:**
spec, diagnostics, data, baselines and probe done; gate approved; training.
**Done:** pick, reach, push ±90°, press (button anywhere), place (owner
approved 2026-10-28); V1–V3, pick from images (approved 2026-10-29). Tags: `legacy-v1`, `push-45`, `press-depth-limit`,
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
| pick from images | 200 | 189 | both cameras blanked: 0 and 0 |

Diffusion passes every skill (push, press, place on two seeds). MLP-BC fails
press (stalls half-pressed: regression averages push-in and pull-out demos)
and place seed 1. Open: MLP-BC push back-of-hand stalls 2.0–2.5 cm short.

**Vision (V1–V3):** two 96 px cameras, per-camera CNN with 32 keypoints
(`vision:` in `configs/learning/method.yaml`). Pick from images: details in
`results/pick_vision/`; Diffusion relies on `policy_left` alone (178/200
with only it, 0/200 without it), MLP-BC needs both.

## V4 — press_vision (val, 200 episodes; details in `results/press_vision/`)

Spec: press without `button_position` and `button_depth` (a sensor in the
button, not on the robot; owner decision), plus both cameras; test seeds
310000–320000. Demos 989/1000 kept, replay 50/50; val demos 99/100.

| Baseline (val) | Success | Phases of failures |
|---|---|---|
| zero, random | 0/200, 0/200 [0, 1.9%] | random: 130 hand collisions |
| expert (state) | 200/200 | |
| state Diffusion / MLP-BC (ceiling) | 197 / 165 | all held but not released |

- **Probe** (encoder alone, 9875 val frames): button position error 0.10 cm
  mean (mean guess 7.0 cm), 0.14 cm while pressed.
- Evaluator now pools workers (`--workers`, identical episodes, ~5× faster)
  and prints progress.

**Gate (approved by the owner 2026-10-29 — "Continue" in reply to the proposal — before any training).** The same as V3: a vision policy
passes if its single 200-episode test run has Wilson lower bound ≥ 80%
(≥ 170/200), and on val with both cameras blanked it succeeds ≤ 10%.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
