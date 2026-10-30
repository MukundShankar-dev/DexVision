# Status

**Active checkpoint:** V4 — press from images (`docs/PLAN.md`). **State:**
trained, selected, tested once per method; awaiting owner sign-off.
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
| press from images | 92 (fails: lower bound 39%) | 194 | both cameras blanked: 0 and 3 |

Diffusion passes every skill. MLP-BC fails press (state and images: it
averages push-in and pull-out demos) and place seed 1. Open: MLP-BC push
back-of-hand stalls 2.0–2.5 cm short. Vision: two 96 px cameras, per-camera
CNN with 32 keypoints; on pick, Diffusion relies on `policy_left` alone.

## V4 — press_vision (details in `results/press_vision/`)

Spec: press without `button_position` and `button_depth` (owner decision),
plus both cameras; test seeds 310000–320000. Demos 989/1000, replay 50/50.

**Gate (approved by the owner 2026-10-29 — "Continue" in reply to the
proposal — before any training):** test Wilson lower bound ≥ 80%
(≥ 170/200), and val with both cameras blanked ≤ 10%.

| 200 episodes each | Diffusion | MLP-BC |
|---|---|---|
| test (one run each) | **194 [93.6, 98.6%] passes** | **92 [39.2, 52.9%] fails** |
| val, selected step | 196 (60k) | 93 (30k; 60k: 36) |
| val, both cameras / only left / only right | 3 / 57 / 56 | 0 / 2 / 0 |
| val ceiling: same method on state | 197 | 165 |
| probe, frozen encoder, button error (scratch 0.10 cm) | 0.15 cm | 0.20 cm |

Baselines on val: zero 0, random 0, expert 200. Every failure of both
methods is "held, not released" except one Diffusion test seed (pressed,
not held). Diffusion vs its state twin: 194 both, p = 1. MLP-BC vs its
state twin: 84 lost, 12 gained, p = 2e-14. Its encoder finds the button
(0.20 cm), so the likely cause is the regression averaging push-in and
pull-out, worse than on state. Unlike pick, Diffusion needs both cameras.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
