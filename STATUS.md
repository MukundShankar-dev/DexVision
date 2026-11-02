# Status

**Active checkpoint:** V5 — push from images (`docs/PLAN.md`). **State:**
trained, selected, tested once per method; owner signed off 2026-11-02.
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
| push from images | 184 | 195 | both cameras blanked: 9 and 59 |

Diffusion passes every skill. MLP-BC fails press (state and images: it
averages push-in and pull-out demos) and place seed 1. Open: MLP-BC push
back-of-hand stalls 2.0–2.5 cm short. Vision: two 96 px cameras, per-camera
CNN with 32 keypoints; on pick, Diffusion relies on `policy_left` alone.

## V5 — push_vision (val, 200 episodes; details in `results/push_vision/`)

Spec: push without `object_*` (pose, velocity, shape, size), plus both
cameras; `goal_position` stays (the instruction; the marker is hidden from
the cameras). Test seeds 320000–330000. Demos 982/1000, replay 50/50.

Val baselines: zero 0, random 5 [1.1, 5.7%], expert 200; state Diffusion /
MLP-BC 192 / 185, and 73 / 47 with `object_*` hidden (R7): the object starts
within ±2 × ±3 cm of a fixed spot, so ignoring it still succeeds ~37%.

**Gate (approved by the owner 2026-11-02, before any training).** (1) Test
Wilson lower bound ≥ 80% (≥ 170/200); (2) on val, blanking both cameras is
significantly worse, exact McNemar p < 0.01 on 200 paired seeds. ("Blanked
≤ 10%" from V3/V4 could fail a policy that uses its cameras, given 37%.)

| 200 episodes each | Diffusion | MLP-BC |
|---|---|---|
| test (one run each) | **195 [94.3, 98.9%] passes** | **184 [87.4, 95.0%] passes** |
| val, selected step | 192 (60k) | 188 (60k) |
| val, both cameras blanked (paired p) | 59 (1e-37) | 9 (3e-54) |
| val, only left / only right | 194 / 65 | 9 / 141 |
| vs state twin on val (paired p) | 187 both, 5/5 (p = 1) | 178 both, 10/7 (p = 0.63) |
| probe, frozen, object error / identity (scratch 0.17 cm / 100%) | 0.44 cm / 95% | 0.43 cm / 97% |

Failures: Diffusion mixes at-goal-not-held and short pushes; MLP-BC's are
almost all pushes stopped 2.0–2.5 cm short, mostly sideways (|angle| > 45°),
the stall seen on state. Each method leans on a different single camera.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
