# Status

**Active checkpoint:** A2 — how robust are the V3–V6 policies to varied
appearance (`docs/PLAN.md`, A). **State:** run and reported 2026-11-10;
awaiting owner sign-off. The A3 gate below is proposed, not yet approved.
**Done:** pick, reach, push ±90°, press (button anywhere), place (owner
approved 2026-10-28); V1–V3, pick from images (approved 2026-10-29); V4, press from images
(approved 2026-10-30); V5, push from images (approved 2026-11-02); V6, place from images (approved 2026-11-03); A1, appearance variation (approved 2026-11-10). Tags: `legacy-v1`, `push-45`, `press-depth-limit`,
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
| place from images | 163 (fails: lower bound 76%, heavy puck 10/32) | 183 | both cameras blanked: 1 and 13 |

Diffusion passes every skill. MLP-BC fails press (state and images: it
averages push-in and pull-out demos), place seed 1 and place from images. Open: MLP-BC push
back-of-hand stalls 2.0–2.5 cm short (state and images). Vision: two 96 px cameras, per-camera
CNN with 32 keypoints; on pick, Diffusion relies on `policy_left` alone.

## A — appearance variation (details in `docs/PLAN.md`, A)

A1 (done): lighting, surfaces and objects (colour, texture, shine) vary per
episode seed in the rendered scene only; physics is bit-identical (tests).
Ranges: `configs/appearance.yaml`. Object colour is fully random; 4 of 16
textures are held out of train seeds; no camera jitter.

A2 (val, 200 paired seeds, no training; `runs/a2_eval.sh`, desktop). Checks:
appearance tests pass on x86-64; the unvaried pick re-run matches the
committed result on 200/200 episodes. Successes out of 200:

| Skill | Unvaried | Lighting | Surfaces | Objects | All (Diffusion) | All (MLP-BC) |
|---|---|---|---|---|---|---|
| pick | 191 | 47 | 64 | 45 | 20 | 18 (from 200) |
| press | 196 | 137 | 70 | 48 | 8 | 6 (from 93) |
| push | 192 | 89 | 79 | 131 | 48 | 17 (from 188) |
| place | 180 | 38 | 43 | 102 | 27 | 2 (from 156) |

- Every drop is significant (McNemar p < 1e-13). On pick and press the
  hand mostly never touches the target. No pattern by brightness, contrast
  or texture.
- Existing encoders on varied frames (frozen probe): position error
  3.1–5.2 cm, close to guessing the mean at 3.0–7.0 cm (unvaried: 0.15–1.2).
- Scratch probe on varied frames: position 0.39–1.17 cm (unvaried
  0.10–0.36), so position is in the pixels. Object identity mostly is not:
  49% pick, 51% place, 29% push; the two pucks are told apart at chance.

**A3 gate (proposed; pick first):** new spec `pick_vision_varied` (all
factors, test seeds 340000–350000), trained on the 982 varied demos, method
unchanged. Pass: Diffusion test Wilson lower bound ≥ 80% (≥ 172/200) on
varied scenes, and both cameras blanked significantly worse on varied val
(McNemar p < 0.01). Reported, not gated: MLP-BC; val on unvaried scenes vs
the V3 policy; success by object and by held-out texture; the A2 probes.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
