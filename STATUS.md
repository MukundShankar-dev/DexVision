# Status

**Active checkpoint:** A3, pick — train on varied appearance
(`docs/PLAN.md`, A). **State:** trained and evaluated on val 2026-11-11;
**the gate is not met.** The test run was held, not spent (owner's call);
diagnostics below; awaiting the owner's choice of what to try next.
**Done:** pick, reach, push ±90°, press (button anywhere), place (owner
approved 2026-10-28); V1–V3, pick from images (approved 2026-10-29); V4, press from images
(approved 2026-10-30); V5, push from images (approved 2026-11-02); V6, place from images (approved 2026-11-03); A1, appearance variation (approved 2026-11-10); A2, robustness of the V3–V6 policies (approved 2026-11-11). Tags: `legacy-v1`, `push-45`, `press-depth-limit`,
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

A2 (val, 200 paired seeds, no training; `runs/a2_eval.sh`, desktop).
Successes out of 200:

| Skill | Unvaried | Lighting | Surfaces | Objects | All (Diffusion) | All (MLP-BC) |
|---|---|---|---|---|---|---|
| pick | 191 | 47 | 64 | 45 | 20 | 18 (from 200) |
| press | 196 | 137 | 70 | 48 | 8 | 6 (from 93) |
| push | 192 | 89 | 79 | 131 | 48 | 17 (from 188) |
| place | 180 | 38 | 43 | 102 | 27 | 2 (from 156) |

- Every drop is significant (McNemar p < 1e-13); no pattern by brightness,
  contrast or texture. Existing encoders lose the object on varied frames
  (frozen probe 3.1–5.2 cm, the mean guess is 3.0–7.0), but a scratch probe
  on the same frames reaches 0.39–1.17 cm: position is in the pixels.

**A3 gate (approved 2026-11-11; pick first):** spec `pick_vision_varied` (all
factors, test seeds 340000–350000), trained on the 982 varied demos, method
unchanged. Pass: Diffusion test Wilson lower bound ≥ 80% (≥ 172/200) on
varied scenes, and both cameras blanked significantly worse on varied val
(McNemar p < 0.01). Reported, not gated: MLP-BC; val on unvaried scenes vs
the V3 policy; success by object and by held-out texture; the A2 probes.

**A3 pick result (val only; desktop, commit `90c89e8`; no test run):**
baselines on varied val are zero 0/200 and random 0/200 (the expert needs
the object pose, so it cannot act on an image-only spec; state ceiling 200).
Selected checkpoints, successes out of 200 paired val seeds:

| | Diffusion (step 60k) | MLP-BC (step 50k) |
|---|---|---|
| Varied scenes | 54, CI [21%, 34%] | 72, CI [30%, 43%] |
| Varied, both cameras blanked | 35 (p = 0.015) | 64 (p = 0.38) |
| V3 policy on the same varied scenes | 20 (p = 8e-6) | 18 (p = 4e-11) |
| Unvaried scenes (V3 policy) | 70 (191) | 120 (200) |

- **Both gate parts fail.** Varied val is far below 172/200, and blanking
  the cameras costs Diffusion 19 successes and MLP-BC 8 (blind, Diffusion
  touches the object in 102 episodes instead of 157; MLP-BC is unchanged).
- New encoders on varied val frames (frozen probe): position error 5.2 and
  4.7 cm, the mean guess is 4.9; object identity 21% and 18%, chance is
  17%. No better than the V3 encoders (4.2 and 4.8 cm). The scratch probe
  on the same frames reaches 0.8 cm, so the frames suffice for locating.
- Diffusion failures: contact without grasp 86 of 146, no contact 43.
  Held-out object textures are no worse (10/31 against 44/169). Light puck
  1/36, heavy puck 5/34.
- **Training scenes** (first 200 train seeds, `--split train`): Diffusion
  114/200, MLP-BC 171/200, both far above their val scores.
- **Overfit check** (demos of the first 100 train seeds, 30k steps;
  successes of 100; the control is the same seeds with one fixed look):

  | | Training scenes | Val |
  |---|---|---|
  | Diffusion, unvaried (control) | 80 | 78 |
  | Diffusion, varied | 70 | 28 |
  | MLP-BC, varied | 83 | 10 |

  With one look, what 98 demos teach transfers to new scenes; with varied
  looks it does not, and ten times the demos does not close the gap. Fitting
  the training scenes is only slightly worse with varied looks (70 against
  80, paired p = 0.04).
- **Steering** (scratch script, 40–60 val episodes; slope of palm xy at
  step 25, before contact, against object xy): V3 policy on unvaried
  scenes 0.96 and 0.75; new Diffusion −0.03 and 0.18; new MLP-BC −0.02 and
  0.47; cameras blanked, about 0. The new policies barely steer to the
  object. Blind policies still touch it because objects spread only 2–3 cm.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
