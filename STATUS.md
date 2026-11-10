# Status

**Active checkpoint:** A2 — how robust are the V3–V6 policies to varied
appearance (`docs/PLAN.md`, A). **State:** eval code, data configs and the
run script are ready; nothing run on the desktop yet (waiting for it).
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

A1 (done): varied per episode seed, from its own streams: lighting; table,
wall, trays and floor (colour, 16 procedural textures, 4 held out of train
seeds); object and button colour (fully random hue), texture and shine.
Applied to the rendered scene only; the model is unchanged and physics is
bit-identical (tests). Ranges: `configs/appearance.yaml`. Owner decisions:
object colour fully random; held-out textures in val and test; no camera
jitter. Contact sheets: `python -m dexvision.apps.appearance_sheet`.

A2 (ready to run, val only, no training; `runs/a2_eval.sh` on the desktop):
1. Checks on x86-64: appearance tests; pick_vision Diffusion nominal val
   reproduces the committed result episode for episode.
2. All factors: Diffusion and MLP-BC on all four skills, 200 val seeds,
   paired with nominal (McNemar), phases, success by key-light brightness,
   object–table contrast and plain / training / held-out texture.
3. One factor at a time (Diffusion).
4. Varied demos (`configs/data/*_vision_varied*.yaml`); scratch probe and
   the existing encoders' frozen probes on varied frames, per object.

The A3 gate is written from these results, before any training.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
