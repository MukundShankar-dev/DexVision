# Status

**Active checkpoint:** V3 — pick from images (`docs/PLAN.md`). V1 and V2
approved. **State:** trained, tested once per method; both pass the gate;
awaiting owner sign-off.
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

Diffusion passes every skill (push, press, place on two seeds). MLP-BC fails
press (stalls half-pressed: regression averages push-in and pull-out demos)
and place seed 1. Open: MLP-BC push back-of-hand stalls 2.0–2.5 cm short.

**V1–V2:** two 96 px cameras (`configs/cameras.yaml`), per-camera CNN with
32 keypoints (`vision:` in `configs/learning/method.yaml`).

## V3 — pick_vision (details in `results/pick_vision/`)

Spec: pick without `object_*`, plus both cameras; test seeds 300000–310000.
Demos 982/1000 (x86-64). Gate (approved 2026-10-29, before training): test
lower bound ≥ 80%, every object ≥ 60%, both cameras blanked ≤ 10% on val.

| 200 episodes | Diffusion | MLP-BC |
|---|---|---|
| **Test (ledgered)** | **189** [90.4, 96.9%], worst puck_light 23/29 | **200** [98.1, 100%] |
| Val | 191 (state: 197; zero, random: 0) | 200 (state: 200) |
| Val, both cameras blanked | 0 | 0 |
| Val, only `policy_left` / only `policy_right` | 178 / 0 | 16 / 37 |

**Both pass.** Diffusion relies on `policy_left` alone; MLP-BC needs both.
Paired with state on val: Diffusion 188 both, 9 only state, 3 only vision
(McNemar p = 0.15). Its 9 failures all touch the object (5 dropped before
0.5 s), with no start-position cluster; keypoint videos reproduce all 9.
Probe (object position from frozen features): scratch 0.22 cm, Diffusion
0.67, MLP-BC 0.91 (mean guess 4.9). Offline action error 0.048 (state 0.047).

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` (pick, push, place) and `button_position` (press) are
  privileged simulator state.
