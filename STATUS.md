# Status

**Active checkpoint:** R5 — Diffusion Policy and MLP-BC, method only (see `docs/PLAN.md`)
**State:** implemented, tests pass. Awaiting owner sign-off before R6.
**Last completed:** R4 — pick data generation with the hardened expert (owner approved 2026-10-21)
**Legacy pipeline:** git tag `legacy-v1`

## Method (`dexvision/learning/`, `configs/learning/method.yaml`)

- **Diffusion Policy:** 1-D conv UNet (128/256/512) with FiLM, 17.7M params;
  DDPM with 100 squared-cosine steps and epsilon prediction; DDIM with 10
  steps. Observation history 2, prediction 16, execute 8. Min-max
  normalization fitted on training data. AdamW at 1e-4, batch 256, 60k steps,
  EMA weights.
- **MLP-BC:** 3×512 MLP regressing the 16-step chunk from the same inputs,
  with the same data, normalization, training settings and policy wrapper.
- **Speed (batch 256), 60k diffusion steps:** RTX 5070 Ti (`ssh desktop`, WSL2)
  34 ms/step, 0.6 h; M1 Pro MPS 265 ms/step, 4.4 h. MLP: 2–4 ms/step. CPU
  inference is about 120 ms per chunk, about 2.5 s per episode.
- **Desktop** (`~/DexVision`, Miniforge env `dexvision`): 66 tests pass, and
  the dataset was copied with all 985 hashes verified. MuJoCo is not
  bit-identical across arm64 and x86-64 (expert val 200/200 on both, but 48
  episodes differ in detail), so all R6 evaluations run on one machine, and
  provenance now records the platform.

## Pick task, expert and data

- **Task** (`configs/tasks/pick.yaml`): one of six objects in an 8 × 12 cm
  region, any yaw. Success (measured): lifted ≥ 5 cm and held by thumb plus
  finger, touching nothing else, for 0.5 s within 8 s.
- **Expert v3** (`dexvision/policies/pick_expert.py`): declared observations
  only. It uses the opposition pinch, or a flat grasp for objects ≤ 2 cm tall,
  which was added after v2 proved fragile on pucks under noise. Tuned on train
  seeds only.
- **Dataset** (`configs/data/pick.yaml`, `datasets/pick/pick/`, untracked): v3
  on train seeds 0–999 with DART noise (2 mm and 0.01 rad per step on the hand,
  0.05 rad on targets); clean labels; commit 308cf7c. 985/1000 kept [97.5%,
  99.1%], 83,698 transitions, every object ≥ 92% lower bound. 200/200
  episodes replay bit for bit.

## Baselines and gates

| Skill | Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|---|
| pick | val | zero | 0 / 100 | [0.0%, 3.7%] | 100 timeouts |
| pick | val | random | 0 / 100 | [0.0%, 3.7%] | 99 timeouts, 1 knocked off the table |
| pick | val | expert v3 | 200 / 200 | [98.1%, 100%] | v2 (signed off in R3) was 199/200. Non-puck episodes are identical to v2 |
| pick | val | expert v3 + DART noise | 197 / 200 | [95.7%, 99.5%] | Robustness measurement, not a policy result. 2 knocked off the table, 1 timeout |

**Pick gate (owner approved 2026-10-20, written before any training).** A
learned pick policy passes only if its single 200-episode test run has a
Wilson 95% lower bound ≥ 80% overall (≥ 170/200) and every object ≥ 60%.
Checkpoints are selected on val only. MLP-BC is reported against the same gate.

## Test ledger and caveats

- `results/test_ledger.jsonl` does not exist yet: no test-split runs so far.
- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` observations are simulator ground truth: privileged, state track
  only.
- Pick v1 is one isolated object in a small start region. Harder variants come
  after a learned policy passes.
