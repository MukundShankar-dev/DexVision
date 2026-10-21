# Status

**Active checkpoint:** R6 — train and evaluate pick (see `docs/PLAN.md`)
**State:** in progress. Training runs on the desktop (RTX 5070 Ti).
**Last completed:** R5 — Diffusion Policy and MLP-BC, method only (owner approved 2026-10-21)
**Legacy pipeline:** git tag `legacy-v1`

## R6 protocol (fixed before any pick training)

1. Train diffusion and MLP-BC with seed 0 on `datasets/pick/pick`, using
   `configs/learning/method.yaml` unchanged. Checkpoints every 10k steps.
2. Selection: every checkpoint runs on val seeds 100000–100199. The most
   successes wins; ties go to the later step
   (`python -m dexvision.apps.select_checkpoint`).
3. One test run per method on the selected checkpoint, 200 episodes,
   ledgered. It is judged against the pick gate below.
4. All evaluations (zero, random and expert baselines re-run too) are on the
   desktop (Linux x86-64), so the table compares like with like.

## Method (`dexvision/learning/`, `configs/learning/method.yaml`)

- **Diffusion Policy:** conv UNet with FiLM, 17.7M params, DDPM/DDIM,
  observations 2 / predict 16 / execute 8, 60k steps. MLP-BC uses the same
  data, normalization, training settings and wrapper.
- **Machines:** training on the desktop takes 34 ms/step, 0.6 h per run.
  MuJoCo is not bit-identical across arm64 and x86-64, so provenance records
  the platform.

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
