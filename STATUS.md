# Status

**Active checkpoint:** R6 — train and evaluate pick (see `docs/PLAN.md`)
**State:** complete. Both methods pass the pick gate. Awaiting owner sign-off.
**Last completed:** R5 — Diffusion Policy and MLP-BC, method only (owner approved 2026-10-21)
**Legacy pipeline:** git tag `legacy-v1`

## R6 protocol (fixed before training) and method

- Diffusion and MLP-BC trained with seed 0 on `datasets/pick/pick` with
  `configs/learning/method.yaml` unchanged: 60k steps, checkpoints every 10k
  (desktop RTX 5070 Ti: diffusion 31 min, MLP 3 min).
- Selection: the most val successes on seeds 100000–100199, ties to the later
  step. Then one ledgered 200-episode test run per method.
- Everything was evaluated on the desktop, because MuJoCo is not bit-identical
  across arm64 and x86-64.

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

## Pick results (desktop, Linux x86-64; all val runs reproduce episode for episode)

| Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|
| val | zero | 0 / 100 | [0.0%, 3.7%] | |
| val | random | 0 / 100 | [0.0%, 3.7%] | |
| val | expert v3 | 200 / 200 | [98.1%, 100%] | oracle ceiling |
| val | expert v3 + DART noise | 197 / 200 | [95.7%, 99.5%] | robustness, not a policy |
| val | MLP-BC, step 60k | 200 / 200 | [98.1%, 100%] | every checkpoint 10k–60k: 200/200 |
| val | Diffusion, step 60k | 197 / 200 | [95.7%, 99.5%] | 186→195→194→197→197→197; 3 puck timeouts |
| **test** | **MLP-BC, step 60k** | **200 / 200** | **[98.1%, 100%]** | every object 100%. Gate: **pass** |
| **test** | **Diffusion, step 60k** | **200 / 200** | **[98.1%, 100%]** | every object 100%. Gate: **pass** |

**Pick gate (owner approved 2026-10-20, written before any training).** A
single 200-episode test run needs a Wilson lower bound ≥ 80% (≥ 170/200) and
every object ≥ 60%. Checkpoints are selected on val only.

## Test ledger and caveats

- `results/test_ledger.jsonl`: 1 of 3 runs used for each of `mlp` and
  `diffusion`. The MLP entry says `dirty: true` because re-run reports in
  `results/` were uncommitted; the code was commit 9ac82a6. The flag now
  ignores `results/`.
- Pick v1 does not separate the methods: both saturate the test. It is one
  isolated object, the policy gets the true object state, and the expert has
  a single strategy, so there are no competing ways to act for diffusion to
  capture.
- The hand is a floating forearm on a mocap weld. `object_*` observations are
  privileged simulator state.
