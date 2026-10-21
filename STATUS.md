# Status

**Active checkpoint:** R7 — reach (R2–R6 repeated; see `docs/PLAN.md`)
**State:** complete. Both methods pass the reach gate. Awaiting owner sign-off
before push (next skill in the plan).
**Last completed:** R6 — pick trained and tested; both methods pass (owner approved 2026-10-21)
**Legacy pipeline:** git tag `legacy-v1`

## Shared protocol (every skill)

- **Method:** `configs/learning/method.yaml`, unchanged across skills.
  Diffusion and MLP-BC, seed 0, 60k steps, checkpoints every 10k, trained on
  the desktop RTX 5070 Ti.
- **Selection:** the most val successes on the first 200 val seeds, ties to
  the later step. Then one ledgered 200-episode test run per method.
- **Platform:** all results are evaluated on the desktop (Linux x86-64),
  because MuJoCo is not bit-identical across arm64 and x86-64. Provenance
  records the platform.

## Pick (done, R6)

- **Task:** one of six objects in an 8 × 12 cm region. Lift ≥ 5 cm, held by
  thumb plus finger, for 0.5 s. Expert v3 (flat grasp for pucks). Data:
  985/1000 noisy expert episodes kept.
- **Results:**
  - val: zero 0/100, random 0/100, expert 200/200, expert + noise 197/200;
  - MLP-BC val 200/200, test **200/200**; Diffusion val 197/200, test
    **200/200**. Both pass the gate.
- **Diagnostics on val (not committed):**
  - With the object pose hidden, success drops to 0/200 (MLP) and 12/200
    (diffusion).
  - Placements up to 8 cm outside the training region mostly still succeed.
    Far forward, even the expert fails.
- **Ledger:** 1 of 3 test runs used per method. The MLP entry says `dirty`
  only because of uncommitted `results/` files; that flag now ignores
  `results/`.

## Reach (`configs/tasks/reach.yaml`, `dexvision/policies/reach_expert.py`)

- **Task:** a goal uniform in a 18 × 20 × 14 cm box above the table, ≥ 5 cm
  from the start, with no objects. Success: the **measured** grasp site
  within 1.5 cm of the goal for 0.5 s, within 5 s. Any hand contact with the
  scene fails the episode.
- **Observation:** palm pose, grasp position, hand joints and velocities,
  fingertips, and the goal (an instruction, not privileged state).
- **Expert:** proportional control of the grasp point toward the goal; holds
  the starting palm orientation. One parameter set, no tuning needed.
  - Train seeds 0–299: 300/300 clean and 300/300 with DART noise (median 26
    steps).
- **Data and training** (commit a3fbcf1): 1000/1000 noisy expert episodes kept,
  28,462 transitions, 200/200 replay exactly. Diffusion trained in 33 min,
  MLP in 2 min.
- **Diagnostic on val (not committed):** with the goal hidden (replaced by its
  training mean), success drops to 1/200 (MLP) and 2/200 (diffusion).

| Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|
| val | zero | 0 / 100 | [0.0%, 3.7%] | 100 timeouts |
| val | random | 0 / 100 | [0.0%, 3.7%] | 26 hand collisions, 74 timeouts |
| val | expert | 200 / 200 | [98.1%, 100%] | oracle ceiling |
| val | expert + DART noise | 200 / 200 | [98.1%, 100%] | robustness, not a policy |
| val | MLP-BC, step 60k | 200 / 200 | [98.1%, 100%] | every checkpoint 200/200 |
| val | Diffusion, step 60k | 200 / 200 | [98.1%, 100%] | every checkpoint 200/200 |
| **test** | **MLP-BC, step 60k** | **200 / 200** | **[98.1%, 100%]** | Gate: **pass** |
| **test** | **Diffusion, step 60k** | **200 / 200** | **[98.1%, 100%]** | Gate: **pass** |

**Reach gate (owner approved 2026-10-21, written before any reach training).** A learned reach policy passes only if its single 200-episode
test run has a Wilson 95% lower bound ≥ 80% (≥ 170/200). Hand collisions
count as failures. Checkpoints are selected on val only.

## Caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- Pick's `object_*` observations are privileged simulator state. Reach has
  none.
- Reach is easy: free space and a smooth single-strategy expert. It checks
  that the pipeline transfers unchanged to a second skill.
