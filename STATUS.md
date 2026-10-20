# Status

**Active checkpoint:** R1 — Honest simulator core (see `docs/PLAN.md`)
**State:** implemented and tests pass. Awaiting owner sign-off before R2.
**Last completed:** R0 — Harness reset (owner approved 2026-10-20)
**Legacy pipeline:** git tag `legacy-v1`

## R1 measurements

Source: `python -m dexvision.apps.sim_report` (full data in `results/sim_report.json`).

| Quantity | Value |
|---|---|
| Hand tracking error at rest | 5.1 mm, 0.85° |
| Tracking error moving at 0.2 m/s | 6.1 mm mean, 7.3 mm max |
| Hand-on-table force at the 1 cm lead limit | 43 N |
| Hand mass | 3.79 kg; only the 3.0 kg forearm is gravity-compensated |

The tracking error is the sag and lag of the physical hand. Observations
report it as measured, so it is no longer hidden.

**Grasp feasibility probe.** This was not a controller: one hand-picked
open-loop posture, with no physics changes. It lifted and held for 2 s:
- block_small: 11.5 cm
- cylinder_short: 9.3 cm
- block_large: 9.7 cm

In each case the thumb opposed the fingers. The flat puck was not lifted by
this crude posture; that is R3's problem, not a physics change.

## Baselines and gates

None yet. Pick baselines arrive in R2. The pick gate is written here in R3,
before any training.

| Skill | Split | Policy | Successes / n | Wilson 95% CI |
|---|---|---|---|---|

## Test ledger

`results/test_ledger.jsonl`. It does not exist yet, so no test-split runs
have happened.

## Open caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in), with no
  arm kinematics or reach limits beyond the workspace box.
- Object state in observations is simulator ground truth. That is privileged,
  and allowed only in the state track.
