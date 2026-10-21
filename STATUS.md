# Status

**Active checkpoint:** R4 — Pick data generation, after re-opening R3 to harden
the expert (see `docs/PLAN.md`)
**State:** implemented, tests pass. Awaiting owner sign-off on the hardened
expert (videos) and the dataset before R5.
**Last completed:** R3 — Pick scripted expert and gate (owner approved 2026-10-20)
**Legacy pipeline:** git tag `legacy-v1`

## Pick task and expert

- **Task** (`configs/tasks/pick.yaml`): one of six objects in an 8 × 12 cm
  region, any yaw. Success (measured): lifted ≥ 5 cm and held by thumb plus
  finger, touching nothing else, for 0.5 s within 8 s.
- **Expert** (`dexvision/policies/pick_expert.py`, `configs/experts/pick.yaml`):
  sees only the declared observation (test-enforced). It chooses a grasp from
  object height alone: the opposition pinch by default; for objects ≤ 2 cm
  tall (pucks), a flat grasp anchored to the near edge and the table, with a
  thumb fence, a finger rake and an overdriven squeeze.
- **Why the flat grasp exists.** The first R4 dataset showed the R3 expert
  was fragile on pucks under DART noise (light puck kept 73/177). The default
  pinch cleared the rim by millimetres, and its squeeze was sensitive to
  target noise.
- **Tuning.** Train seeds only: v1 fist 8/60; v2 pinch 300/300 clean; v3
  (flat grasp) 300/300 clean and 293/300 with noise on seeds 0–299. Val was
  never used for tuning.

## Pick dataset (`configs/data/pick.yaml`, `datasets/pick/pick/`, untracked)

- **Generation.** Expert v3 on train seeds 0–999 with DART noise (σ: 2 mm
  and 0.01 rad per step on the hand, 0.05 rad on actuator targets). Labels
  are the clean expert actions. Commit 308cf7c, clean tree.
- **Kept.** 985/1000 [97.5%, 99.1%]: 83,698 transitions, median 82 steps.
  Every object ≥ 92% lower bound (puck_light 175/177, puck_heavy 156/162).
- **Replay.** 200/200 episodes reproduce the outcome and every observation
  bit for bit.

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
