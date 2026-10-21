# Status

**Active checkpoint:** R4 — Pick data generation (see `docs/PLAN.md`)
**State:** implemented, tests pass. Awaiting owner decision on the light-puck
finding below before R5.
**Last completed:** R3 — Pick scripted expert and gate (owner approved 2026-10-20)
**Legacy pipeline:** git tag `legacy-v1`

## Pick task and expert

- **Task** (`configs/tasks/pick.yaml`): one of six objects in an 8 × 12 cm
  region, any yaw. Success (measured): lifted ≥ 5 cm and held by thumb plus
  finger, touching nothing else, for 0.5 s within 8 s.
- **Expert** (`dexvision/policies/pick_expert.py`): sees only the declared
  observation (test-enforced), with one parameter set for all objects, using
  an opposition pinch. Tuned on train seeds only (v1 fist 8/60; v2 pinch
  300/300).

## Pick dataset (`configs/data/pick.yaml`, `datasets/pick/pick/`, untracked)

- **Generation.** Expert on train seeds 0–999 with DART noise (σ: 2 mm and
  0.01 rad per step on the hand, 0.05 rad on finger targets). Labels are the
  clean expert actions. Commit 2771fca, clean tree.
- **Kept.** 869/1000 successes, 95% CI [84.7%, 88.8%]: 75,721 transitions,
  median 85 steps. Dropped: 129 timeouts, 2 objects knocked off the table.
- **Replay.** 200/200 replayed episodes reproduce the outcome and every
  observation bit for bit.
- **Finding: the expert is fragile on pucks.** Kept rates: puck_light 73/177
  [34%, 49%], puck_heavy 142/162 [82%, 92%], every other object ≥ 95% lower
  bound. Noise leaves the first close badly seated, so the slip check backs
  off, and the second attempt often runs out of horizon. The clean expert
  hides this (val 199/200). puck_light is under-represented in the data, and
  a learned policy's own errors act like this noise.

## Baselines and gates

| Skill | Split | Policy | Successes / n | Wilson 95% CI | Notes |
|---|---|---|---|---|---|
| pick | val | zero | 0 / 100 | [0.0%, 3.7%] | 100 timeouts |
| pick | val | random | 0 / 100 | [0.0%, 3.7%] | 99 timeouts, 1 knocked off the table |
| pick | val | expert | 199 / 200 | [97.2%, 99.9%] | Every object ≥ 86% lower bound. The 1 failure is a light puck that slipped after 3.5 cm (seed 100183) |

**Pick gate (owner approved 2026-10-20, written before any training).**
A learned pick policy passes only if its single test run of 200 episodes
meets both conditions:
1. the Wilson 95% lower bound on overall success is at least 80%, which
   needs at least 170/200;
2. every object's success rate is at least 60%.

Checkpoint selection uses val success only. The MLP-BC baseline is reported
against the same gate.

## Test ledger

`results/test_ledger.jsonl` does not exist yet: no test-split runs so far.

## Open caveats

- The hand is a floating forearm on a mocap weld (an arm stand-in).
- `object_*` observations are simulator ground truth: privileged, state
  track only.
- Pick v1 is easy for a privileged expert: one isolated object and a small
  start region. Harder variants (clutter, a wider region) can come after a
  learned policy passes.
