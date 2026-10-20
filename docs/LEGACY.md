# Legacy pipeline (`legacy-v1`)

The original DexVision pipeline (Levels 1–5.7, June–October 2026) is
preserved unchanged at git tag `legacy-v1`. That includes the uncommitted
5.7 instrumented-observation work and the old `AGENTS.md`. To inspect it:

```bash
git switch --detach legacy-v1
```

Large local data (`datasets/`, `outputs/`, `data/`) was never deleted. It is
only untracked on `main`.

## Why it was retired

An adversarial review (2026-10-20) found that the headline results did not
measure what they claimed.

**Physics cheats**
- Every 8 physics steps during grasp, lift, transport and place, the grasped
  object's orientation was overwritten and its spin zeroed
  (`preserve_free_joint_orientation`).
- Object friction was boosted, up to 4×, until the expert succeeded. Puck
  friction was raised whenever randomization sampled it low.
- The bin walls and button did not collide during pick/place.
- Start-state randomization was about 1 mm.

**Measured vs commanded**
- Robot pose and reach success were read from the mocap *target*, not the
  physical hand.
- The hand sagged about 16 mm under gravity with no compensation.

**Engineered, not learned**
- In every accepted bundle the network produced 1–3 scalars. Direction,
  posture, phase, gating and release were hand-coded decoders, different for
  each skill.
- Nothing measured the learned contribution against a no-learning baseline.

**Per-seed tuning**
- Place used per-seed floors, exclusions and `if seed == 5003` branches.

**Expert lookahead**
- Scripted experts simulated candidates in a copied `MjData` and executed the
  first one that succeeded.
- Expert parameters were tuned after test sessions.

**Statistics**
- Counts were inflated 4× by identical render-label repeats, and no
  confidence intervals were reported.
- 25 place versions and 29 perception versions iterated against "fresh"
  seeds from the same distribution.

**5.7 perception**
- The camera period (0.102 s) exceeded the 0.1 s freshness limit.
- Wall-clock time was added to simulator time.
- Together these made most reported failures observer aborts.

**Engineering**
- Source-file hash pinning forced a copy-per-version codebase: 121k lines,
  80 `place_*` modules and 9,586 pinned digests.

## Lessons worth keeping

- Closed-loop success, not offline loss, is the only meaningful selection
  metric.
- Normalizing a near-constant feature by its tiny std blows it up. Floor the
  std.
- Single-step full-action BC compounds errors. Use action chunking and
  observation history, and consider DAgger with the scripted expert.
- Coordinate frame and sign errors silently break generalization. Test
  frames explicitly.
- Fail-closed boundaries and recording negative results are good habits.
  Keep them, without the ceremony.
