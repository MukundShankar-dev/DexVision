# Level 5.2 — State-Grounded Reach Results

Completed September 14, 2026. Automated implementation and evaluation checks
pass; all six trained policies fail the frozen qualification gates.

## Scope and method

The clean starting branch was `main`; CURRENT_STATUS and the active Level 5
progress file agreed that 5.1 was complete and 5.2 was next. Only reach/state is
implemented. The frozen 5.0 plan, all YAML/checksum locks, released datasets and
completed progress sections are preserved.

Both frozen models use the same 80 training and 32 validation episodes,
training-only normalization, 128/128 ReLU trunk, Adam settings, cell/episode/frame
sampling and seeds 5001/5002/5003. The baseline predicts bounded deltas relative
to its previous applied action. The reference predicts absolute full actions.
Neither uses an expert during execution. Early stopping and candidate selection
follow the unchanged frozen rules. Every seed is retained.

All candidate checkpoints receive the complete validation matrix. Only its
safety, success, worst-cell success, normalized terminal error, offline loss and
epoch select the checkpoint. Each selected checkpoint then receives exactly one
120-attempt state test: 30 distinct physical resets, each independently run
under four render-condition labels. Those labels have no effect on physics or
state observations. The repetitions are not 120 independent reset samples and
make no perception claim.

Reset metadata is verified against the release. Reach starts at frame zero, so
no test action arrays are read. Tests verify that live policy features equal
the causal training features across replayed actions. Terminal dwell is reset
at policy start. Rejected requests and safety failures stay in the denominator
and retain their full available trajectory and state evidence.

## Results

**Checkpoint complete with a negative policy-qualification result.** All six
selected policies failed. Validation-selected success and held-out success are
zero for every seed. No policy is state-qualified or enabled for execution.

| Model | Seed | Training epochs | Selected epoch | Test successes | Workspace violating values | Joint-limit violating values | Mean jerk | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 5001 | 30 | 10 | 0 / 120 | 0 | 120 | 0.194800 | failed |
| baseline | 5002 | 23 | 23 | 0 / 120 | 8 | 112 | 0.151500 | failed |
| baseline | 5003 | 23 | 3 | 0 / 120 | 8 | 124 | 0.244481 | failed |
| reference | 5001 | 100 | 100 | 0 / 120 | 284 | 52 | 0.000000 | failed |
| reference | 5002 | 23 | 23 | 0 / 120 | 340 | 56 | 0.000000 | failed |
| reference | 5003 | 26 | 26 | 0 / 120 | 248 | 0 | 0.000000 | failed |

The 29 candidates produced **3,480 validation attempts**. The six selected
checkpoints produced **720 held-out attempts**, all saved with explicit terminal
results. Per-session/object/goal/boundary/condition success is also zero, so
aggregate reporting conceals no successful subgroup. Invalid-action and
object-workspace counts are zero for every test run; workspace and actuator
limit failures still disqualify every policy. Multiple scalar violations can
occur in one attempt, so violation counts can exceed 120.

Baseline jerk exceeds the frozen 0.0238697379352 gate for every seed. Reference
jerk is zero because these policies terminate at their first action; it is not
evidence of useful smooth control. Bounded per-step baseline deltas do not
prevent accumulated absolute actuator/workspace violations. This is a measured
failure of the frozen baseline, not a reason to tune from held-out outcomes.
No post-test model, protocol, data or threshold change was made.

### Direct Level 3 comparison

Level 3.4 recorded 7 workspace and 14 joint-limit violating values across 35
attempts. Its corrected validation-selected v2 checkpoint recorded 20 and 15,
with zero success. The table above retains the same two failure categories and
scalar-count convention. Baseline 5001 eliminates workspace violations in its
120 attempts, but all attempts still terminate on actuator limits. The other
baseline seeds retain both types; reference failures are predominantly workspace
violations. Different goals and reset matrices prevent a paired numerical
transfer claim. None of these results qualifies a skill.

### Automated acceptance and manual status

- Focused reach tests: **19 passed** (including two final added guard tests).
- Focused reach plus documentation regressions: **31 passed**.
- Full suite: **749 passed, 1 skipped in 606.78 seconds**. This run collected
  the original 17 new reach tests; the final two additions passed in the focused
  run. No implementation changed after that full-suite run began.
- Repository-wide Ruff and whitespace checks pass.
- Independent saved-artifact readback passes all **720** test trajectories,
  checksums, validation-only selection rankings, matrix membership, finite
  states/applied actions, quaternion norms, action history, and recomputed jerk
  and summary metrics. Physics was not rerun. The receipt is
  `outputs/level5/reach_52_comparison/artifact_audit.json`.

**Manual verification: not required.** No policy meets the numerical gate that
would trigger viewer acceptance. No user approval, human visual observation or
state qualification is invented. Checkpoint 5.2 is complete as an evaluated
negative result; 5.3 remains planned and was not started.

### Evidence

- Comparison: `outputs/level5/reach_52_comparison/comparison.json`
- Comparison SHA-256: `a7a836660e19c1df2dbfca4a0745e0d34c4774408486ff0cdeae37328fd8fb87`
- Offline curves: `outputs/level5/reach_52_comparison/training_curves.svg`
- Command/test logs: `outputs/level5/reach_52_logs/`

![Offline training and validation curves with held-out success](../outputs/level5/reach_52_comparison/training_curves.svg)


## Reproduction commands

Run from the repository root. These are the experiment commands used, with
output redirection omitted. Existing run/evaluation directories are refused;
retain the original artifacts and choose a new run id for a separate complete
reproduction. Do not rerun a selected checkpoint's held-out test for tuning.

```bash
conda activate dexvision
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for model in baseline reference; do
  for seed in 5001 5002 5003; do
    run="outputs/level5/level5-evaluation-v1/reach_object/$model/seed_$seed/run_52_v1"
    python -m dexvision.apps.train_skill \
      --config configs/level5/skills/reach_object.yaml \
      --model "$model" --seed "$seed" --output-dir "$run"
    python -m dexvision.apps.evaluate_skill \
      --config configs/level5/skills/reach_object.yaml \
      --select-candidates "$run" --output-dir "$run/validation" \
      --release-root outputs/level4/manual_release_v1/restored
    python -m dexvision.apps.evaluate_skill \
      --config configs/level5/skills/reach_object.yaml \
      --checkpoint "$run/selected.pt" --track state --output-dir "$run/state" \
      --release-root outputs/level4/manual_release_v1/restored
  done
done

python -m dexvision.apps.evaluate_skill \
  --config configs/level5/skills/reach_object.yaml \
  --compare-runs \
    outputs/level5/level5-evaluation-v1/reach_object/baseline/seed_5001/run_52_v1 \
    outputs/level5/level5-evaluation-v1/reach_object/baseline/seed_5002/run_52_v1 \
    outputs/level5/level5-evaluation-v1/reach_object/baseline/seed_5003/run_52_v1 \
    outputs/level5/level5-evaluation-v1/reach_object/reference/seed_5001/run_52_v1 \
    outputs/level5/level5-evaluation-v1/reach_object/reference/seed_5002/run_52_v1 \
    outputs/level5/level5-evaluation-v1/reach_object/reference/seed_5003/run_52_v1 \
  --output-dir outputs/level5/reach_52_comparison

conda run -n dexvision pytest -q tests/test_reach_skill_training.py tests/test_reach_skill_rollout.py
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest -q
git diff --check
```

The optional `--release-root` points at the existing verified restore. Every
requested file is still checked against its immutable release hash. Omit that
argument to read the archive directly. Training in this experiment reads the
immutable archive. CLI entry points also run on Windows; shell-loop/export
syntax above is for the owner's macOS shell. No webcam, GUI or GPU is used.

## Artifacts and limitations

Working outputs remain ignored under `outputs/level5/`. Each training run saves
normalization, manifests, environment/config snapshots, all candidates, selected
weights, validation selection evidence and checksums. Every evaluation saves
`provenance.json`, `report.json`, and per-attempt `trajectory.npz`,
`world_states.json`, `result.json`, and checksums. Test reservations prevent
repeating a selected run under a different output path. Interrupted attempts
remain retained; the evaluator does not silently restart them.

The comparison includes complete per-session, entity, goal, boundary, condition
and seed breakdowns, every candidate's validation metrics, and offline curves.
Its historical comparison uses the pinned Level 3 v2 checkpoint digest
`d30f3122b50047693426aa7eb3e52616e7c0d1a8b3174370703ee313aade5356`.
That checkpoint's contact-only xyz goal and tracking-quality/untyped-object
observation contract are incompatible with the Level 4 approach-pose contract.
There are zero directly compatible scenarios. No legacy policy was silently
adapted or claimed to transfer; the new reference is a separately trained model
from the frozen reference family.

The existing reach metric reads `WorldState.robot.base_position`, the commanded
mocap base target. Policy proprioception instead reads the measured free joint.
Physical object displacement supplies the disturbance check. Actuator-limit
counts measure unsafe requested target values, consistent with Level 3; they do
not certify zero physical soft-joint overshoot. This checkpoint preserves those
frozen semantics and does not claim broader physical qualification.

The first action has no prior safety mask; subsequent observations use the
preceding applied/commanded action and safety record. Every invalid action
receives a finite fallback; workspace/actuator-limit requests are clipped and
terminate as failures. The supervisor also reports wrong-entity contact and
objects leaving the board. No clipping, short-trajectory jerk, rejected request,
or pooled seed average can turn a failed run into a qualified policy.
