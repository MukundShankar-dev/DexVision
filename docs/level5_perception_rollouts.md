# Level 5.7: inferred-control boundary and observability preflight

This document preserves the earlier v1 preflight and its then-current outcome.
The active instrumented amendment is in `level5_perception_integration.md`.
Do not rerun these historical commands with the new active recipe.

October 16, 2026. **Incomplete.** Last Completed remains 5.6; Next Target
remains 5.7. The owner authorized this checkpoint after correcting the stale
5.6 completion-checklist entry. No 5.8 work is included.

## Evidence reviewed before implementation

The 5.5 progress section, contact results, push repair history, acceptance
receipts and verification explain why substituting a pose tensor is insufficient.
Push V1 failed generalization/safety; V5/V6 changed representation and causal
completion state; V7 passed success but failed workspace safety. Accepted V8
keeps V7 weights with explicitly engineered entry encoding, physical settling
holds and workspace bounds. Its selected epochs are 200/61/20, with confirmation
116/128, 116/128 and 120/128. Press V1 remains separately accepted at 120/120 per
seed. These complete controllers, physical predicates and cadence must remain
intact. New inferred poses cannot replace their required physical feedback.

The 5.6 progress/contract, perception results, repair/development history,
V29 verification and owner acceptance preserve the failed and consumed attempts.
V29 retains V25's exact current-RGB learned weights and calibrated decoder,
adding cuboid body-center association and uncertain replacement-ID rejection.
Fresh V8 passes the visual gates with 100% primary recall, 0.954 mm median and
3.675 mm p95 position error, and zero primary ID switches. That visual acceptance
explicitly leaves all five inferred-control tracks unsupported. Semantic object
identity, full orientation/motion, robot telemetry, contact/retention/phase and
fixture depth are not supplied by this baseline. Place V25 also copies live
MuJoCo state into its accepted collision preview; retaining that preview would
leak truth even if neural pose inputs were replaced.

The reach repair and training guidance were also reviewed: preserve selected
weights plus exact transforms, scales, decoder, causal state and control cadence;
keep confirmation isolated; audit measured body origins; never substitute an
offline-loss selection or a truth-filled state for qualified control.

## Implemented behavior

`PerceptionPolicy` accepts only calibrated RGB and acquisition/delivery times.
It resets the original V29 tracker between independent requests, invokes the
unchanged baseline and records the shared typed estimate. Original 0.8 confidence
and 0.1-second age limits remain. Wall-clock inference latency is added to
delivery time before checking age. Invalid, ambiguous, uncertified, low-confidence
and stale fields stay explicit. There is no task, simulator, default-state,
semantic-id lookup, sensor surrogate or truth callback in this interface.

This adapter is **rejection-only** because the accepted compatibility inventory
blocks every skill. Even a typed schema cannot certify an absent producer or
remove the privileged place preview. No controller is invoked and no action is
produced. This is the implemented portion of 5.7, not a completed rollout engine.

The evaluator verifies V29's original recipe/model pins and all 15 selected
state checkpoints against owner receipts. Exact receipt-bound recipes, overlays,
selection/physical evidence and adjacent runtime sidecars are indexed; checkpoint
schemas are loaded for audit only. Accepted state evaluation/training/confirmation
entry points are never called by this path. Historical confirmation rates are
not substituted for a matched new state baseline.

Before RGB inference, the separately locked protocol and complete request matrix
are saved and reserved at a fixed protocol-owned path. New output paths cannot
repeat a reserved run. The new workcell resets 57071000–57071029 are validation
neutral scenes, before any task entry, contact or held-object setup. Thirty
unique scenes are shared by all five skill requests and three selected seeds:
450 requests, not 450 independent resets. Calibration is checked exactly against
the accepted pixel-center camera. Only rendered RGB/time cross the adapter;
no task-derived goal, hidden phase or physical truth is passed into inference.

Every request saves its typed fields, rejection, selected checkpoint identity,
latency, zero actions and unchanged qpos/qvel/ctrl/mocap/time check. Every scene
is saved as PNG. Five display-only panels project visual hypotheses and show
the terminal rejection. They do not assign semantic target identities. Reports
show per-skill observability failure, missing paired execution/metrics as null,
and zero qualifying rollouts. There is no claimed physical safety pass from
having executed zero actions. Noise/bias/dropout/control-latency experiments
remain deferred because no policy-visible control features are supported.

## Commands and results

All commands run from the repository root in the `dexvision` environment:

```bash
conda run --no-capture-output -n dexvision pytest -q tests/test_perception_policy_integration.py tests/test_perception_grounded_rollout.py
conda run --no-capture-output -n dexvision pytest -q tests/test_perception_policy_integration.py tests/test_perception_grounded_rollout.py tests/test_level5_learning_plan.py
conda run --no-capture-output -n dexvision ruff check dexvision tests
conda run --no-capture-output -n dexvision pytest -q
git diff --check
```

Focused tests: 19 passed; including original/independent config locks: 29 passed.
Repository Ruff and whitespace checks pass. Full regression: **1,210 passed,
one skipped in 786.62 seconds**, exit code 0. Exact bindings and checks are
recorded in `level5_perception_rollouts_verification_v1.json`; the full log is
`outputs/level5/perception_rollouts_checks/full_pytest_final.log`.

The checkpoint's listed evaluation command was run:

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_skill --config configs/level5/evaluation_perception.yaml --all-required-skills --track perception --output-dir outputs/level5/perception_rollouts
```

It completes the preflight with **450/450 rejected before execution**, 90 per
skill, zero policy calls/actions and all simulator-unchanged checks passing.
Each skill remains `failed` for missing observable inputs. Median inference plus
boundary latency is 105–112 ms and p95 141–161 ms under concurrent regression
load; all 450 exceed the accepted 34 ms cadence. This is a measured local
diagnostic, not an isolated hardware benchmark or closed-loop timing pass.

The initial sandboxed headless render failed with `invalid CoreGraphics
connection` before producing requests. Its protocol, matrix, bundle inventory
and failure receipt are preserved under
`outputs/level5/perception_rollouts_startup_failure_v1/`, with the original
reservation unchanged. A standalone headless renderer smoke passed outside
the sandbox. Protocol run revision 2 then completed with that macOS access.
Review caught a CLI dispatch bug: the new perception-only options had to be
removed before calling legacy state evaluators. Explicit `--split test` is now
also rejected for this validation-only protocol, while state commands retain
their original default test split. A regression test preserves the old state
call signature. The existing lock test now registers the separate new recipe
without changing the historical 5.0 lock. Earlier partial full-suite runs were
stopped for these repairs; the lock failure and repair record are preserved in
`outputs/level5/perception_rollouts_checks/startup_and_dispatch_repairs.json`.

Successful revision 2 data remain unchanged in
`outputs/level5/perception_rollouts_validation_v2/`, with exact original sources
in `outputs/level5/perception_rollouts_v2_source_snapshot/`. Revision 3 binds the
final CLI/review fixes and repeats the same 30 validation scenes; it is not new
independent evidence. The final result uses that revision alone. These are
disclosed validation repairs, not held-out retries. Frozen
5.6 releases, consumed tests, working data and historical receipts are unchanged.

The successful protocol is reserved. Do not rerun it into another output path
or delete its reservation. Inspect saved results or use the review below.

## Manual rejection review (pending)

```bash
conda activate dexvision
python -m dexvision.apps.evaluate_skill --config configs/level5/evaluation_perception.yaml --all-required-skills --track perception --viewer --review-dir outputs/level5/perception_rollouts
```

The OpenCV viewer verifies saved artifact hashes before displaying all five
panels. Press any key to advance; Escape cancels. It never writes acceptance.
The static panels can also be opened directly from the output directory.

Pass this limited review when all five panels and corresponding request JSON
are readable and consistent: projected markers are explicitly visual hypotheses,
no semantic target/action is asserted, unavailable fields are preserved, and
every request is `rejected_before_execution` with zero policy calls/actions.
Fail for missing/corrupt media, mismatched report/overlay, hidden truth filling
or admission of any unsupported controller. Owner confirmation remains required.

**This review cannot complete 5.7.** Its required held-out rollout per skill,
overlay/action/terminal review, 30+ qualifying rollouts, matched state baseline,
worst-family/degradation/safety gates and robustness diagnostics remain blocked.
They require an explicit versioned integration/producer amendment and evidence
for the accepted fields and place preview; missing inputs must never be filled
with simulator truth. Keep 5.7 unchecked and stop before 5.8.

## Changed files

```text
dexvision/learning/perception_policy.py
dexvision/evaluation/perception_rollouts.py
dexvision/evaluation/evaluate_skill.py
dexvision/apps/evaluate_skill.py
configs/level5/evaluation_perception.yaml
configs/level5/evaluation_perception.sha256
tests/test_perception_policy_integration.py
tests/test_perception_grounded_rollout.py
tests/test_level5_learning_plan.py
docs/CURRENT_STATUS.md
docs/progress_level_5.md
docs/module_contracts.md
docs/level5_perception_rollouts.md
docs/level5_perception_rollouts_verification_v1.json
```

Completed 5.5/5.6 method/result sections and historical receipts are preserved.
The only completed-checkpoint checklist change is the owner-authorized stale
5.6 completion entry. Models, data payloads and existing releases are unchanged.
