# Pick diagnosis and repair (5.3A)

In progress September 17, 2026 at the owner's request. The completed 5.3 negative
result, v1/v2 recipes, all original runs, releases and 5.0 locks are preserved.
5.4 is not started. Pick remains unqualified until all three seeds pass the
unchanged numerical gates and the owner accepts manual review.

## Diagnosis using training and validation

The 18,401 training frames have 13 varying finger commands, but their delta
matrix is rank one: first singular value 14.3868690, second 3.35e-14. All closure
commands are the same scalar interpolation between the open and closed posture.
V2 instead predicts independent increments; small errors can accumulate outside
the demonstrated posture or actuator bounds. V3 learns one signed closure
increment, with an explicit geometric decoder for the audited synergy. Targets
must reconstruct exactly, and off-synergy training or prior commands are rejected.
The learned increment is limited to 0.05 per control sample, and the decoder
bounds remaining aperture to [0,1]. Zero output holds; negative output can reopen.
Lift and all three orientation outputs remain learned. This is not full-hand
independent joint learning.

A diagnostic projection of the existing selected v2 models onto the closure
synergy did not solve validation: nominal successes were 29/30, 27/30 and 28/30
with remaining drops and one board escape. This diagnostic is neither retrained
nor qualified. Its 90 trajectories remain in
`outputs/level5/pick_repair/probe_synergy/`.

Source inspection also found an important provenance caveat: Level 4 recording
and replay overwrite the selected object's rotation and angular velocity during
lift/transport. Learned rollouts do not receive this aid. A validation-only
oracle ablation replays each saved expert action slice with ordinary physics:
all 30 nominal resets pass. Thus the missing hold is a source-distribution
limitation, but does not alone explain these validation failures. The saved
continuations are diagnostic only and never enter policy inference or fresh
confirmation. Evidence is retained in
`outputs/level5/pick_repair/source_without_hold/`.

## V3 frozen experiment

`configs/level5/pick_policy_v3.yaml` and its independent checksum freeze the
closure repair before training. The 121 causal inputs, fixed scales, 128x128
ReLU MLP, Adam, balanced sampling, three seeds and 200-epoch budget otherwise
remain unchanged. Epochs 20, 80, 200 and the best offline checkpoint all receive
the full original 120-attempt validation matrix. Selection uses the original
safety-first rule. V3 never reruns the exposed original test.

Before selection, freeze 54 fresh resets (three for each original test cell)
with seeds starting 530260900 and the original procedural sampler. These use a
frozen source approach prefix strictly before acquisition; source continuation
is excluded. New size/mass/friction/pose variations are fixed before evaluating
a policy. All entries stay in the denominator, including any rejected setup.
The original test families and instances are already exposed: these are fresh
resets in exposed families, not unseen-family evidence. All three selected
models must first pass validation before one-time fresh confirmation; an
exclusive reservation prevents retries under a different output directory.

## Commands

```bash
conda activate dexvision
python -m dexvision.apps.repair_pick freeze-confirmation --config configs/level5/pick_policy_v3.yaml --output-dir outputs/level5/pick_repair/v3/confirmation_resets --release-root .
python -m dexvision.apps.train_skill --config configs/level5/pick_policy_v3.yaml --seed 5001 --release-root .
python -m dexvision.apps.evaluate_skill --config configs/level5/pick_policy_v3.yaml --select-candidates outputs/level5/pick_repair/v3/seed_5001 --track state --release-root . --output-dir outputs/level5/pick_repair/v3/validation_5001
python -m dexvision.apps.repair_pick confirm --config configs/level5/pick_policy_v3.yaml --run-dir outputs/level5/pick_repair/v3/seed_5001 --confirmation-dir outputs/level5/pick_repair/v3/confirmation_resets --output-dir outputs/level5/pick_repair/v3/confirmation_5001
pytest -q tests/test_pick_skill_training.py tests/test_pick_skill_rollout.py tests/test_level5_learning_plan.py
```

Run training/selection/confirmation for seeds 5001, 5002 and 5003. Training and
selection outputs are immutable run directories; the commands above reproduce
into new locations rather than overwrite existing evidence. Confirmation must
not run until all three validation selections pass. Detailed final results and
any required manual review command will be appended after evaluation.

## V3 full validation results

Every seed completed 200 epochs / 14,400 optimizer updates, and every retained
candidate received all 120 original validation attempts. No confirmation was
used in selection. This includes the failed later candidates below.

| Seed | Epoch | Success | Safety violations | Drop rate | Mean jerk | Selected |
|---|---:|---:|---:|---:|---:|---|
| 5001 | 20 | 116/120 | 0 | 0.0333 | 0.002001 |  |
| 5001 | 80 | 116/120 | 0 | 0.0333 | 0.002067 |  |
| 5001 | 186 | 120/120 | 0 | 0.0000 | 0.002142 | yes |
| 5001 | 200 | 120/120 | 0 | 0.0000 | 0.002182 |  |
| 5002 | 20 | 120/120 | 0 | 0.0000 | 0.001904 | yes |
| 5002 | 80 | 96/120 | 4 | 0.1667 | 0.002305 |  |
| 5002 | 174 | 116/120 | 0 | 0.0333 | 0.002177 |  |
| 5002 | 200 | 120/120 | 0 | 0.0000 | 0.002223 |  |
| 5003 | 20 | 120/120 | 0 | 0.0000 | 0.001985 |  |
| 5003 | 80 | 116/120 | 0 | 0.0333 | 0.002201 |  |
| 5003 | 173 | 112/120 | 4 | 0.0333 | 0.002308 |  |
| 5003 | 200 | 120/120 | 0 | 0.0000 | 0.002111 | yes |

Selected epochs 186, 20 and 200 each pass 120/120 attempts with zero safety
violations and all unchanged numerical gates. The fresh reset manifest hash is
`fb5138fda10c17616863f7569457fe960936bf51e5d687ef2ff1640e0fe069d0`.
It was frozen before any candidate selection. Fifty-three of the 54 entries
meet preconditions; the rejected entry remains in every seed's denominator.
No setup was replaced after seeing that result.

## V3 fresh confirmation — negative

The one-time reports are preserved. Every seed fails success, worst-family and
drop gates. The closure representation removes all joint-command limit
violations, but does not repair retention generalization. No manual acceptance
is requested, no policy is qualified, and 5.3A remains in progress.

| Seed | Success | Worst family | Drop rate | Safety count | Mean jerk |
|---|---:|---:|---:|---:|---:|
| 5001 | 144/216 | 0.1111 | 0.3148 | 0 | 0.002135 |
| 5002 | 144/216 | 0.1111 | 0.2963 | 4 | 0.002022 |
| 5003 | 108/216 | 0.1111 | 0.4259 | 12 | 0.002512 |

All 780 regression tests pass (one skip), all 27 focused checks pass, Ruff passes,
and original configuration locks and release readiness pass. Scientific
qualification remains failed despite these software checks. Further diagnosis
uses train/validation only; V3 confirmation is now exposed and cannot be reused
as independent evidence or silently retried.

## V4 compact geometry and phase-compatible decoder

The training-only audit independently proves three additional facts: all lift
increments are nonnegative and occur at full commanded closure; maximum command
lift is 0.08/0.10/0.11 m for cuboid/cylinder/puck respectively; puck pick rotation
increments are exactly zero. A proposed exact geometric orientation-feedback
decoder was rejected: causal entry orientation and recorded state do not exactly
reconstruct all source rotation changes. V4 therefore continues learning rotation
increments for cuboids/cylinders rather than substituting a scripted controller.

V4 uses 28 state features: closure fraction, command/object lift, command height,
calibrated tracking lag, object-relative position and linear/angular motion in
the prior-command hand frame, six relative entry-orientation error components,
support/held/contact, and three families. These omit absolute hand posture and
independent finger positions/velocities. Common world-yaw rotation leaves the
features unchanged, verified by a synthetic equivariance test. Translation and
linear velocity use fixed 0.1 m and 0.1 m/s scales; angular velocity uses 1 rad/s,
and rotation-matrix errors use 0.1. No empirical tiny divisors are fitted.

Five model outputs remain active: lift gain, three hand-frame orientation
increments, and closure gain. The decoder supplies monotone closure/lift, the
full-closure lift interlock, demonstrated family lift ceilings, hand/world frame
conversion and fixed puck orientation. These are engineered constraints, not
learned full-hand control. Closure/lift gains at their fully closed/raised bounds
are unidentifiable and masked in the loss, preventing a learned zero at the bound
from prematurely stalling just below it. Every source target must still decode
exactly. Source assistance remains absent from evaluation. The model, optimizer,
200 epochs, three seeds and candidate schedule are unchanged.

Recipe: `configs/level5/pick_policy_v4.yaml` with independent checksum. Outputs:
`outputs/level5/pick_repair/v4/`. Run the commands above with `v3` replaced by `v4`.
Fresh seeds start at 540260900. The 54-reset manifest was frozen before selection,
with hash `8c273c6870a796b39a09ed7f2b1f44ba0f0c1397f65230552da91ca99feff6ac`.
Two rejected setup entries remain in the denominator. Original and V3 evaluated
families/instances are already exposed; V4 makes no unseen-family claim.

### V4 full validation

All three seeds completed 200 epochs / 14,400 updates. Every retained candidate
received 120 original validation attempts; the selected epochs are 20, 74 and 20.
Each selected candidate passes all gates with 120 successes and zero safety
violations. Exact target reconstruction across all 24,539 train/validation frames
has maximum error 4.44e-16. All 29 focused checks pass.

| Seed | Epoch | Success | Safety violations | Drop rate | Mean jerk | Selected |
|---|---:|---:|---:|---:|---:|---|
| 5001 | 20 | 120/120 | 0 | 0.0000 | 0.001952 | yes |
| 5001 | 38 | 112/120 | 0 | 0.0667 | 0.002220 |  |
| 5001 | 80 | 120/120 | 0 | 0.0000 | 0.002088 |  |
| 5001 | 200 | 120/120 | 0 | 0.0000 | 0.002138 |  |
| 5002 | 20 | 116/120 | 0 | 0.0333 | 0.001942 |  |
| 5002 | 74 | 120/120 | 0 | 0.0000 | 0.002106 | yes |
| 5002 | 80 | 120/120 | 0 | 0.0000 | 0.002137 |  |
| 5002 | 200 | 120/120 | 0 | 0.0000 | 0.002198 |  |
| 5003 | 20 | 120/120 | 0 | 0.0000 | 0.001896 | yes |
| 5003 | 30 | 116/120 | 0 | 0.0333 | 0.001965 |  |
| 5003 | 80 | 116/120 | 0 | 0.0333 | 0.002038 |  |
| 5003 | 200 | 120/120 | 0 | 0.0000 | 0.002123 |  |

### Entry-protocol audit preregistration

After V4 negative confirmation, audit setup using validation cells only, with
fresh seeds starting 610260900. First compare one fixed cell per family, then
the full validation-cell set if the geometry check passes. Keep V4 selected
weights unchanged. Compare literal copied prefixes with a deterministic open-hand
approach to the source pre-acquisition pose expressed in the fresh object's frame.
No source action at or after acquisition may be read. Keep safety, preconditions,
physics and outcome gates unchanged. A new reset seed changes the base workcell
object pose as well as the small procedural perturbation; simply replaying an old
world-coordinate prefix may therefore miss the intended approach pose. This is
an audit hypothesis, not a reclassification of the preserved V3/V4 failures.

### Preserved V4 confirmation result

| Seed | Success | Worst family | Drop rate | Safety count | Mean jerk |
|---|---:|---:|---:|---:|---:|
| 5001 | 124/216 | 0.0556 | 0.3333 | 12 | 0.001633 |
| 5002 | 116/216 | 0.0556 | 0.3519 | 16 | 0.001745 |
| 5003 | 120/216 | 0.0556 | 0.3148 | 20 | 0.001668 |

All three fail success, worst-family, drop and safety qualification. No manual
acceptance is required for these failed reports. The complete software suite
passes 782 tests with one skip; scientific qualification remains failed.

## Fresh-approach protocol correction, fixed V4 weights

The literal-prefix confirmations changed the reset pose and procedural dynamics
but replayed world-coordinate approach actions from another reset. Validation
audit demonstrates the mismatch: cuboid wrist yaw must be conditioned on the
current object yaw; historical anchors also used an older negative-yaw-only
rule and older grasp templates. Some current procedural templates differ by
several millimetres from those anchors. These failed studies remain useful
stress evidence; they are not silently relabeled successes.

Setup protocol V1 stopped before policy execution when an anchor's recorded
wrist did not match its declared signed-yaw rule. V2 recognized the historical
rule but used an approximate family-aware waypoint generator. All three fixed
models failed its fresh validation (84/120, 84/120 and 80/120); its confirmation
was never executed. Further control-sequence comparison found a missing hold
at the first waypoint, and the reused early templates did not reproduce the
full current procedural expert setup.

`configs/level5/pick_setup_v3.yaml` preregisters a separate corrected protocol
with the exact already-selected V4 model hashes. It uses the frozen collection
expert's deterministic approach planner and shared settings resolver, including
instance/procedural overrides and copied-state grasp-template preflight. Only
open-hand approach actions reach the live task. The planner stops at the last
approach waypoint before querying or applying acquisition. No policy is queried
in setup, and no live object pose/orientation is overwritten. A regression test
reconstructs the recorded source prefix exactly.

The expert's **scratch** preflight retains the historical declared orientation
hold; this is disclosed planning assistance before skill entry. Neither the live
prefix nor the learned pick rollout has that hold. Qualification is therefore
for a state policy starting from a scripted, family-aware approach, not arbitrary
poses within a six-centimetre distance sphere, learned composition, or vision.
Original and prior confirmation families/instances are exposed.

V4 weights remain unchanged and cannot be reselected in this study. Thirty fresh
validation resets start at 670260900; 54 confirmation resets start at 680260900.
All are frozen before the new validation, without filtering or replacement.
Twenty-nine validation starts and all 54 confirmation starts meet entry checks.
All four original conditions are independently executed: 120 validation and 216
confirmation attempts per seed. Original success/drop/safety/jerk gates remain
unchanged. All three models must pass the entire new validation before a single
reserved confirmation per model is allowed.

The settings resolver moved unchanged from the recording app into
`dexvision/sim/scripted_expert_settings.py`, with compatibility imports/wrapper
retained. This lets recording and setup share the same rules without making an
evaluation module depend on CLI business logic. Datasets and completed Level 4
sections are unchanged.

```bash
python -m dexvision.apps.qualify_pick_setup freeze --config configs/level5/pick_setup_v3.yaml --output-dir outputs/level5/pick_repair/conditioned_setup_v3/resets --release-root .
python -m dexvision.apps.qualify_pick_setup validate --config configs/level5/pick_setup_v3.yaml --reset-dir outputs/level5/pick_repair/conditioned_setup_v3/resets --output-dir outputs/level5/pick_repair/conditioned_setup_v3/validation_5001 --seed 5001
python -m dexvision.apps.qualify_pick_setup confirm --config configs/level5/pick_setup_v3.yaml --reset-dir outputs/level5/pick_repair/conditioned_setup_v3/resets --output-dir outputs/level5/pick_repair/conditioned_setup_v3/confirmation_5001 --seed 5001
```

Validation and any authorized confirmation run for all seeds 5001, 5002, 5003.
These are one-time artifact directories and reservations, not retry commands.


The fixed-weight setup V3 validation completed with the rejected start retained:

| Seed | Success | Worst family | Drop rate | Mean jerk | Safety violations | Gates |
| --- | --- | --- | --- | --- | --- | --- |
| 5001 | 112/120 | 0.80 | 0.033333 | 0.002223961 | 0 | pass |
| 5002 | 108/120 | 0.70 | 0.066667 | 0.002041042 | 0 | pass |
| 5003 | 116/120 | 0.90 | 0 | 0.001837163 | 0 | pass |

These reports authorized the single reserved confirmation for each fixed model.
The viewer (`qualify_pick_setup review`) never executes another confirmation or
changes weights. It requires all three passing numerical reports and physical
audits before opening the GUI and includes examples of recorded failure reasons.


## Setup V3 confirmation — negative

All three one-time reservations finished. No weights, thresholds, resets or
setup were changed after these outcomes. The attempt directories and earlier
failures remain intact.

| Seed | Success | Worst family | Drop rate | Board violations | Mean jerk | Gates |
| --- | --- | --- | --- | --- | --- | --- |
| 5001 | 136/216 | 0.166667 | 0.314815 | 16 | 0.001659105 | fail |
| 5002 | 136/216 | 0.111111 | 0.296296 | 16 | 0.001628763 | fail |
| 5003 | 132/216 | 0.166667 | 0.314815 | 20 | 0.001499885 | fail |

All other safety counters are zero and jerk passes, but every seed fails the
unchanged aggregate-success, worst-family, drop and object-workspace gates.
The setup correction is real; it does not solve the remaining generalization
problem. Pick is **not repaired or qualified**. No saved-state qualification
audit or owner viewer acceptance is requested for these failed reports.

## Coverage finding and proposed amendment (not implemented)

A training/validation-only audit counts 144 training episodes: 48 small-block,
48 short-cylinder and 48 light-puck examples across three goal cells each.
Validation contains 16 examples for each of these same three instances. There
are no training or validation examples for large block, tall cylinder or heavy
puck. The latest confirmation's reported instance summaries show heavy-puck
retention and large-block board escape remain failures; no confirmation actions
are reused as training labels or used for another model selection.

Source trajectories additionally retain the historical orientation assistance;
there is no qualified unassisted teacher coverage across the larger/heavier
operating range. These facts support a data/expert feasibility amendment, but do
not establish that more data alone will solve the problem.

Proposed next work **within an explicitly extended 5.3A**, pending owner direction:

1. Freeze a separate training/validation-only feasibility matrix over all three
   shape families and explicit size/mass/friction/pose ranges. Preserve the old
   split and disclose every exposed family/instance and confirmation.
2. Qualify a deterministic scripted teacher under ordinary physics with no
   object-orientation/pose overwrite during acquisition, lift or retention.
   Retain infeasible cases and failures; do not manufacture success labels.
3. Only after teacher feasibility passes, collect an additive, versioned dataset
   with separate whole sessions and frozen partitions. Preserve `data/demos/`,
   existing archives, manifests and original 5.0/Level 4 locks unchanged.
4. Audit necessary geometry/contact features, freeze a new recipe/budget and
   fresh confirmation before training, then repeat all three seed gates. Keep
   owner visual acceptance mandatory after numerical qualification.

This proposal has not collected data, altered a release, trained another method,
or advanced to 5.4. `docs/level5_training_guidance.md` requires: “If fields or
coverage are absent, record the deficiency and request a versioned data/schema
amendment rather than inventing labels or editing the immutable release.”

## Final software checks for this repair attempt

All commands use the `dexvision` Conda environment. Long CPU jobs additionally
set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; PATH uses the Command Line Tools Git
because `/usr/bin/git` is blocked by the local Xcode-license state.

```bash
conda run --no-capture-output -n dexvision pytest -q
conda run --no-capture-output -n dexvision pytest -q tests/test_pick_skill_training.py tests/test_pick_skill_rollout.py tests/test_level5_learning_plan.py
conda run --no-capture-output -n dexvision ruff check dexvision tests
/Library/Developer/CommandLineTools/usr/bin/git diff --check
shasum -a 256 -c configs/level5/SHA256SUMS
```

Full suite: **786 passed, 1 skipped** (554.25 seconds). After adding the saved
review guards, all **36 focused tests passed** (18.54 seconds), including three
new guard cases. Ruff, whitespace and original configuration-lock checks pass.
Release readiness previously reverified all 33,350 payload files during this
repair; no dataset files have since changed. Software checks do not override
failed physical qualification.

No manual command is required now. The saved viewer intentionally rejects this
protocol because its confirmation gates fail. Last Completed stays 5.3; Next
Target stays 5.3A, incomplete. Suggested commit message:
`Diagnose pick failures and preserve guarded 5.3A repair experiments`.


## Owner-authorized data amendment

On September 17, 2026 the owner replied **“approved, continue”** to the versioned
unassisted-expert/data proposal above. This extends 5.3A only. The first step is
teacher feasibility under ordinary physics. Additive data collection follows
only after that feasibility succeeds; original releases, split history and failed
experiments remain unchanged. New confirmation must be frozen before model
selection and cannot be described as unseen-family evidence.


### Teacher feasibility V1/V2

`pick_teacher_v1.yaml` freezes 24 exposed-instance development resets (four per
instance, alternating nominal/boundary) from seed 690260900. It explicitly
selects ordinary physics for the grasp preflight and live loop. Historical
teacher-only orientation preservation gates become pi because the unchanged
pick qualification contract has no orientation-preservation requirement; no
success, retention, speed, workspace or safety gate is weakened. The historical
orientation-hold mode remains the default for existing Level 4 configurations.

V1 succeeds on 20/24 attempts with zero live safety interventions, but all four
heavy-puck trials drop despite passing the old copied-state preflight. This
fails teacher qualification. Its files remain under `teacher_v1/`.

V2 freezes the identical development cases for a paired diagnostic comparison.
It selects the first original family fallback whose **exact live controller
loop** passes unassisted on a separate identical initial state. That same loop
then executes live. Failed candidates are retained in each attempt's receipt;
failed resets are never removed or replaced. The old approximate grasp
preflight and its historical behavior remain available for original recipes.
This is teacher development, not fresh learned-policy confirmation.

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_pick_teacher --config configs/level5/pick_teacher_v1.yaml --output-dir outputs/level5/pick_repair/teacher_v1
conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_pick_teacher --config configs/level5/pick_teacher_v2.yaml --output-dir outputs/level5/pick_repair/teacher_v2
```

These output directories preserve one completed attempt per recipe; use a new
version for an intentional changed experiment, never overwrite them.


V2 passed **24/24** teacher development trials with zero safety violations. Each
heavy-puck reset rejects the primary `[-0.015, 0, 0.012]` approach after its live
loop predicts a drop, then succeeds with the existing `[-0.004, 0, 0.020]`
fallback. Thus the previous setup admitted a grasp that was viable only under
a different preflight simulation sequence. No new template, force, friction,
actuator limit or qualification threshold was introduced.

A bounded diagnostic then held V4 seed5001 weights unchanged and ran all 24
teacher-development starts: **24/24 succeeded**. This is exposed development
evidence, not independent confirmation. It indicates that correcting grasp
selection may suffice without new data or retraining. Accordingly the authorized
amendment first tests that lower-cost hypothesis; data collection remains unused
unless the corrected setup fails. The owner's requested end state remains a
physically qualified pick policy, not merely a successful teacher.

### Setup V4 preregistration

`pick_setup_v4.yaml` locks the same three selected V4 weights plus the exact
unassisted `pick_teacher_v2.yaml` hash. It freezes 30 new validation resets from
730260900 and 54 confirmation resets from 740260900 before either is evaluated.
All original cells, four independent conditions and numerical/safety gates remain
in force. No retraining, weight selection, reset filtering or replacement is
permitted. Every seed must pass validation before its single confirmation.

Both scratch preflight and live setup use ordinary physics. The deterministic
teacher's full-state preflight chooses the grasp; live setup applies only its
open-hand approach and stops before querying acquire. Once entered, only the
unchanged learned bundle controls the hand. This is substantial engineered
planning assistance before entry and qualifies only those scripted starts.
All instance/family types have now been exposed in development; this cannot be
claimed as unseen-instance generalization, learned vision, arbitrary-start
robustness or learned reach-to-pick composition. Earlier failures remain failures.

The new regression spies on the live expert to forbid even querying acquire and
verifies exact action/state equality with the successful teacher prefix. It also
forbids object-orientation writes in every scratch and live workcell.


Setup V4 frozen entries: **30/30 validation and 54/54 confirmation**, with no
replacement. All three fixed models then passed full validation:

| Seed | Success | Worst family | Drop rate | Mean jerk | Safety violations |
| --- | --- | --- | --- | --- | --- |
| 5001 | 112/120 | 0.80 | 0.066667 | 0.001955365 | 0 |
| 5002 | 112/120 | 0.80 | 0.066667 | 0.002126573 | 0 |
| 5003 | 108/120 | 0.80 | 0.100000 | 0.002056136 | 0 |

Seed5003 meets the drop threshold exactly; the controller is not flawless.
These passing reports authorized one reserved confirmation per seed. There are
30 unique validation resets and 54 unique confirmation resets; the four original
condition labels are independently executed but do not change this state policy's
physics or observations. Do not interpret repeated condition runs as additional
unique randomized resets or perception evidence.

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_pick_setup freeze --config configs/level5/pick_setup_v4.yaml --output-dir outputs/level5/pick_repair/conditioned_setup_v4/resets --release-root .
conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_pick_setup validate --config configs/level5/pick_setup_v4.yaml --reset-dir outputs/level5/pick_repair/conditioned_setup_v4/resets --output-dir outputs/level5/pick_repair/conditioned_setup_v4/validation_5001 --seed 5001
conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_pick_setup confirm --config configs/level5/pick_setup_v4.yaml --reset-dir outputs/level5/pick_repair/conditioned_setup_v4/resets --output-dir outputs/level5/pick_repair/conditioned_setup_v4/confirmation_5001 --seed 5001
```

Validation and confirmation above were run for all three seeds 5001/5002/5003,
with CPU thread limits as described earlier. They preserve completed output
directories and are not commands to retry an already reserved confirmation.


### Setup V4 confirmation result: safety failure remains

| Seed | Success | Worst family | Drop rate | Board violations | Mean jerk | Gates |
| --- | --- | --- | --- | --- | --- | --- |
| 5001 | 212/216 | 0.944444 | 0.018519 | 0 | 0.001904113 | pass |
| 5002 | 208/216 | 0.888889 | 0 | 8 | 0.002080814 | **fail** |
| 5003 | 212/216 | 0.944444 | 0.018519 | 0 | 0.001933429 | pass |

The all-seed gate remains failed because seed5002 violates the zero-safety
requirement. No manual acceptance is requested and no pick policy is admitted.
The setup correction substantially improves success but cannot finish the repair
alone. All one-time attempts remain preserved. Full regression passes **791 tests
with one skip** (569.21s); 38 focused tests, Ruff and original locks pass.

Proceed with the owner-authorized additive unassisted dataset. Before collection,
a frame audit corrects a tempting mistaken hypothesis: the teacher's
`world.robot.base_orientation_wxyz` is mocap-command orientation, not measured
wrist orientation. No new measured-wrist feature is justified by that code.
The actual context mismatch is reference time: teacher orientation feedback
uses pre-approach object orientation while the policy uses skill-entry orientation.
Teacher V3 freezes skill-entry orientation as the reference and records causal
pre-action state inputs; its feasibility must pass before data collection.


## Additive unassisted data and V5

Teacher V3 aligns its reference at the acquisition boundary and passes 24/24
feasibility cases. The owner-authorized `pick_data_v1.yaml` then freezes 144 train
and 48 validation resets before collection: six exposed instances, three training
goal cells and one separate validation cell per instance, eight resets per cell.
Session identifiers name deterministic simulator recording groups and are disjoint
across splits. All **192/192** attempts succeed with zero recorded safety
violations. No failed case is replaced, no old demonstration is rewritten, and no
old test trajectory is relabeled as training data.

The new local immutable release is `datasets/level5-pick-unassisted-v1.tar.gz`,
with checksum and manifest sidecars. It contains only canonical episode payloads,
not the staging tree or diagnostic runs. LFS attributes and archive checksum pass.
The loader verifies the archive and each payload, explicit session/episode/seed
ownership, intervals and source bindings. Original input documents stay unchanged;
V5 explicitly declares an additional source and replaces only its own train/val
membership. Original test episodes remain historical and are never retried.
All six instance types are now training-exposed; confirmation will test fresh
randomized resets only, not unseen-instance generalization.

V5 retains the small 128x128 MLP, 200 full epochs and three seeds. Its inputs are
the 28 compact geometry features plus a causal consecutive full-closure/held
sample count, scaled by10 and capped at1. The counter uses no expert phase or
future state. It learns **three scalars**: closure increment, lift speed, and
orientation-correction magnitude. Closure direction, lift direction/bounds and
orientation direction are engineered. The orientation direction is derived from
current object pose versus the entry reference (full cuboid rotation, cylinder
axis tilt, no puck rotation); the learned magnitude is capped by remaining error
and0.02001rad. Zero holds and negative predicted magnitudes cannot reverse the
correction. This is constrained state behavior cloning, not independently learned
orientation direction, full-hand control or vision.

Training-only analysis motivates that direction. The full subsequent audit covers
18,368 training and6,119 validation frames. All source actions reconstruct to
floating-point precision (maximum absolute action residual4.44e-16, measured
rotation residual0), and offline/online causal feature histories match. The
frozen2e-6rad tolerance covers the legacy quaternion interpolation approximation;
it does not rewrite or clip source actions. Active orientation-magnitude targets
range0.001139–0.999504 in train (std0.361854) and0.000828–0.999504 in validation
(std0.368310). Magnitude is meaningfully varying, not a constant output.

The V5 recipe and checksum bind the new release and teacher. Fresh confirmation
starts at770260900 with54 resets and all four original condition executions;
it is frozen before training and selection. Full candidate selection uses all
48 new validation resets ×4 conditions=192 attempts per candidate per seed,
including rejected/failed starts if present. Numerical and safety thresholds are
unchanged. Earlier V4/setup failures remain preserved.

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_pick_amendment freeze --config configs/level5/pick_data_v1.yaml --output-dir outputs/level5/pick_repair/data_v1
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_pick_amendment collect --config configs/level5/pick_data_v1.yaml --output-dir outputs/level5/pick_repair/data_v1 --split train
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_pick_amendment collect --config configs/level5/pick_data_v1.yaml --output-dir outputs/level5/pick_repair/data_v1 --split validation
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_pick_amendment release --config configs/level5/pick_data_v1.yaml --output-dir outputs/level5/pick_repair/data_v1 --release-dir datasets
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_pick_amendment verify --config configs/level5/pick_policy_v5.yaml --output-dir outputs/level5/pick_repair/data_v1/physical_audit
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_skill --config configs/level5/pick_policy_v5.yaml --seed 5001 --dry-run
conda run --no-capture-output -n dexvision python -m dexvision.apps.repair_pick freeze-confirmation --config configs/level5/pick_policy_v5.yaml --output-dir outputs/level5/pick_repair/v5/confirmation_resets --release-root .
```

Each collection/packaging directory above is append-only and already reserved;
these commands document execution, not permission to overwrite existing artifacts.


## Post-integration evidence correction (V6)

Before V5 training, the independent saved-state audit found **191/192** physical
successes. `pick_unassisted_v1_train_750262201` has only nine consecutive stable
saved samples: the last cached speed is 0.01729754 m/s, but recomputing the same
qpos/qvel gives 0.02118818 m/s, exceeding the unchanged 0.0200 m/s gate. Exact
replay reproduces all qpos with zero residual. The issue is MuJoCo's cached
pre-integration derived fields, not changed data or a replay divergence.

V1 release, its failed audit, V5 recipe/cache and unused confirmation remain
preserved. V5 was never trained or selected. Teacher V4 recomputes terminal
contacts/pose/velocity on a separate MuJoCo data copy. The live integrator,
warm starts, causal controller inputs and commands remain unchanged. A regression
reproduces the old failed audit, verifies the new complete dwell, and proves the
live state is unchanged by the evidence read. Teacher V4 passes all 24/24
feasibility cases with zero safety violations. The new data V2 plan repeats the
same 192 frozen cases and partitions, with new episode/session identifiers and
no replacement. V6 will retain V5's model, features, decoder and training budget,
bind the corrected release, and freeze fresh confirmation starting at 780260900.

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_pick_teacher --config configs/level5/pick_teacher_v4.yaml --output-dir outputs/level5/pick_repair/teacher_v4
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_pick_amendment freeze --config configs/level5/pick_data_v2.yaml --output-dir outputs/level5/pick_repair/data_v2
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_pick_amendment collect --config configs/level5/pick_data_v2.yaml --output-dir outputs/level5/pick_repair/data_v2 --split train
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_pick_amendment collect --config configs/level5/pick_data_v2.yaml --output-dir outputs/level5/pick_repair/data_v2 --split validation
```


The corrected release collects **144/144 training** and **48/48 validation**
successes, all with zero safety violations. Its independent saved-state audit
passes **192/192**, including the case that failed V1. Corrected preflight may
select another existing fallback; it does not replace the reset or alter live
physics. Archive checksum and Git LFS attributes pass. V6 is frozen against this
release, teacher V4 and the same 200-epoch three-seed recipe. A complete, matching
passing source audit is now an executable pre-training requirement.

Software verification: full suite **802 passed, 1 skipped** (571.80 s). Subsequent
focused suite **53 passed** (37.10 s), including four source-audit guard tests
added after the full suite started. Ruff, whitespace, original config locks and
all old/new archive checksums pass. Logs are in the V6 output directory. These
software/source checks do not yet establish learned-policy qualification.


V6 pre-training checks pass on **18,367 train + 6,119 validation frames**.
Maximum absolute target/decoder residual is 4.44e-16; measured rotation residual
is zero, and every offline/online causal feature history matches. Its 54/54 valid
confirmation reset manifest SHA256 is
`4539e2d85254dc042d012a089fc40132e58a010fa110a15d2ddd1a37a1fac1cf`.
It was frozen before all three training runs. Each candidate receives all 192
validation attempts, and confirmation is reserved only after all three selected
models pass full validation.

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_pick_amendment release --config configs/level5/pick_data_v2.yaml --output-dir outputs/level5/pick_repair/data_v2 --release-dir datasets
conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_pick_amendment verify --config configs/level5/pick_policy_v6.yaml --output-dir outputs/level5/pick_repair/data_v2/physical_audit
conda run --no-capture-output -n dexvision python -m dexvision.apps.repair_pick freeze-confirmation --config configs/level5/pick_policy_v6.yaml --output-dir outputs/level5/pick_repair/v6/confirmation_resets --release-root .
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_skill --config configs/level5/pick_policy_v6.yaml --seed 5001 --dry-run
# Each of 5001, 5002 and 5003; full budget, then full candidate selection:
for seed in 5001 5002 5003; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.train_skill --config configs/level5/pick_policy_v6.yaml --seed "$seed"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_skill --config configs/level5/pick_policy_v6.yaml --select-candidates "outputs/level5/pick_repair/v6/seed_$seed" --track state --output-dir "outputs/level5/pick_repair/v6/validation_$seed"
done
```

These are the commands for the recorded append-only run. Completed output paths
cannot be reused to repeat qualification; intentional changes need new versions.
On this Mac, commands that capture Git provenance prefix PATH with
`/Library/Developer/CommandLineTools/usr/bin:` because the system Git is blocked
by the local Xcode-license state. Windows can run the same Python CLI separately
for each seed in its activated `dexvision` environment.

The V6 training-only active-target audit adds an important limitation: all 2,880
supervised closure gains are 1 within floating-point tolerance. That head fits
a constant demonstrated closing rate; the decoder supplies aperture bounds and
clipping at full closure. It does **not** learn a varied closure schedule or when
to approach an object. Lift targets vary from 0 to 1 (16,927 active samples,
standard deviation 0.382214); orientation magnitude varies from 0.001139 to
0.999504 (9,908 samples, standard deviation 0.361854). The model emits three
scalars, but only lift and orientation magnitude have varying active targets in
this release. `v6/training_target_variation.json` records the exact measurements.
No recipe or selected policy was changed by this explanatory audit.

### V6 confirmation and conditional owner review

Only after all three validation selections pass, the recorded pipeline executes
one confirmation per seed, then audits the saved physical states:

```bash
for seed in 5001 5002 5003; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.repair_pick confirm --config configs/level5/pick_policy_v6.yaml --run-dir "outputs/level5/pick_repair/v6/seed_$seed" --confirmation-dir outputs/level5/pick_repair/v6/confirmation_resets --output-dir "outputs/level5/pick_repair/v6/confirmation_$seed"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.repair_pick audit --config configs/level5/pick_policy_v6.yaml --run-dir "outputs/level5/pick_repair/v6/seed_$seed" --confirmation-dir outputs/level5/pick_repair/v6/confirmation_resets --report-dir "outputs/level5/pick_repair/v6/confirmation_$seed" --output-dir "outputs/level5/pick_repair/v6/audit_$seed"
done
```

These are append-only qualification commands, not instructions to rerun an
already reserved confirmation. If any numerical or physical audit fails, no
manual acceptance is requested and the guarded viewer refuses review.

If every automated gate passes, the owner reviews the saved states on this Mac:

```bash
cd /Users/mukunds/Desktop/DexVision
for seed in 5001 5002 5003; do
  PATH="/Library/Developer/CommandLineTools/usr/bin:$PATH" conda run --no-capture-output -n dexvision mjpython -m dexvision.apps.repair_pick review --config configs/level5/pick_policy_v6.yaml --run-dir "outputs/level5/pick_repair/v6/seed_$seed" --confirmation-dir outputs/level5/pick_repair/v6/confirmation_resets --report-dir "outputs/level5/pick_repair/v6/confirmation_$seed" --output-dir outputs/level5/pick_repair/v6/manual_review
 done
```

**Pass:** finish the sequence for all three seeds; each shown successful
instance/boundary case visibly grasps the intended object, lifts it clear of the
support and retains it stably. Motion must be physically plausible, with no
teleportation, artificial object-orientation hold, wrong-object grasp, board
exit or unexplained clipping. Any shown failed attempt must agree with its
printed failure reason and remain a failure. Expected, honestly reported
failures within the numerical gates are not relabeled successful.

**Fail:** a reported success lacks visible acquisition/lift/retention; a clip
shows a hidden reset, assistance, unexplained collision/board exit or incorrect
object; a shown failure is misreported; the viewer errors; or review is stopped
before all clips complete. Do not infer acceptance from opening the viewer.
The checkpoint remains incomplete until the owner explicitly confirms this
review passed. This does not qualify perception, arbitrary starts, composition,
registry admission or deployment, and does not start 5.4.


### V6 full validation selection

All three seeds finish 200 epochs and 14,400 updates. All four preregistered
candidates per seed receive 192 attempts: **2,304 validation executions** total.
The validation-only selection is:

| Seed | Selected epoch | Successes | Safety violations | Mean jerk |
|---|---:|---:|---:|---:|
| 5001 | 138 | 192/192 | 0 | 0.002157788935 |
| 5002 | 74 | 192/192 | 0 | 0.002070223095 |
| 5003 | 20 | 192/192 | 0 | 0.001970941025 |

Every selected candidate passes all unchanged gates, including worst-family
success 1.0 and drop rate 0. Earlier failing candidates remain in their original
reports. One-time fresh confirmation is in progress; no confirmation result
influenced this selection, and no manual acceptance is claimed.


### V6 fresh confirmation — safety failure

The one-time confirmation fails the all-seed zero-safety gate. Results remain
immutable and exposed; no owner review is requested.

| Seed | Successes | Worst family | Drops | Board violations | Mean jerk |
|---|---:|---:|---:|---:|---:|
| 5001 | 212/216 | 0.944444 | 0.000000 | 4 | 0.002153365546 |
| 5002 | 208/216 | 0.888889 | 0.000000 | 8 | 0.002030669735 |
| 5003 | 212/216 | 0.944444 | 0.000000 | 4 | 0.001963229574 |

Training-only target analysis shows lift, as well as closure, has only binary
active targets (0 or 1; largest lift residual from binary values 1.10e-13).
A validation-only fixed-weight diagnostic tests binary command gates at the
0.5 midpoint to investigate soft predicted values changing closure/lift timing.
No fresh confirmation case or action is used in that diagnostic.


## V7 binary command gates

Training active closure/lift targets are binary to within 1.10e-13. A fixed-weight,
validation-only diagnostic compares V6 continuous decoding with scores thresholded
at the 0.5 midpoint. All 144 comparisons (48 resets ×3 selected seeds) previously
closed one sample late; the binary decoder matches teacher closure timing in
all 144. Mean absolute lift-onset error falls from 1.145833 to 0.215278 samples;
exact lift-onset matches increase from 6.25% to 86.11%. Diagnostic successes are
48/48, 47/48 and 48/48, with zero safety violations; the seed 5002 drop is retained.
This diagnostic is not full qualification and uses no confirmation episodes.
Exact results and timing are in V6's `binary_validation_diagnostic_*` directories
and `binary_validation_timing.json`.

V7 freezes binary close/lift gate scores with the existing masked-MSE model and
0.5 threshold. Gate rates and directions are engineered from demonstrated actions;
closure's active target is constant 1, so no diverse learned closure schedule is
claimed. Orientation magnitude remains continuously learned. Source targets are
never rewritten, and cache generation rejects nonbinary targets. The same V2
release, source physical audit, 29 features, 128×128 model, three seeds, 200 epochs
and four-candidate full-validation schedule remain. A new fresh confirmation
starts at 790260900 and is frozen before training. V6 and its failed confirmation
remain preserved; no safety threshold, reset denominator or historical dataset
changes. Zero-output hold and the full-closure lift interlock have dedicated tests.

The V7 commands use the same CLI forms documented for V6, substituting
`configs/level5/pick_policy_v7.yaml` and `outputs/level5/pick_repair/v7/`.
V2 source verification is reused because its exact archive/manifest pins are
unchanged; the pre-training guard checks that full passing audit again.

V7 pre-training audit passes on all 24,486 source frames, with maximum action
residual 3.33e-16 and zero measured rotation residual. All 54 fresh entries are
valid; reset manifest SHA256 is
`582f88f91b4f88dee15bcf45c190e127c31037577ba49f33326cae50efe3274d`.
All three runs complete 200 epochs / 14,400 updates. At epoch 200, every model tensor
is bitwise identical to its corresponding V6 tensor: the deterministic repeated
training kept identical numeric targets, inputs, seeds and budgets. The change
is the explicitly versioned decoding semantics and bundle schema; it is not a
claim that V7 learned better weights. Selection may choose different epochs
because the closed-loop decoder differs. `v7/paired_weight_comparison.json`
records that comparison. Full validation and fresh confirmation remain required.


V7 software checks: **808 passed, 1 skipped** in the full suite (596.77 s);
**46 passed** in focused tests (21.85 s). Ruff and whitespace checks pass.
The full suite includes the physical-evidence, source-audit, all-seed review
guards, binary-target rejection and exact-cadence tests. Logs are in
`outputs/level5/pick_repair/v7/pytest_{full,focused}.log`.


### V7 full validation selection

All four candidates per seed receive the complete192-attempt matrix:2,304
validation executions. Each selected candidate passes every unchanged gate.

| Seed | Selected epoch | Successes | Safety violations | Mean jerk |
|---|---:|---:|---:|---:|
| 5001 | 138 | 192/192 | 0 | 0.002211843869 |
| 5002 | 20 | 192/192 | 0 | 0.002163098872 |
| 5003 | 80 | 192/192 | 0 | 0.002179128629 |

Worst-family success is1.0 and drop rate0 for all three selections. The change
in selected epochs for seeds5002/5003 demonstrates why validation must execute
the exact deployed decoder. At this selection handoff, fresh V7 confirmation
was pending; its final results follow below. No confirmation result was used
for selection or retraining.


## V7 final automated result — owner review pending

All three seeds pass the unchanged fresh-confirmation gates and independent
saved-state audit. The checkpoint remains incomplete until owner manual acceptance.

| Seed | Selected epoch | Validation | Fresh confirmation | Safety violations | Mean jerk |
|---|---:|---:|---:|---:|---:|
| 5001 | 138 | 192/192 | 212/216 | 0 | 0.002173100950 |
| 5002 | 20 | 192/192 | 212/216 | 0 | 0.002141928588 |
| 5003 | 80 | 192/192 | 212/216 | 0 | 0.002169260865 |

Every seed has 98.1481% success, 94.4444% worst-family success and 1.85185%
drops. Each matrix contains 54 fresh resets executed under four recorded
conditions; these are state-only repetitions, not RGB/perception evidence.
The same nominal `block_small` reset, `pick_confirmation_790261402`, drops for
each seed in all four conditions. All twelve failed executions remain recorded;
they are not relabeled, filtered or used to retune the policy. This is bounded
qualification at the original gates, not a claim of perfect success.

All **648/648** independent saved-state audits agree with the claimed physical
outcomes, including failures. The viewer guard verifies all three numerical
reports and matching complete audits. Selected weights match their full-validation
candidates, all artifact checksums pass, and current Python implementation hashes
match the one-time confirmations. The real viewer command rejects failed V6
with expected exit code 2. `v7/final_integrity.json` records the final checks.

The main repairs are an exact unassisted teacher/data source, recomputed
post-integration physical evidence, a causal contact-history feature, an explicit
orientation-direction decoder with learned magnitude, and binary closing/lifting
gates that preserve demonstrated timing. Closing/lifting rates, geometry,
posture synergy and bounds are engineered. The closure target is constant one;
this does not establish a varied learned grasp schedule. Selected V7 weights,
29-feature transform, gate/rotation decoder, fixed scales and cadence are one
bundle. All six instances are exposed; only new randomized resets are claimed.
Scripted family-aware approach and scratch preflight remain required. No vision,
arbitrary-start, composition, registry or deployment qualification is claimed.

### Exact owner manual verification command (macOS)

Run from Terminal using the dedicated environment. This replays saved states;
it does not repeat confirmation or alter the policy:

```bash
cd /Users/mukunds/Desktop/DexVision
for seed in 5001 5002 5003; do
  PATH="/Library/Developer/CommandLineTools/usr/bin:$PATH" conda run --no-capture-output -n dexvision mjpython -m dexvision.apps.repair_pick review --config configs/level5/pick_policy_v7.yaml --run-dir "outputs/level5/pick_repair/v7/seed_$seed" --confirmation-dir outputs/level5/pick_repair/v7/confirmation_resets --report-dir "outputs/level5/pick_repair/v7/confirmation_$seed" --output-dir outputs/level5/pick_repair/v7/manual_review
done
```

**Pass:** finish all clips for all three seeds; every shown success visibly grasps
the intended object, lifts it clear of support and holds it stably, without hidden
resets, teleportation, artificial orientation holds, wrong-object grasp, board exit
or unexplained clipping. The explicitly labeled small-block drop must appear as
a failure and match its printed reason; that retained failure is expected under
the unchanged numerical gates.

**Fail:** a claimed success lacks visible grasp/lift/retention; there is an
unreported failure, unexplained collision/clipping/board exit or hidden physical
assistance; a failure is misreported; the viewer errors; or the review stops before
all clips finish. Report pass/fail explicitly. Opening the viewer is not acceptance.

Last Completed stays 5.3; Next Target stays 5.3A. Automated acceptance is recorded,
owner acceptance remains false, and 5.4 has not started. No commit or remote upload
was made. Suggested commit: `Repair pick training with audited unassisted data and binary action gates`.


### Recorded V7 automated commands

These completed runs are append-only; do not rerun confirmation under another
path. The actual parallel execution log is `v7/pipeline.log`. All Python commands
ran in the `dexvision` Conda environment with single-thread BLAS/OpenMP settings.

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.repair_pick freeze-confirmation --config configs/level5/pick_policy_v7.yaml --output-dir outputs/level5/pick_repair/v7/confirmation_resets --release-root .
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_skill --config configs/level5/pick_policy_v7.yaml --seed 5001 --dry-run
# Run for each seed: 5001, 5002, 5003.
for seed in 5001 5002 5003; do
  conda run --no-capture-output -n dexvision python -m dexvision.apps.train_skill --config configs/level5/pick_policy_v7.yaml --seed "$seed"
  conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_skill --config configs/level5/pick_policy_v7.yaml --select-candidates "outputs/level5/pick_repair/v7/seed_$seed" --track state --output-dir "outputs/level5/pick_repair/v7/validation_$seed"
done
# Only after all three full selections passed:
for seed in 5001 5002 5003; do
  conda run --no-capture-output -n dexvision python -m dexvision.apps.repair_pick confirm --config configs/level5/pick_policy_v7.yaml --run-dir "outputs/level5/pick_repair/v7/seed_$seed" --confirmation-dir outputs/level5/pick_repair/v7/confirmation_resets --output-dir "outputs/level5/pick_repair/v7/confirmation_$seed"
  conda run --no-capture-output -n dexvision python -m dexvision.apps.repair_pick audit --config configs/level5/pick_policy_v7.yaml --run-dir "outputs/level5/pick_repair/v7/seed_$seed" --confirmation-dir outputs/level5/pick_repair/v7/confirmation_resets --report-dir "outputs/level5/pick_repair/v7/confirmation_$seed" --output-dir "outputs/level5/pick_repair/v7/audit_$seed"
done
conda run --no-capture-output -n dexvision pytest -q
conda run --no-capture-output -n dexvision pytest -q tests/test_pick_feedback.py tests/test_pick_skill_training.py tests/test_pick_skill_rollout.py tests/test_level5_learning_plan.py
conda run --no-capture-output -n dexvision ruff check dexvision tests
/Library/Developer/CommandLineTools/usr/bin/git diff --check
shasum -a 256 -c configs/level5/SHA256SUMS datasets/dexvision_level2_v1.tar.gz.sha256 datasets/level4-v1.tar.gz.sha256 datasets/level5-pick-unassisted-v1.tar.gz.sha256 datasets/level5-pick-unassisted-v2.tar.gz.sha256
```

On this Mac, the recorded training/selection/confirmation processes set
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and prepend the Command Line Tools Git path,
as explained above. Teacher and data collection/verification commands are recorded
in their corresponding earlier sections. The V7 reconstruction diagnostic ran
`conda run --no-capture-output -n dexvision python /private/tmp/audit_pick_v7_roundtrip.py`;
its complete numeric results are preserved in `v7/full_target_reconstruction.json`.


## Owner manual acceptance — October 3, 2026

The owner confirmed “Manual review passed” after receiving the prescribed
three-seed V7 saved-state viewer command and pass/fail criteria.
`docs/level5_pick_acceptance.json` records this statement and binds it to the
selected epochs 138/20/80, model hashes, recipe, reset manifest, confirmation
reports and physical audits. This is owner-reported review; no agent visual
observation or viewer completion log is asserted. No further manual check remains.

At acceptance, all recorded artifact hashes and the original working-tree
inventory matched the automated handoff. The read-only review guard passed for
all three seeds, and execution-code hashes still matched confirmation. Existing
automated results remain 808 passed / one skipped and 46 focused tests passed.
No training, confirmation, physical rollout or full regression was repeated for
this documentation-only update. Historical automated receipts above and under
outputs remain unchanged, including their then-pending acceptance status.
The twelve recorded small-block failures and all method limitations remain.

All automated and manual acceptance gates now pass. CURRENT_STATUS additionally
requires a commit before formal checkpoint completion. No commit or push was
made in this acceptance update, so Last Completed remains 5.3 and Next Target
remains 5.3A pending that commit. Level 5.4 has not started.

Suggested commit: `Repair pick training with audited unassisted data and binary action gates`.


## Completion and artifact handoff — October 3, 2026

The owner authorized “commit and push” after manual acceptance. This completion
update supersedes the earlier pending-commit status: 5.3A is complete, Last
Completed is 5.3A and Next Target is 5.4. No 5.4 implementation was started.
The acceptance receipt and current verification summary reflect this handoff;
all earlier automated outputs and failed experiments remain unchanged.

All three V7 training runs completed 200 epochs / 14,400 updates. Selected epochs
are 138, 20 and 80. Each seed directory under `outputs/level5/pick_repair/v7/`
retains `selected.pt`, resumable `last.pt`, `best_offline.pt`, validation candidates,
configuration and normalization snapshots, training manifest, selection and
checksums. Validation, confirmation trajectories, physical audits and test logs
also remain saved locally. The run manifests are immutable pre-training
provenance and may retain their initial pending-selection field; `selection.json`
and the selected checkpoint record the final selection.

Generated outputs are Git-ignored: this push does not back up trained weights or
rollout files. It publishes code, versioned recipes, documented metrics/digests,
owner acceptance and both additive dataset archives through Git LFS. A fresh
clone needs the local output artifacts or reproducible training to recover the
models; a model release is outside this checkpoint. No outputs or working raw
data were force-added, deleted, or rewritten.

Final commit checks: the focused suite passed again (46 tests, 22.97 seconds),
Ruff passed, both new dataset archive hashes matched, and all three saved final
checkpoints reloaded at 200 epochs / 14,400 updates with optimizer and RNG state.
Selected checkpoints reloaded at epochs 138/20/80 and run checksum indexes passed.
The staged diff exposed an existing terminal blank line in the checksum-bound
`scripted_expert_settings.py`; a path-specific Git whitespace attribute preserves
the exact evaluated source bytes while allowing the staged whitespace check.
