# Place policy repair (5.4A)

Status: **5.4A complete; V25 state gates and owner viewer acceptance passed**.
This work follows the preserved negative 5.4 experiment. The owner explicitly
requested a repair; owner acceptance was confirmed on October 5, 2026 with
"Passed". The separate receipt is `docs/level5_place_acceptance.json`.
Level 5.5 remains unstarted.
Earlier failed versions and development diagnostics are preserved below;
the final V25 section distinguishes fresh confirmation from exposed probes.

## Why the V1 policy fails

V1's selected policies achieve 4/216, 0/216 and 12/216 on the already exposed
held-out matrix. Closed-loop traces show small finger-opening commands
accumulating during transport. The original Level 4 scripted source holds object
orientation during collection, but V1 policy control uses ordinary physics.
The source place config declares 0.070 m placement clearance and 0.050 m
release tolerance; the frozen independent place gate requires release within
0.025 m. An ad hoc replay of the exact thirty validation source continuations
after the original entry yielded 3/30 successes under ordinary physics and
4/30 with the source's orientation hold, when measured by the independent
place gate. These are development probes, not a source-release amendment.

The 5.2A repair corrected measured/commanded frames, used fixed physical
scales, and selected a bounded geometric decoder by full closed-loop
validation; more epochs alone did not work. The 5.3A repair added an
ordinary-physics teacher/data source, kept exact causal entry and physical
audits, and used a coupled action decoder with causal binary gates. Those
lessons support checking the place source and release coupling directly.
They do not authorize copying the reach or pick decoder unchanged.

## Exposed validation diagnostics

Holding V1 seed 5001 epoch-20 finger commands until measured object-to-target
distance was within 0.025 m improved 30 unique validation resets from 5 to 19
successes. It still failed on two premature releases, eight bounces and one
workspace violation. The immutable diagnostic is in
`outputs/level5/place_54_v1/diagnostic_gated_candidate20/`.

Lowering the release trigger to 0.015 m alone made the old V1 motion controller
stall or leave the workspace. A diagnostic that used a bounded short correction
near the target, coupled release fingers, and held the commanded hand pose
during opening and settling reached 23/30 validation successes. It still had
five premature physical drops and two bounces, so it **fails** the unchanged
premature-release gate. These attempts are retained in ignored
`outputs/level5/place_repair/diagnostics/latched_015/`; they are not selected
weights or a policy bundle. Replacing the old entry with the qualified V7 pick
policy on the same exposed starts did not repair the V1 place network's
out-of-distribution transport, especially for cylinders. No held-out retest
or threshold change was made.

## New unassisted teacher feasibility

`configs/level5/place_teacher_v1.yaml` is a separate checksum-locked teacher
diagnostic. It uses the existing exact-loop unassisted pick V4 teacher to
establish a physically held object from a fresh task reset, then runs the
deterministic place expert under ordinary physics. The repair teacher holds
the audited rank-one finger posture during transport, lowers the place
clearance to 0.030 m, makes at most a 0.001 m measured correction near target,
and latches the hand pose during coupled opening/settling. Development probes
showed that a 0.023 m cuboid release trigger avoids two hand-table contacts;
cylinder and puck triggers remain 0.015 m. Every trigger is inside the frozen
0.025 m release gate. No object state is written after reset and no failed
case is replaced. The expert, its grasp-template preflight and its bounded
decoder are **scripted data-source controls, not a learned place policy**.

The V1 command was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher \
  --config configs/level5/place_teacher_v1.yaml \
  --output-dir outputs/level5/place_repair/teacher_v1
```

It ran all thirty exposed validation starts exactly once and wrote each
outcome, commanded place trajectory, physical qpos/qvel and a checksum index.
V1 preserved **28/30** successes, one premature cuboid drop and one cylinder
bounce, with zero safety violations. It fails its frozen 95% teacher threshold.
The complete negative result is in
`outputs/level5/place_repair/teacher_v1/summary.json`.

A paired validation-only probe on the cylinder rebound passed when the
rank-one opening step was halved. `place_teacher_v2.yaml` freezes exactly that
release-rate change and retains the same thirty starts, 95%/90%/zero-safety
teacher gates and place qualification thresholds. Its full run achieved
**29/30** successes: 9/10 cuboid, 10/10 cylinder and 10/10 flat-puck, with one
preserved premature cuboid drop and zero safety or workspace saturations. The
result and all attempts are in `outputs/level5/place_repair/teacher_v2/`.
Run it with the V1 command above, substituting `place_teacher_v2.yaml` and
`teacher_v2` in the two paths. This passes a **teacher feasibility screen only**.

The separately locked `place_teacher_coverage_v1.yaml` then used thirty new
development seeds, one nominal reset for every pairing of six objects and
five target cells. The exact command was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher \
  --config configs/level5/place_teacher_coverage_v1.yaml \
  --output-dir outputs/level5/place_repair/teacher_coverage_v1
```

The broader screen **failed**: 15/30 successes, five hand-table contact stops,
six premature releases and four object-workspace violations. Target successes
were 3/6 left receptacle, 1/6 inspection pad, 3/6 slot A, 4/6 slot B and
4/6 right receptacle. All attempts, source seeds, actions and physical states
are saved under that ignored output directory. Representative failure probes
show that some grasp configurations hit the table before the object reaches
the 0.025 m release distance; other starts place larger objects at the board
safe-margin boundary before place control. The small slot-B validation success
therefore does not establish a feasible all-target source. No failed reset is
removed, and **no dataset release is authorized by this evidence**.

Further exposed diagnostics kept the same failing resets and tried existing
cuboid grasp fallbacks and a bounded set of wrist pitches. None rescued the
inspection-pad hand-table contact without an acquisition failure or another
physical drop. The qualified V7 pick bundle also reached the same contact
boundary in representative cells. These are failed source-design probes, not
policy evaluations or a license to alter the frozen terminal/safety gates.

The next repair work must establish ordinary-physics feasibility across all
required object/target cells without changing the frozen safety or terminal
gates. Only then can an additive immutable training source and a new learned
policy recipe be frozen. No learned place policy, closed-loop policy selection,
fresh confirmation, physical policy audit or owner manual acceptance is
claimed yet.

### Further all-cell failure diagnosis

Saved-state contact inspection of inspection-pad reset `820260901` identified
`rh_forearm` touching the left edge of `workcell_table` during descent. A
validation-only, bounded 15-degree wrist yaw after the physically verified
scripted pick cleared that contact on four of five originally failing
hand-table cases; the fifth (`820260919`) still hit the table. The rotation
kept the measured object held and checked its board margin and hand-table
contact at every sample. This was a temporary diagnostic, not a frozen teacher
recipe or learned policy action.

On the broader thirty-cell development matrix, a diagnostic combination of
that cuboid yaw, 0.09 m transport height for cylinders and pucks, 0.015 m
cylinder placement clearance, and a 0.023 m cylinder release trigger achieved
**25/30** measured settled placements. The trigger remains inside the frozen
0.025 m release gate. The five failures were three cuboid setup-stage physical
failures, one cuboid hand-table contact, and one puck workspace violation.
All attempts are retained in `outputs/level5/place_repair/diagnostics/combo_v1_*`.
These changes were applied by a temporary probe, so they require a versioned
implementation, complete setup-safety audit and a fresh coverage screen before
they can support any additive dataset.

Further paired probes on the five failures found that an existing alternative
cuboid grasp template rescued `820260919`; another rescued `820260917` but
failed the other large-block starts. A short inboard motion or diagonal lift
rescued puck reset `820260926`. The two remaining large-block edge starts,
`820260915` and `820260918`, either crossed the board safe margin or lost the
object before transport. Testing slower transport, reduced lift height,
alternate pre-existing grasp templates, a fixed hand orientation, and the
qualified V7 pick entry did not establish a safe successful continuation for
those starts. These are exposed, unselected diagnostics; combining their
individual successes would be selection on the same development resets and
would still need a single frozen protocol and new independent evidence.

### Frozen V3 source and fresh all-cell screen

`place_teacher_v3.yaml` froze the bounded post-pick grip/yaw setup, declared
large-block lift extension, per-family transport/clearance adjustments and
small first-lift inboard shift for a left-edge puck. The exact combined exposed
`820260900` probe passed 30/30, but that matrix informed the recipe and is
development evidence only. The implementation saves the scripted pick, every
post-pick setup action and physical state, and the place trajectory. Setup
checks measured retention, board margin, hand-table contact and unintended
contact after each action. It makes no post-reset object-state writes and does
not alter the frozen terminal or safety gates.

The checksum-locked `place_teacher_coverage_v2.yaml` then froze 30 new nominal
seeds (`830260900` through `830260929`), one for each object/target cell, before
execution. Run the recorded command with a new output directory for any repeat:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher \
  --config configs/level5/place_teacher_coverage_v2.yaml \
  --output-dir outputs/level5/place_repair/teacher_coverage_v2
```

The fresh screen **failed at 22/30**: cuboid 5/10, cylinder 9/10 and puck 8/10.
It retains two setup board-margin violations, three table-contact stops, one
premature release and two bounces. All 30 attempts and the failed gate are in
`outputs/level5/place_repair/teacher_coverage_v2/`. The exposed-matrix 30/30
did not generalize. No additive dataset, learned place policy, qualification or
manual acceptance is justified by this source.

Paired diagnostics on the fresh failures remain development-only. A different
bounded wrist yaw removed three large-block table contacts. Grip tightening
rescued the tall-cylinder premature release; slower finger opening rescued the
two puck bounces. The two large-block left-edge starts `830260915` and
`830260919` remain unresolved: each standard grasp-template fallback either
fails preflight or crosses the frozen board margin during subsequent grip/yaw.
Inboard transport tested so far either drops the object or reaches placement
with a physical bounce. The next source revision needs a causal, physically
safe way to establish and retain those edge grasps, then a new frozen recipe
and an independent all-cell development screen before data publication.

### Frozen V4 and exact-reset search development

`place_teacher_v4.yaml` couples a 0.002 m inward hand-command increment to each
large-block tightening action, varies bounded yaw by large-block target, adds
small tall-cylinder tightening, and slows puck finger opening. On the exposed
`830260900` matrix it passed **29/30**, retaining one large-block bounce and
zero safety violations. It was frozen before `place_teacher_coverage_v3.yaml`
selected fresh `840260900` starts. That independent development screen
**failed at 26/30**: three large-block table contacts and one tall-cylinder
bounce, with zero safety violations. The exact command and every attempt are in
`outputs/level5/place_repair/teacher_coverage_v3/`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher \
  --config configs/level5/place_teacher_coverage_v3.yaml \
  --output-dir outputs/level5/place_repair/teacher_coverage_v3
```

Paired probes on the four exposed failures found bounded alternatives that
settled each: changed yaw for two large-block targets, changed yaw plus a
0.024 m cuboid release trigger for the inspection pad, and slower cylinder
opening. The trigger remains inside the frozen 0.025 m release gate. These
individual probes are not an independent screen. `place_teacher_search_v1.yaml`
therefore freezes a small candidate grid and tests it on separately reset,
ordinary-physics task copies. It retains every preview, selects the first
physical pass with zero safety counters, and reruns that candidate live from
the exact same reset. The original reset is never filtered or replaced. The
selection is a **scripted source preflight**, like the 5.3A pick preflight; it
is not a learned policy. The focused `840260916` preview and live receipt
match. The separate frozen `850260900` all-cell screen passed **29/30** with
9/10 worst-family success and zero safety violations. One small-block
premature release remains counted as a failure. All 53 previews, 30 live
continuations and reset details are in
`outputs/level5/place_repair/teacher_search_coverage_v1/`. The exact command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher \
  --config configs/level5/place_teacher_search_coverage_v1.yaml \
  --output-dir outputs/level5/place_repair/teacher_search_coverage_v1
```

An independent file read compared every saved array in the selected preview
with its live continuation: **540 arrays across all 30 resets match exactly**,
including actions, qpos, qvel and timestamps. The coverage checksum index and
all 30 attempt indexes verify. This checks deterministic reproduction of the
preflight choice; it does not substitute for an additive-release audit or
learned-policy evaluation.

The 29 successful live place trajectories contain 3,189 control frames:
1,026 transport, 1,562 place, six source-labeled release and 595 latched
opening/settling frames. All thirteen varying finger channels form one
numerical rank-one opening direction (next singular value below `5e-15`).
No finger channel moves before the latched release, whereas wrist orientation
moves on 1,787 frames and Cartesian transport/lowering remains active. A new
learned recipe therefore needs a causal release interlock and coupled opening,
plus active position and orientation control; it cannot reuse the V1 free
finger head or reach's fixed-wrist decoder. These measurements use exposed
source data only and do not select a model or authorize training yet.

This clears the scripted-teacher feasibility item of 5.4A. It does not publish
an additive dataset or qualify a learned policy. A separately versioned search
protocol must be used for future source collection and held-object setup; failed
resets stay in the denominator.

Repair-tool checks pass: 19 focused teacher/lock-inventory tests and Ruff over
`dexvision` and `tests`. A full regression started before the search changes
passed **823 tests with one skip**; the new search tests and final full suite
must still pass. These checks verify software behavior, not policy
qualification.

### Additive data V1 stopped at a new safety failure

`place_data_v1.yaml` froze 144 training and 48 validation reset assignments,
whole simulator-session ownership by split/cell, a 95% split success minimum,
90% worst-family minimum and zero live safety violations. The data path replays
the selected pick, setup and place actions exactly from reset and records 195
causal features before each place action. A focused integration test passed.
A strict floating-point equality check initially rejected one valid quaternion
rounded by about `1.1e-16`; replacing that check with a `1e-12` tolerance
leaves the original safety, action and saved-state checks intact.

The independent saved-state audit restores each commanded mocap hand target
before MuJoCo forward; qpos/qvel alone do not locate that body correctly.
It records hand-table contact before the release command, in the release
window, and on either side of physical detachment. A diagnostic large-block
success has six little-finger table-touch samples during post-release
settling. Another has contact from the release-trigger boundary onward,
including three samples before physical detachment. Neither bounces or fails
the frozen terminal gate. The scripted controller skips its active-lowering
contact stop once release is latched, so the audit fails contact before the
release command and retains release-window counts for owner viewer review.
This is a disclosed physical limitation, not zero-contact evidence.

The saved-state audit reports the optional 0.04 m transport-displacement
indicator but does not add it to the frozen place success gate. Eight
successful `setup_slot_b` validation starts were already close enough to the
target to settle with less displacement. They remain successful settled
placements under the original terminal definition, and their short-motion
geometry remains visible in the audit. This is not a long-transport claim.

Frozen training reset `860260906` then found a real source-planning flaw. All
12 candidates failed, but V1 executed its first candidate live even though it
crossed the board margin; other candidates had zero safety violations but
failed placement. The live attempt is a retained **object-workspace safety
failure**. Because the frozen data publication gate requires zero live safety,
data V1 is ineligible for publication. Collection was stopped after ten
completed assignments and an interrupted eleventh; the plan, source attempts
and successful staging payloads remain under ignored
`outputs/level5/place_repair/data_v1/`. No release archive was created, and no
reset was replaced or relabeled.

`place_teacher_search_v2.yaml` preserves the exact V1 candidate grid. When no
candidate passes, it chooses the first zero-safety failure for the single live
continuation; if none is safe, it rejects before live place execution. Both
paths remain failures in the denominator. The frozen
`place_teacher_search_coverage_v2.yaml` schedules 60 fresh development resets:
nominal and boundary variation for all 30 object/target cells. It passed
**59/60**, with 19/20 worst-family success, one preserved premature release,
and zero selected live safety violations. The exact command and all 76
previews, 60 live outcomes are under
`outputs/level5/place_repair/teacher_search_coverage_v2/`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.qualify_place_teacher \
  --config configs/level5/place_teacher_search_coverage_v2.yaml \
  --output-dir outputs/level5/place_repair/teacher_search_coverage_v2
```

`place_data_v2.yaml` froze new 144 training and 48 validation assignments
against that exact report, using new seeds, episode/session ids and release
name. The completed collection passed 140/144 training and 47/48 validation
resets, exceeding the 95% split and 90% family publication minima with zero
live safety violations. All five failures remain in the reports and were
excluded from supervision. The independent saved-state audit reconstructed
every successful place trajectory with MuJoCo forward and passed **187/187**.
The worst training family is 46/48 and the worst validation family is 15/16.
It records 16 short-transport starts and 770 release-window hand-table contact
samples across the full source, with zero before the release command, rather
than silently presenting them as long transports or contact-free releases.
`datasets/level5-place-unassisted-v2.tar.gz` contains only those 187 canonical
place payloads and is bound to its checksum, LFS attribute and manifest. V1
remains a failed partial collection with no release. No learned place policy
has been trained or qualified from V2 yet.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.collect_place_amendment verify --config configs/level5/place_data_v2.yaml --output-dir outputs/level5/place_repair/data_v2
shasum -a 256 -c datasets/level5-place-unassisted-v2.tar.gz.sha256
git check-attr filter -- datasets/level5-place-unassisted-v2.tar.gz
```

An early V2 causal-input audit found that the task's online phase labels lag
the expert's geometric stages: among the first 33 captured episodes, 1,917
expert `place` frames still have the causal `transport` label, while some
place-entry frames retain `approach`, `acquire`, `lift` or `stabilize`. The
payload records both labels but only the causal online label enters its
195-field observation. A learned decoder must infer lower/release readiness
from current object-target geometry and grip state rather than treating the
expert stage label as an input. This is the same causal-phase lesson seen in
5.3A pick and is an exposed training-source audit, not policy evaluation.

### Learned policy development after additive release

The checksum-locked V2 direct-delta recipe trained all three seeds for 200
epochs and 12,200 updates each on 140 successful training episodes, with 47
successful validation episodes reserved for offline loss. It learned XYZ,
rotation-vector and release scores, and coupled the physical opening vector.
Seed 5001 epoch 43 had 11 premature releases, four object-workspace exits and
four unsettled attempts in the first 19 independent closed-loop validation
attempts. The policy drifted in XY and did not lower reliably. This exposed
interface failure stopped the full matrix; no selection or confirmation was
claimed. All weights and partial rollouts remain under ignored
`outputs/level5/place_repair/policy_v2/`.

V3 froze a goal-directed hand-waypoint decoder and separately trained the same
three full-budget seeds. Its seed 5001 epoch 20 and 183 single-reset diagnostics
both timed out without reaching the physical release gate. The held object's
offset from the commanded hand changed during transport, leaving the object
about 3 cm above the intended center. These are negative development results,
not qualification. The V3 weights and attempts remain under `policy_v3/`.

V4 freezes live measured-object feedback and family release error vectors
computed as medians at the first coupled opening command on **training only**:
target minus object is `[0.0076, 0.0012, -0.0209]` m for cuboids,
`[0.0013, 0.0009, -0.0220]` m for cylinders and
`[0.0052, 0.0006, -0.0133]` m for flat pucks. The MLP learns a move gate,
orientation correction and release gate; the decoder supplies direction,
bounded steps, family release distance, coupled finger opening and a latch.
All three V4 seeds completed 200 epochs. Partial full-matrix validation on
seed 5001 suffices to reject every candidate against the unchanged worst-family
gate: after all 64 cuboid attempts, epochs 20/80/183 passed 32/36/36 (< 60%);
epoch 200 passed only 28/64 cylinder attempts. The four interrupted runs
retain 145/136/158/168 completed attempts and their failure receipts in
`policy_v4/partial_validation_summary.json`; there is no selected V4 policy
or confirmation. The task-local scripted place setting resolves to a
**0.005 m** transport step, while V4 declared 0.002 m. This lengthened held
transport and produced slips and timeouts. The difference was independently
read from the task's merged configuration, not inferred from held-out results.

V5 freezes the corrected 0.005 m transport step without changing the source,
training targets, budget, family release medians, safety limits, or fresh
confirmation reservation. All three seeds completed 200 epochs and 12,200
updates each. Complete 48-reset by four-condition closed-loop validation
selected epochs 20, 200 and 80 for seeds 5001, 5002 and 5003. Their selected
validation success counts were 172/192, 172/192 and 188/192; each passed the
frozen numerical gates on that matrix. The validation reset assignment covers
only `setup_slot_b`, however. A separate new-seed, one-condition all-cell
development screen for seed 5001 passed 24/30 with one large-block
object-workspace violation, one unsettled large-block attempt, one small-block
premature drop and three bounces. It fails zero safety and the settled-among-
nominally-placed gate. **No V5 confirmation reset was used.** Its selected
weights and all failed attempts remain local negative transfer evidence.

Paired development probes found three bounded changes: a 0.0245 m cuboid
release trigger (inside the unchanged 0.025 m physical release limit) avoided
the large-block board exit; a goal-relative waypoint step while genuinely
held and outside the release distance resolved the large-block stall; and
halving coupled cylinder opening to 0.0125 avoided one bounce. The combined
diagnostic on the already exposed nominal all-cell seeds passed **27/30** with
zero safety, 0.8 worst-family success and all frozen state gates. A
one-condition replay of the exposed slot-B validation assignments passed
**43/48**, zero safety and all gates. Neither diagnostic selects a checkpoint.

The separate checksum-locked V6 recipe freezes exactly those decoder changes
and preserves the same source release, learned targets, physical scales,
transport/descent cadence, three seeds, 200-epoch budget, safety gates and
unused `930260900` confirmation reservation. The MLP still learns orientation,
move and release scores. The decoder supplies movement direction and bounds,
forces a waypoint while held outside the release region, and couples finger
opening. This is state-based behavior cloning with engineered feedback, not
learned vision or unconstrained full-hand control. V6 trained all three seeds
for 200 epochs and 12,200 updates each; its model tensors match the
corresponding V5 seeds because the learning targets and optimizer are
unchanged. Before V6 confirmation, another distinct `950260900` boundary
development screen passed **27/30** across all cells, with zero safety and all
frozen state gates. These 940/950 development resets are exposed and are not
confirmation. Full V6 closed-loop candidate validation later selected epochs
20, 200 and 80. The fresh confirmation remained unused.

### All-target selection and V8 safety overlay

Seed 5002's V6 selected epoch 200 had 168/192 validation successes and zero
safety but **failed** the original settled-among-nominally-placed gate at
89.36% versus 90%. A separate new `960260900` nominal/boundary all-target
development screen for that bundle achieved 50/60 with one cylinder
object-workspace violation. Epoch 80 on the same exposed starts achieved
51/60, zero safety and all gates. This showed that the slot-B-only ranking
was insufficient for all-target qualification.

`place_selection_v7.yaml` froze a second, additive selection screen on
`990260900` nominal/boundary starts for candidates passing **every** full
validation gate. It found seed 5001 epoch 20 at 54/60 and seed 5003 epoch 20
at 54/60, both with zero safety and all gates. Neither eligible seed 5002
candidate passed zero safety: epoch 20 reached 51/60 with two workspace
violations, and epoch 80 reached 49/60 with one. The selector correctly
refused to create a seed 5002 bundle. These attempts remain under ignored
`outputs/level5/place_repair/policy_v7_selection/`; none is confirmation or
owner acceptance.

Saved-state inspection found that one cylinder failure built a large
command-to-object lag during held transport, while a large-block receptacle
failure stayed near the target without releasing as the learned release score
remained negative. A development-only 0.05 m change-from-initial-offset hold
prevented the cylinder board exit. An outward-moving cuboid near a receptacle
edge settled when a geometric fallback opened within the **unchanged**
physical release distance. Slower coupled cuboid opening reduced three
receptacle/slot bounces; retaining the original opening rate on planar
targets avoided a new pad bounce. The exact combined decoder passed **53/60**
on the already exposed 990 starts for seed 5002 epoch 80, with 80%
worst-family success, zero safety and every frozen state gate. This is
development evidence and cannot serve as V8 selection.

`place_policy_safety_v8.yaml` locks that decoder as an overlay on the
existing three 200-epoch V6 weight sets. It changes no learned weights or
training data. The MLP still supplies move, orientation and release scores;
geometric waypoint movement, abnormal-lag hold, target-specific coupled
opening and the conditional edge release are engineered controls. Each
selected bundle must bind the V6 model/normalization, exact V8 overlay and
control cadence. The separately locked `place_safety_selection_v8.yaml`
reserves new `1000260900` all-target selection starts after complete V8
validation. The original `930260900` fresh confirmation remains unused.
The following selection result supersedes the earlier V8 pending status.

### V8 selection result and V9 opening repair

All frozen V8 validation candidates completed. The V8 all-target selector
selected seed 5001 epoch 80 (53/60 fresh selection successes, zero safety)
and seed 5002 epoch 20 (55/60, zero safety). Seed 5003 epoch 20 reached
50/60 with zero safety but only **50/58** settled nominal placements, below
the original 90% gate. Epoch 80 reached 46/60 and had two object-workspace
violations. The selector made no seed-5003 bundle; **no confirmation reset
was used**. All candidates and failed attempts remain in ignored local outputs.

The V8 selection matrix is now exposed development evidence. Saved failures
showed multiple bounces after nominal release, not a target-distance or
premature-release threshold problem. Targeted replays tested smaller finger
opening steps without changing V6 weights, release distance or transport
cadence. A 0.00625 step for nonplanar cuboids, cylinders and pucks, retaining
the V8 planar cuboid step, passed the complete exposed 60-attempt matrix for
the predeclared V6 epochs 80, 20 and 20: **51/60, 54/60 and 53/60**, with
zero safety and all original state gates. A planar cuboid adjustment merely
traded one bounce for another and was discarded. These reused-reset results
do not qualify a policy.

`place_policy_slow_v9.yaml` freezes the smaller opening rates as a narrow
overlay on the checksum-locked V8 controller and unchanged fully trained V6
weights. `place_slow_selection_v9.yaml` predeclares the three candidate epochs
from prior V8 evidence. It requires full V9 validation for each and a distinct
fresh `1020260900` all-target screen. Only then may the untouched original
`930260900` confirmation run once per seed. Full V9 validation passed every
frozen state gate for all three predeclared epochs, with zero safety
violations. The distinct fresh 1020-series selection screen passed seed 5001
epoch 80 at **55/60** and zero safety, but rejected seed 5002 epoch 20 at
**50/60** (50/56 nominal placements settled, 89.29%) and seed 5003 epoch 20
at **53/60** (53/60 settled, 88.33%). Both failures miss only the unchanged
90% settling gate. V9 therefore has no three-seed selected bundle and **no
confirmation reset was used**. A lower geometric release point and a
measured-velocity pause were tested on these now-exposed starts; the lower
point regressed badly on complete development screens and the pause did not
resolve enough targeted bounces. Neither was frozen as a new recipe. Physical
audit and owner viewer acceptance remain pending.

Suppressing the learned orientation output resolved a few targeted bounces
but failed the complete exposed 1020-series matrices for all three seeds
(49/60, 52/60 and 51/60 successes, each below the 90% settling gate). Source
orientation labels have measurable variation, especially in their second
component, so a fixed-orientation decoder would also discard a supported
training signal. This diagnostic is retained locally and was not frozen.
The next repair within 5.4A needs a physical release mechanism that remains
stable across fresh reset sets; repeated tuning of opening, release height or
orientation on the exposed matrices is not a qualification result.

## V10 release-clearance diagnosis and frozen candidate

A paired teacher/policy replay on an exposed cylinder receptacle reset showed
the V9 palm command about 5 mm lower during opening. The V9 fingers also took
six small opening steps to detach the object, while the qualified scripted
source detached in two to three steps. Replacing V9 opening rates with the
source rates failed complete exposed screens, so opening rate alone did not
repair the physical bounce. Small upward palm clearance at the first release
command helped, but one global offset regressed other target types.

The V10 overlay freezes a single rule before any fresh V10 reset: 1.5 mm
clearance for cuboids, 3 mm for cylinders over receptacles and pucks over
planar zones or slots, and zero for the other cylinder/puck targets. The
policy retains exact V6 learned weights, V9 opening, V8 safety, all original
terminal gates and the original 930-series one-time confirmation boundary.
The rise is applied once when the causal controller first latches release.
It is an engineered release decoder rule, not a learned output.

The fixed rule passed all frozen numerical gates on two **exposed** complete
60-reset nominal all-target screens for each predeclared candidate. On the
1000-series matrix, seeds 5001/5002/5003 achieved 53/60, 53/60 and 56/60
successes; settled nominal placements were 94.64%, 91.38% and 96.55%.
On the 1020-series matrix, they achieved 56/60, 53/60 and 58/60; settling
was 96.55%, 94.64% and 96.67%. Every screen had zero safety violations.
These results are development evidence because both matrices informed the
repair. The checksum-locked V10 recipe and 1040-series selection protocol
were frozen before any V10 full validation or fresh selection run. No
confirmation reset was consumed by these diagnostics.

Complete V10 validation covered 192 independent state/physics rollouts per
seed. Seeds 5001 and 5003 passed all frozen gates, but seed 5002 **failed the
worst-family gate**: only 36/64 cuboid attempts succeeded (56.25%). The added
1.5 mm cuboid clearance turned 12 previously settled slot placements into
bounces. V10 did not run its fresh 1040 selection screen or confirmation.
The result and all failed attempts remain under
`outputs/level5/place_repair/policy_v10/`.

V11 freezes one narrow correction: zero *additional* hand rise for cuboid
placement slots, while retaining V10 clearance for all other object/target
pairs. This follows the physical slot bounce evidence; it does not alter the
learned weights or acceptance gates. Recombining the saved, deterministic
V9/V10 full-rollout results predicted passing gates on validation and both
exposed all-target matrices for all three seeds. That calculation is a
development prediction only. V11 has its own checksum-locked implementation,
complete validation requirement and a new 1060-series all-target selection
matrix. The original 930-series confirmation remains unused.

Complete V11 validation passed all original state gates for predeclared
epochs 80, 20 and 20, with zero safety violations. The separately frozen
1060-series all-target screen selected seed 5001 at 49/60 successes and
49/54 settled nominal placements, but **rejected** seed 5002 at 52/60 and
52/58 settled (89.66%) and seed 5003 at 49/60 and 49/58 settled (84.48%).
Both miss the unchanged 90% settling gate. Seed 5001's selected bundle and
all failed screens remain local evidence. There is no three-seed V11 bundle,
the original 930-series confirmation has not run, and physical audit and
owner viewer acceptance remain pending.

### Further V11 failure diagnosis

The fresh 1060 failures led to bounded development-only release probes. A
0–6 mm static rise, source-rate finger opening, a measured-offset/vertical-
velocity opening rule, support-triggered hand retreat, and a geometric puck
opening fallback each rescued some exact exposed resets but regressed others.
The complete adaptive-rate screen passed all three seeds on exposed 1060;
cross-checking without changing the rule failed seed 5003 on exposed 1000
(51/58 settled nominal placements). Restoring the small cuboid clearance
passed that matrix and all three seeds on 1060, but its complete exposed
1020 screen failed seeds 5002 and 5003 (49/56 and 52/60 settled). A source-
inspired faster finger opening alone also failed seed 5002's complete 1060
settling gate. These are not frozen or selected policy bundles.

Saved-state reconstruction of a persistent heavy-puck bounce showed the
object contacting the table while still touching the fingers, then losing
support with a finger contact still present. Retreating the hand after the
first supported observation failed nearly all targeted bounces. Starting
puck opening earlier inside the unchanged 0.025 m tolerance, with or without
a bounded geometric fallback when the learned release score was late,
rescued some targeted attempts but caused regressions on another exposed
matrix. No fallback was frozen from these probes.

Unused fully trained V6 checkpoint epochs were tested under the unchanged
V11 decoder on the exposed 1060 matrix: seed 5002 epoch 80 reached 49/60
with two safety violations; seed 5002 epoch 200 reached 46/60 with four;
seed 5003 epoch 80 reached 47/60 with two. All failed the unchanged gates.
Their outputs and every attempted trajectory remain under the Git-ignored
`outputs/level5/place_repair/dev_v*` directories. The one-time 930-series
confirmation and owner viewer gate remain unused. Level 5.4A is still open.

### Measured grasp-offset diagnosis after V11

A paired replay on the now-exposed `1060262900` heavy-puck/inspection-pad reset
started from identical simulated position and velocity. The V2 scripted
source settled; V11 seed 5003 epoch 20 bounced. At the first opening command,
the source palm command was 0.0486 m above the puck. V11's command was
0.0371 m above it before its 0.003 m release rise, and the fingers still
contacted the puck as it reached the table. This is a command/object geometry
difference on one exposed reset, not an independent performance estimate.

An audit of the 140 successful V2 *training* payloads found median commanded
palm-to-object vertical gaps at first opening of 0.0322 m for large blocks,
0.0366 m for small blocks, 0.0414 m for short cylinders, 0.0314 m for tall
cylinders, 0.0574 m for heavy pucks and 0.0563 m for light pucks. These gaps
vary by object and reset. The paired heavy-puck source gap is below its
training median, so a single median-height release command is not justified.

Four development-only probes tested the mechanism while preserving V11
weights and all terminal gates. Restoring a heavy puck's initial vertical
hand/object gap at release rescued two previously bounced attempts on the
exposed seed-5001 screen, but converted four previously settled attempts to
bounces (6/10 heavy-puck successes versus V11's 8/10). Pausing descent when
that gap shrank caused premature releases on the first two tested puck
attempts; a fixed source-style descent waypoint did the same. Holding the
initial wrist orientation on heavy pucks failed to rescue the first two
seed-5003 bounces and caused a new bounce among the first seven attempts.
These incomplete targeted probes are retained under ignored `dev_v21` through
`dev_v24` directories. None is a frozen recipe, selected bundle or permission
to consume the reserved 930-series confirmation. The next 5.4A repair still
needs a source-grounded motion/release rule that passes complete exposed
matrices before a new frozen candidate is selected.

Saved-state MuJoCo reconstruction refined this diagnosis. The *physical*
palm-to-puck vertical separation remained approximately 0.0286 m in both
paired runs, so command-to-object height alone did not explain the bounce.
V11 continued lateral hand motion at release (about 0.011 m command-to-
calibrated-measured planar lag on this failed reset), while the scripted
source's hand was nearly stationary and centered. The scripted source also
committed to a descent waypoint computed before release; V11 repeatedly
recomputed its waypoint from the moving puck. Its lower commanded release
height on this reset is therefore evidence of the differing control path,
not proof that the physical grasp opened from a lower palm pose.

### V12 adaptive decoder candidate

Further exposed-reset development isolated two narrow causal changes on the
unchanged V6 weights and V11 safety/clearance path. First, for a heavy puck
going to the inspection pad with an entry command/object vertical gap below
0.053 m, hold the first computed descent waypoint instead of chasing the
puck downward. The seven successful *training* pad starts had gaps of
0.0565–0.0586 m; this lower-gap condition is an explicit extrapolation
repair informed by exposed failures. It rescued the same pad bounce on all
three V6 seeds in the exposed 1060 matrix, while applying it to every heavy
puck target or every pad grasp caused regressions. Second, for cuboids over
the left receptacle only, choose between coupled opening steps of 0.00625
and 0.0125 from the pre-action command/object vertical gap and measured
vertical velocity. This uses the previously tested V14 physical rule only
for the target on which its targeted cross-check had no regression.

The combined *targeted* V12 diagnostic preserved every checked 1020 outcome
for all three seeds and changed the 1060 outcomes by +1, +1 and +4 settled
placements for seeds 5001, 5002 and 5003, with no checked safety violation.
Merging those live changed attempts with saved unchanged V11 attempts
predicts all frozen numerical gates pass on that **exposed** 1060 matrix;
the predicted settled fractions are 50/54, 53/58 and 53/58. These are not
fresh selection results or a substitute for complete live evaluation.
The exact V12 policy and selection recipes were checksum locked with a new
1080 all-target selection matrix, the same three predeclared V6 epochs and
the untouched original 930 confirmation. Before using 1080, the independent
saved-state physical audit rejected V12 seed 5003 on the exposed 1060 pad
reset: its nominal success had **15 hand-table-contact frames before first
opening**. Partial V12 validation was stopped. No V12 fresh selection or
confirmation was run. V12 is preserved as a failed candidate.

### V13 lateral-drift recovery candidate

The V12 failure occurred while the short-grasp heavy puck descended with
the hand command more than 3 mm to the negative-y side of the puck. Successful
scripted training pad releases were instead centered or 0.7–1.2 mm to the
positive-y side. A narrowly gated V13 fallback opens at object-target
distance at most 0.024 m, inside the original 0.025 m physical release limit,
only when that negative drift is measured, the object is still held, and it
is moving downward faster than 0.02 m/s. It uses the source-supported 0.0125
coupled finger-opening step. These rules are engineered physical feedback;
V6 weights and the frozen terminal and safety gates are unchanged.

On the six low-offset pad attempts from exposed 1000 and 1060 matrices,
the development fallback had six reported settled placements and six passing
independent saved-state audits, with zero pre-release hand-table contacts.
This threshold was informed by exposed attempts, so complete V13 validation,
fresh 1100 all-target selection, the untouched 930 confirmation, independent
physical audit, and owner viewer acceptance remain mandatory. Level 5.4A
remains open.

Complete V13 validation passed every frozen state gate for all three
predeclared V6 checkpoints (192 rollouts per seed, zero recorded safety
violations). Its separate fresh 1100 all-target selection **failed**:
seed 5001 had 54/60 successes and 54/56 settled nominal placements but
one object-workspace violation; seed 5002 had 51/60 and 51/54 settled with
one object-workspace violation; seed 5003 had 51/60 and 51/59 settled,
below the 90% settling gate. No V13 selection bundle or one-time 930
confirmation was created. The 1100 resets are now exposed development
evidence. Post-selection saved-state comparisons suggest wrist motion during
large-block left-bin transport and short-cylinder/heavy-puck slot descent
as possible causes; those diagnoses are under development and are not
qualification results.

### V14 pre-action wrist candidate

Paired saved-state traces of the exposed 1100 failures showed that wrist
rotation changed the held object contact geometry. The scripted source also
failed the hard large-block left-bin reset by bounce, but avoided the board
violation; its wrist excursion was bounded rather than fixed. Broadly freezing
the learned wrist regressed exposed 1020 and seed-5001 slot outcomes. V14
therefore holds the verified post-setup wrist only for three measured entry
states: a large block sent to the left bin with initial quaternion scalar
above 0.82; a short cylinder sent to slot B whose first learned wrist command
decreases that scalar; or a heavy puck sent to slot A with a command/object
vertical grasp offset below 0.053 m or the same first-command decrease. All
other learned rotation commands remain active. The gate is engineered
feedback, not learned pose inference or vision.

Replaying every changed 1100 attempt across the three predeclared seeds and
merging those attempts with unchanged V13 reports predicts all original
numeric gates pass: seed 5001 is 54/60 with 54/57 settled nominal
placements, seed 5002 is 51/60 with 51/55 settled, and seed 5003 is 54/60
with 54/59 settled. Every changed attempt passed independent saved-state
audit. Cross-checks on the exposed 1000, 1020, 1040 and 1060 affected cells
found no safety violations, but individual placement outcomes varied; the
1000 seed-5002 left-bin block bounced, for example. This is development
evidence only. The V14 recipe and separate unused 1120 all-target selection
are checksum locked; complete validation, fresh selection, one-time 930
confirmation, independent audit and owner viewer acceptance remain required.

Complete V14 validation passed every frozen state gate for all three seeds
(192 rollouts per seed, zero recorded safety violations). Its distinct fresh
1120 all-target selection passed seed 5001 (49/60, 49/54 settled, zero
safety), but rejected seed 5002 for one object-workspace violation (53/60,
53/56 settled) and seed 5003 for 51/59 settled nominal placements (51/60,
zero safety). Thus no three-seed V14 bundle or 930 confirmation exists.
The 1120 resets are now exposed development evidence. A later repair must
preserve these negative reports and use another unused selection set.

### V15 measured-state physical decoder candidate

The 1120 saved states showed the large block rising while still held near the
left-bin margin before its safety violation. V15 issues a bounded 2 mm
inboard and 3 mm upward command during that measured collision state. The
same source audit found successful short-cylinder left-bin release wrist
scalars near 0.734–0.750 and heavy-puck releases near 0.766; several V14
bounces had lower wrist scalars or distinct downward speeds. V15 chooses
between the existing coupled 0.00625 and 0.0125 opening steps using the
pre-action wrist scalar, command/object vertical grasp offset, target and
measured object velocity. It retains V6 weights, V14 wrist guards and all
frozen terminal and safety limits. The new rules are engineered feedback,
not learned vision, hand control or correction from a human operator.

Live V15 changed attempts plus unchanged saved V13/V14 outcomes passed every
frozen numeric gate on both exposed 1100 and 1120 all-target matrices for
all three seeds, with zero safety violations. The 1120 settled fractions
were 51/54, 55/56 and 55/59 for seeds 5001–5003. Every changed attempt
passed independent saved-state audit; the safe timed-out block collision
remains a counted failed attempt. Exact packaged V15 replays matched four
development results. This is development evidence only. The V15 recipe
reserves an unused 1140 all-target selection set and preserves the original
unused 930 confirmation.

Complete V15 validation passed every frozen state gate for the predeclared
V6 epochs 80, 20 and 20 (192 rollouts per seed, zero recorded safety
violations). Success rates were 168/192, 176/192 and 176/192; settled
fractions among nominal placements were 42/44, 1.0 and 44/47. The distinct
1140 all-target screen also passed all three seeds: 53/60, 50/60 and 53/60
successes, with 53/55, 50/55 and 53/58 settled nominal placements and zero
safety violations. The three selected bundles bind the exact V6 weights to
the V15 decoder and selection evidence. The one-time 930 confirmation,
independent audit and owner viewer acceptance remained required.

The reserved 930 confirmation then passed every frozen numerical state gate
for all three seeds: 204/240, 208/240 and 216/240 physical settled
placements, with zero recorded safety violations. Its independent saved-state
audit **rejected all three bundles**. Respectively 8, 2 and 7 distinct
reported successful resets had hand-table contact before finger opening,
replicated across the four render conditions. These attempts remain
successful under the original numerical terminal metric but fail the
additional physical audit. The 930 confirmation is consumed and exposed;
its reports and failed audits remain immutable. V15 is not qualified, no
viewer acceptance is requested, and 5.4A stays open. Subsequent diagnostic
replays on 930 are development evidence only and cannot be called fresh
confirmation. A source-grounded wrist/clearance repair is being diagnosed
against those exposed states; it will need a separate frozen selection and
confirmation matrix if it is carried forward.

An exploratory entry-wrist hold and source-range minimum hand height was
then tested on the now-exposed 930 nominal resets. The full 60-reset
diagnostic passed the original numerical gates for all three seeds (53/60,
51/60 and 53/60, zero recorded safety violations), but independent per-reset
reconstruction still rejected one reported success for seed 5001 and two for
seed 5002. Seed 5003's exposed diagnostic audit passed. Earlier height-only
diagnostics prevented release on several targets; wrist-only diagnostics
left small pre-opening contacts and caused a light-puck premature release.
No V16 recipe, new selection, or fresh confirmation was frozen from these
diagnostics. The contact failure requires a more general physical correction;
5.4A remains open.

### V2 source coverage audit and V3 collection plan

The V2 release has successful training source only for left receptacle,
inspection pad and slot A (18 cells). Slot B occurs only in validation (six
cells); none of the six right-receptacle cells occurs in either source split.
The 30-cell selector and confirmation included those absent right-receptacle
cells. This is a source-coverage finding, not a retroactive change to V2 or
its historical evaluation.

`configs/level5/place_data_v3.yaml` freezes 240 training and 120 validation
resets across all 30 object/target cells in each split. The exact V2 scripted
teacher search and passed 59/60 feasibility screen remain pinned. V3 uses
new reset seeds, episode ids and simulator sessions in disjoint splits; cell
identities are intentionally exposed in both splits. The 95% split, 90%
family and zero live-safety publication gates are unchanged. All failed
attempts remain counted and excluded from source payloads. V3 reserves a new
1240-series confirmation start, which cannot be used before a learned recipe
and selection protocol are frozen. Collection passed 237/240 training and
117/120 validation resets, with zero live safety and all six failures
retained without payloads. The worst families passed 78/80 training and
38/40 validation. Independent saved-state MuJoCo reconstruction passed
354/354 successful trajectories. The immutable archive, checksum and
manifest are under `datasets/level5-place-unassisted-v3*`; its LFS attribute
and archive checksum pass. V3 source alone does not qualify a learned policy.

### V16 contact-safe decoder and fresh qualification protocol

The V15 930 saved-state audit exposed hand-table contact before opening even
when the numerical place gate passed. Exposed 930, 1100 and 1140 replays were
used to design a V16 overlay around the exact, fully trained V6 weights
(epochs 80, 20 and 20). The MLP still supplies orientation, move and release
scores. Engineered controls hold the entry wrist on six observed high-contact
cells, use calibrated measured-object orientation feedback for seeds 5001 and
5002, and enforce a bounded commanded-hand height for the affected cells.
Seed 5001 excludes small-block slot A from orientation feedback after that
feedback caused one new audited contact on exposed 930. Seed 5003 uses a
0.0625 m slot-A floor: 0.063 m failed the original 90% settling gate on
exposed 930. These per-seed controls and exact helper-code digests are frozen
in `place_policy_contact_v16.yaml`; they are engineering, not learned vision
or new learned action dimensions.

The complete exposed 930, 1100 and 1140 nominal matrices, with only the
affected cells replayed after each narrow change, pass the original
numerical gates and independent saved-state audits for all three seeds. The
V16 implementation reproduced one probe rollout array-for-array. This is
development evidence. `place_contact_selection_v16.yaml` reserves fresh
1160-series all-cell selection and 1180-series one-time confirmation resets.
Selection requires both the original numerical gates and a complete
saved-state physical audit before any bundle can be chosen. Full 192-rollout
V16 validation passed every frozen numerical state gate on 192 rollouts per
seed: 180/192, 172/192 and 172/192 successes, zero recorded safety
violations, and settled fractions 1.0, 1.0 and 0.956. The fresh 1160
selection then rejected all three candidates. Seed 5001 had 51/60 successes
and settled only 51/59 nominal placements (86.4%, below 90%); one reported
success also had pre-opening hand-table contact. Seeds 5002 and 5003 passed
the numerical gates at 53/60 and 57/60 with zero recorded safety, but their
independent saved-state audits found one and three reported-success contact
failures. All 180 attempts and physical audits remain in the selection
screens. No V16 bundle was selected; 1180 confirmation is untouched, and no
viewer acceptance or qualification is claimed.

### V17 source-wrist decoder

The V3 all-cell source passed its independent saved-state audit. Successful
first-opening wrist geometry and V16's failed 1160 physical screen informed
a narrow V17 overlay on the exact V6 weights. It holds the setup wrist for
cylinders and pucks while held, raises the large-block inspection-pad command
floor from 0.0625 to 0.063 m, and enables the existing measured-object
orientation feedback for two seed-5003 cells. These are engineered controls;
the learned outputs and V2-trained weights are unchanged. V3 informs the
controls but has not trained these weights.

On exposed 1160 development resets, the unchanged full wrist-hold runs plus
targeted pad and seed-5003 feedback replays predict all original numerical
and saved-state physical gates passing for all three seeds. Four packaged V17
trajectories reproduced corresponding diagnostic actions array-for-array.
These exposed results are development evidence only. The checksum-locked V17
recipes reserved distinct 1260 all-cell selection and 1280 one-time confirmation
resets. Full V17 validation passed every frozen numerical gate for all three
seeds: 172/192, 164/192 and 168/192 successes, with zero recorded safety.
The fresh 1260 selection passed 52/60, 51/60 and 52/60 with every numerical
gate true, zero recorded safety and zero independent physical audit failures
across all 180 saved attempts. Exactly epochs 80, 20 and 20 were selected.
The one-time 1280 confirmation passed all numerical gates for seeds 5001
and 5002 (212/240 each), but seed 5003 failed the unchanged settling gate:
192/240 successes and 192/220 nominal placements settled (87.3%). Its
large-block pad and slot attempts often never opened because the learned
release score remained below the frozen gate. Independent saved-state
reconstruction completed all 720 attempts: each reported success agreed
with physical evidence and no reported-success contact failure was found.
The audit's physical consistency for seed 5003 does not repair its numerical
failure. The 1280 matrix is consumed and exposed, V17 is not qualified,
and owner viewer review remains untriggered. A further repair requires
different frozen selection and confirmation resets.

The frozen V17 commands use seed 5001/epoch 80 and seeds 5002 and
5003/epoch 20. Run each command once per seed with those values:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_place_source_wrist validate --config configs/level5/place_policy_source_wrist_v17.yaml --checkpoint outputs/level5/place_repair/policy_v6/seed_5001/validation_candidates/epoch_0080.pt --output-dir outputs/level5/place_repair/policy_v17_validation/validation_seed5001/epoch_0080
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_place_source_wrist screen --config configs/level5/place_source_wrist_selection_v17.yaml --seed 5001 --epoch 80
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_place_source_wrist select --config configs/level5/place_source_wrist_selection_v17.yaml --seed 5001
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_place_source_wrist confirm --config configs/level5/place_policy_source_wrist_v17.yaml --checkpoint outputs/level5/place_repair/policy_v17_selection/seed_5001/selected.pt --output-dir outputs/level5/place_repair/policy_v17_selection/confirmation_seed5001
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.audit_place_repair --config configs/level5/place_policy_source_wrist_v17.yaml --selection-config configs/level5/place_source_wrist_selection_v17.yaml --seed 5001
```

These directories already contain the local evidence; reproduction must use
new directories and new, separately frozen reset assignments. The 1280
confirmation is one-time evidence and cannot be rerun as fresh evidence.

### V18 bounded geometric release fallback

V17's failed 1280 confirmation exposed a seed-5003 large-block release
failure: several pad/slot attempts reached the region within the original
0.0245 m cuboid release bound, but the learned release score stayed below
the opening threshold. V18 retains the exact V6 learned weights, all V17
controls, target and safety limits, and adds a geometric opening fallback
only for seed 5003's large block at the pad or slots and only within that
existing bound. The fallback uses the already frozen finger opening and
target-specific clearance. This is engineered state feedback, not an extra
learned output. It does not waive the distance, numerical or physical gates.

On the consumed 1280 nominal development replays, four of five distinct
previously unreleased pad/slot attempts became settled; the fifth remained
outside the distance bound and failed. All changed attempts passed
independent saved-state audit with zero pre-opening hand-table contact.
The earlier exposed 1160 and 1260 large-block replays also retained passing
physical audits. The packaged V18 policy matched one 1280 diagnostic
trajectory action-for-action. These are diagnostic results only. The frozen
V18 recipes reserve unused 1300 all-cell selection and 1320 one-time
confirmation. Complete three-seed validation passed every frozen numerical
gate: 172/192, 164/192 and 172/192 successes, zero recorded safety. The
fresh 1300 all-cell screen then rejected every seed on the 90% settling
gate: 51/59, 47/56 and 53/59 nominal placements settled. Seed 5001 also
had one reported-success saved-state physical audit failure; seeds 5002 and
5003 passed their physical audits. No V18 bundle was selected, the 1300
matrix is exposed, and 1320 confirmation remains unused. V18 is not
qualified and owner viewer review is not triggered.

### V19 full-source learned repair

The V2 source used for every V6–V18 learned weight set omitted the right
receptacle entirely. V19 now trains new move, orientation and release scores
from the published, physically audited V3 release. Its 237 training episodes
(25,997 frames) and 117 validation episodes (13,040 frames) each cover all
30 cells with disjoint sessions and episode seeds. The six failed V3 source
attempts remain excluded and counted. Causal features, fixed physical scales,
phase-balanced sampling, 200 epochs for each of three seeds, the V18 bounded
physical decoder, and every original state/safety limit are frozen in a new
checksum-locked recipe. Expert phase is a training label only. Runtime loads
the V6 checkpoint to establish the pinned decoder controls, then replaces
all learned model weights with the V19 checkpoint before any action.

The prior 48-reset validation helper is specific to V2, so V19 freezes a
new 60-reset, 30-cell 1330 validation matrix, followed by distinct unused
1340 selection and 1360 one-time confirmation sets. All three seeds
completed 200 epochs and 20,400 optimizer updates. Offline best epochs
199, 195 and 173 differ from the predeclared closed-loop candidate epochs
20, 80 and 200. All nine fresh 1330 validation reports are complete.
The frozen safety-first ranking chose epochs 20, 20 and 200: 220/240,
220/240 and 216/240 successes, all numerical gates passed, zero recorded
safety. The seed-5003 epochs 20 and 80 failed family and workspace gates;
seed-5001 epoch 80 failed workspace. Those failures remain retained.
The distinct 1340 all-cell selection rejected all three seeds on the unchanged
90% settling gate: 50/59 (84.7%), 51/58 (87.9%) and 50/58 (86.2%) nominal
placements settled. Aggregate successes were 50/60, 51/60 and 50/60, with
zero reported safety violations. Saved-state audit additionally found one
reported-success pre-opening hand-table contact each for seeds 5001 and 5003;
seed 5002 had zero audit failures. No V19 bundle was selected. The 1340
matrix is exposed, the reserved 1360 one-time confirmation remains unused,
and no viewer acceptance is claimed. Slower opening and post-detachment hand
retreat did not repair the shared exposed bounce cases. A bounded descent
during opening improved five of ten targeted seed-5001 failures without
audit contact, but one more nominal success still failed the physical audit;
that diagnostic is not a frozen recipe or qualification result. On the entire
exposed 1340 matrix, the same supported-opening rule regressed previously
successful cells: successes were 47/60, 48/60 and 48/60, settling was
47/58, 48/57 and 48/58, and each seed had one saved-state audit failure.
It is rejected as a general V20 candidate; no fresh qualification reset was
used for it.

The safe V19 source/checkpoint and software checks are:

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_place_full_source --config configs/level5/place_policy_full_source_v19.yaml --seed 5001 --dry-run
shasum -a 256 -c datasets/level5-place-unassisted-v3.tar.gz.sha256
git check-attr filter -- datasets/level5-place-unassisted-v3.tar.gz
conda run --no-capture-output -n dexvision pytest -q tests/test_place_full_source_policy.py tests/test_place_source_capture.py tests/test_place_repair_policy.py
conda run --no-capture-output -n dexvision ruff check dexvision tests
```

The V19 training, validation and selection output directories already contain
sealed local evidence. A new qualification recipe needs new versioned paths
and fresh reset assignments; the untouched 1360 set cannot be used to rescue
the failed V19 selection.

A second exposed development probe replaced the V2 family release waypoint
offsets with per-object medians computed only from the V3 training split.
It yielded 51/60 successes for each seed, but settling remained below 90%
(51/59, 51/58 and 51/58). Seeds 5001 and 5002 each had one reported-success
pre-opening hand-table audit failure; seed 5003 had none. This source-grounded
calibration is insufficient alone and was not frozen for fresh selection.

### V20 training-source object calibration

The V19 decoder used one V2 release waypoint offset per object family. The
V3 training split has 39–40 successful examples per object instance and
shows distinct release offsets for short and tall cylinders and for small
and large blocks. V20 computes the median target-minus-object vector at
first source opening from the audited V3 **training** episodes only and
checks all six frozen vectors against the archive on every recipe load. It
retains the exact V19 learned weights, causal inputs, V18 bounded waypoint
and wrist decoder, and all original state, safety and terminal gates.
During opening, held cylinders descend at most 1 mm per command while more
than 8 mm above target; a small block over slot B keeps the commanded hand
at or above 0.073 m. These are explicit engineered controls, not newly
learned outputs. Slower opening, broad supported descent and simple wrist
hold failed their exposed diagnostics and were not included.

On the exposed 1340 development matrix, the combined controls passed all
frozen numerical gates for the three predeclared V19 weight epochs 20, 20
and 200: 54/60 successes each, settled nominal placements 54/59, 54/58
and 54/58, zero recorded safety and 60/60 independent saved-state audits
per seed. Packaged V20 actions reproduced two diagnostic trajectories
exactly. This is **development evidence only**. The checksum-locked V20
policy and selector reuse the exposed 1330 matrix for complete 240-rollout
policy validation, and reserve distinct fresh 1380 all-cell selection and
1400 one-time confirmation resets. V20 has no qualification or viewer
acceptance yet; the 1360 V19 confirmation remains unused.

All three complete 1330 V20 policy-validation reports passed the original
numerical gates with zero recorded safety: seed 5001 had 220/240 successes
and 94.8% settling; seed 5002 had 212/240 and 91.4%; seed 5003 had
216/240 and 93.1%. The distinct 1380 all-cell selection screens and
independent saved-state audits are in progress. The 1400 one-time
confirmation remains untouched.

The fresh 1380 all-cell selection then rejected V20: seed 5001 passed all
numerical and physical gates (54/60 successes, 54/57 settled, 0/60 audit
failures), but seed 5002 had two reported-success pre-opening hand-table
contacts despite passing numerical gates (52/60 successes, 52/55 settled),
and seed 5003 had one object-workspace violation (49/60 successes, 49/54
settled, 0/60 audit failures). No three-seed bundle was selected. The 1380
matrix is exposed; 1400 one-time confirmation and owner viewer review remain
untouched. V20 is not qualified.

An exposed follow-up tested a 0.070 m large-block left-bin floor and a
0.073 m small-block slot-A floor on both resets of those cells for all
three seeds. The large-block guard kept seed 5001 successful, but seed
5002 still accumulated 273 pre-opening hand-table contact frames on one
reported success and stalled on the other; seed 5003 stalled or released
prematurely. The slot-A guard also introduced contact or premature release
in previously successful attempts. These controls were rejected as a
general repair. No V21 recipe or fresh reset matrix was frozen.

The recurring failure points to a representation/control gap: the V19/V20
195-field causal record exposes object support and hand–object contact, but
does not expose hand–table contact that the independent audit rejects.
The large-block left-bin decoder can descend to the workspace lower z bound
while still held, and a height clamp alone stalled or dropped the object.
A subsequent 5.4A method needs a training-only audit of contact observability,
release pose and active motion variation, followed by a separately frozen
recipe and fresh reset sets. Exposed 1380 outcomes cannot become a new
confirmation or justify weakening the zero-contact gate.

Software verification after V20: full regression **876 passed, one skipped**;
focused V20/V19/place repair tests **13 passed**; the config lock regression
also passed after registering the new V20 selection recipe. Ruff, Git diff
whitespace, V3 archive SHA-256 and its Git LFS filter check passed. These
software checks do not override the failed V20 physical/safety selection.

### V21 copied-state transport preview (negative confirmation)

The V20 failure frames show `rh_forearm` contacting `workcell_table` as a
large block descends toward the left bin; the 195-field policy source does not
carry that contact. V3 successful training releases in that cell used hand
heights from 0.0591 to 0.0746 m with object-to-target distance around
0.022 m. A fixed wrist pose and simple height floor did not solve the exposed
failures. The V21 recipe instead keeps exact V19 weights and the V20 decoder,
then previews the learned transport action and twelve nearby pose actions in a
copy of the current MuJoCo state for the large-block left-bin and small-block
slot-A cells. It prefers the learned action when its predicted physical state
is clear and alternate progress is within 1 mm. It cannot change finger
opening or latched release. The original numerical and independent physical
audit gates remain unchanged.

In a disposable exposed-1380 probe, all twelve attempts in those two cells
settled and passed saved-state audit. Combined with the 48 untouched exposed
attempts per seed, the numerical gates pass at 54/60, 54/60 and 52/60
successes, with zero recorded safety; these are development estimates, not
V21 selection. The checksum-locked V21 implementation adds an explicit action
bounds check and was separately replayed on the seed-5002 large-block contact
case with a passing saved-state audit. Complete 1330 validation passed every
original numerical gate for all three exact V19 candidate epochs: 212/240,
208/240 and 212/240 successes, 93.0%, 91.2% and 93.0% settling, and zero
recorded safety. Fresh 1420 all-cell selection and its per-attempt independent
saved-state audit passed for all three seeds: 55/60, 57/60 and 54/60
successes, every original numerical gate, zero recorded safety and 180/180
passing physical audits. The exact predeclared epochs 20, 20 and 200 are
packaged as selected bundles. One-time 1440 confirmation rejected V21 on the
unchanged 90% settling gate for seeds 5001 and 5002: each settled 204/228
nominal placements (89.47%). Seed 5003 passed the numerical gates at 208/228
settled (91.23%). All three achieved at least 204/240 successes with zero
recorded safety. Independent saved-state reconstruction passed all 720
confirmation attempts, including the counted failures. The 1440 matrix is
exposed. No V21 three-seed qualification or owner viewer review is claimed.

The confirmed bounce resets concentrate in large-block bin/slot and
short-cylinder bin placements. Several opened within the unchanged release
distance around 0.022 m yet later lost stable support. A subsequent 5.4A
recipe needs new selection and one-time confirmation resets. Full software
regression after V21: 880 passed, one skipped; focused V21, V20, V19 and
config-lock tests: 16 passed.

### V22 short-cylinder supported opening (rejected)

V21's exposed 1440 confirmation missed the settled-placement gate by one
distinct bounced reset for seeds 5001 and 5002. Three short-cylinder bin
resets bounced for all three seeds even though opening began inside the
unchanged release distance. A bounded diagnostic doubled only the held
short-cylinder bin opening descent from 1 to 2 mm per action while the object
remains more than 8 mm above target. It converted all nine bounced 1440
attempts to settled placements. All twelve 1440 cylinder-bin attempts and
all twelve 1330 cylinder-bin attempts passed saved-state audit; on the twelve
exposed 1420 selection attempts, the same right-bin cell exchanged which
reset bounced, preserving the per-seed totals. Combined with untouched
attempts, the 1440 diagnostic predicts passing original gates for all three
seeds. These reused outcomes are development only.

V22 freezes that narrow change over the exact V21 recipe and V19 learned
weights, with a new 1460 all-cell selection and 1480 one-time confirmation
matrix. The integrated V22 replay passed one exposed bounce case and its
physical audit. Complete 1330 V22 validation passed every original numerical
gate for all three seeds: 216/240, 212/240 and 216/240 successes; 94.7%,
93.0% and 94.7% settled nominal placements; zero recorded safety. The
full software suite passed 882 tests with one skip. Fresh 1460 selection
rejected V22: seeds 5001 and 5003 missed the unchanged 90% settling gate
at 52/58 (89.66%) and 50/58 (86.21%); seed 5002 passed numerical gates at
53/58 (91.38%). Independent saved-state screening found one reported-success
pre-opening hand-table contact each for seeds 5001 and 5002, in the same
small-block slot-B reset. The common fresh bounces include two short-cylinder
right-bin and two heavy-puck bin resets per seed. No three-seed V22 bundle was
selected. The 1460 matrix is exposed, 1480 confirmation remains unused, and
no viewer review is triggered.

### V23 collision-only preview and bin opening (rejected)

The exposed 1460 failures isolated two issues beyond V22's cylinder change:
heavy-puck bin bounces and small-block slot-B hand-table contact. V23 keeps
the exact V19 learned weights and V20/V21 baseline. In the small-block slot-B
cell, it uses copied-state preview only if the baseline next action predicts
hand-table contact; a safe baseline retains the learned action. Held opening
over either bin descends 3 mm per action for a short cylinder above 8 mm
object clearance, or 1 mm for a heavy puck above 5 mm clearance. No puck
slot-B change was retained because it regressed exposed outcomes.

Disposable changed-attempt probes and independent saved-state audits on
complete exposed 1330, 1420, 1440 and 1460 matrices predicted passing all
original numerical gates for every seed. This is development evidence only.
The checksum-locked V23 policy and selection recipes freeze distinct 1500
all-cell selection and 1520 one-time confirmation boundaries. Two integrated
replays (prior slot-B contact and puck-bin bounce cells) passed saved-state
audit. Complete 1330 V23 validation passed all original gates for exact
V19 epochs 20, 20 and 200: 204/240, 200/240 and 204/240 successes,
92.7%, 90.9% and 92.7% settled nominal placements, and zero recorded
safety. Full software regression passed 885 tests with one skip; Ruff,
recipe checksums and diff whitespace checks pass. Fresh 1500 all-cell
selection passed every numerical gate at 54/60, 54/60 and 53/60 successes,
with zero recorded safety and 98.2%, 98.2% and 96.4% settling. Independent
saved-state audit passed 179/180 attempts. Seed 5002's reported-success
large-block/right-bin reset `1500261100` had 11 pre-opening hand-table
contact samples despite its numerical success; it is a physical gate failure.
V23 is rejected, the 1500 matrix is exposed, and 1520 one-time confirmation
remains unused. A disposable collision-only preview removed contact but made
this reset time out, so that broad extension is not frozen as a repair. No
V23 bundle or manual acceptance is claimed.

### V24 verified setup wrist hold (rejected)

Re-reading the 5.2A and 5.3A repairs reinforces three method requirements:
audit causal features and physical frames, make demonstrated direction and
posture constraints explicit in the decoder, and select with complete
closed-loop validation before fresh confirmation. The passing pick repair
also used a verified scripted entry and source-grounded wrist semantics.
Those lessons do not justify changing place release/safety gates.

The V23 failed large-block/right-bin reset starts from identical scripted
setup for all policies. Seed 5002 then rotates its wrist in the opposite
direction from the other policies: at transport action 110 the held object's
vertical offset from the hand is about -3 mm, versus -16 to -18 mm for the
other seeds. It must descend to a 38 mm hand command before reaching the
unchanged release bound; hand-table contact begins at action 148. Nearby
rotation/position pulses and low collision previews did not repair that
audited success. A broad collision-only preview caused timeouts and regressed
the worst-family gate for seed 5003, so it was rejected.

V24 holds the exact verified setup command quaternion during transport only
for `block_large` to `return_bin_right`. It retains V19 learned weights and
V23 translation, release and preview controls. The wrist hold is engineered,
not learned; other cells keep their existing rotation semantics. All six
exposed 1500 right-bin attempts settle and pass saved-state audit, including
the V23 physical failure. Six 1330 right-bin validation probes also settle and
pass audit. All six 1460 right-bin probes pass audit, with one counted bounce
per seed. These are development results. One integrated frozen V24 replay
matches the passing probe's actions and qpos exactly and passes physical
audit. The locked recipes reserve fresh 1540 selection and 1560 one-time
confirmation. Focused tests pass 16/16; full regression passes 888 tests with
one skip. Ruff, recipe checksums and whitespace checks pass. Complete V24
validation passed every original numerical gate at 204/240, 200/240 and
204/240 successes, 92.7%, 90.9% and 92.7% settling, and zero recorded
safety. Fresh 1540 selection passed every numerical gate at 53/60, 53/60
and 51/60 successes, zero recorded safety and 96.4%, 94.6% and 92.7%
settling. Independent physical audit passed 179/180 attempts. Seed 5002's
large-block/inspection-pad reset `1540260900` had one pre-opening table
contact sample; V24 is rejected and 1560 confirmation remains unused.
The 1540 matrix is exposed. A pad wrist hold alone preserves the contact.
Saved causal observations show object distance already inside the unchanged
0.0245 m bound at action 107, but the learned gate delays opening to 109.
A disposable geometric trigger opens at 107 and produces an audited settled
success with zero pre-opening contact. This is development evidence only.
5.4A remains open.

### V25 bounded pad opening timing (complete with owner acceptance)

V25 retains the exact V19 learned weights and V24 runtime bundle, and adds
an engineered geometric release trigger only for `block_large` on
`inspection_pad`. It opens at the first held measured distance within the
unchanged 0.0245 m cuboid bound, using the existing opening rate and release
rise. This opens the V24 physical-failure reset at action 107 instead of 109,
before its table contact. All six exposed 1540 pad attempts settle and pass
independent physical audit with zero pre-opening contact. Six 1330 pad probes
preserve one timeout outside the bound and one audited success per seed.
These are development results. The integrated frozen V25 replay matches the
passing diagnostic's actions and qpos exactly and passes its physical audit.
Focused tests pass 18/18; Ruff passes. New 1580 selection and 1600 one-time
confirmation resets were frozen before use. Full regression passed 890 tests
with one skip. Complete V25 validation passed every numerical gate at
204/240, 200/240 and 204/240 successes, 92.7%, 90.9% and 92.7%
settling, and zero recorded safety. Fresh 1580 selection passed every
numerical and physical gate at 53/60, 53/60 and 52/60 successes, zero
recorded safety and 98.1%, 96.4% and 92.9% settling. All 180 independent
saved-state audits pass. Exactly epochs 20, 20 and 200 are packaged;
their model tensors and manifests match the frozen V19 candidates. Reserved
1600 confirmations were verified unused before launch and ran once per seed.
All three pass every original numerical gate and every independent saved-state
physical audit. The selected weights and full evidence stay in local ignored
outputs. Evidence hashes, metrics and changed-file inventory are recorded in
`docs/level5_place_repair_verification_v25.json`.

| Seed | Selected epoch | Confirmed successes | Settled / nominally placed | Worst family | Premature release | Physical audits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5001 | 20 | 196/240 (81.7%) | 196/208 (94.2%) | 65% | 20/240 (8.33%) | 240/240 |
| 5002 | 20 | 204/240 (85.0%) | 204/216 (94.4%) | 75% | 20/240 (8.33%) | 240/240 |
| 5003 | 200 | 192/240 (80.0%) | 192/208 (92.3%) | 65% | 20/240 (8.33%) | 240/240 |

Recorded safety violations are zero for every seed. Action jerk is
0.0025731, 0.0023820 and 0.0025274. All 720 attempts, including failures,
pass physical/report consistency audits. Recipe/source pins, artifact indexes,
exact candidate tensors/manifests, one-time markers and viewer prerequisites
verify. Each seed covers 60 unique resets across 30 cells, each run under four
conditions. This is state/physics evidence; the conditions do not qualify RGB
perception. Previously exposed families are not unseen. Some individual cells
have zero success, and 44, 36 and 48 failures remain counted. Post-opening
hand-table contact is permitted by the frozen audit; passing the audit does
not claim zero contact throughout release. The MLP learns move, orientation
and release scores; geometry, physical scales, fixed joints, wrist hold,
collision preview and the bounded pad opening trigger are engineered.

The owner confirmed the prescribed three-target-type seed-5001 saved-state
review on October 5, 2026 with "Passed". This is owner-reported acceptance,
not an agent visual observation or viewer-log completion claim.
`docs/level5_place_acceptance.json` binds that statement to the exact recipes,
three selected bundles, numerical reports and physical audits. Original
automated receipts and verification retain their historical pending status.
Last Completed is now 5.4A, Next Target is 5.5, and 5.5 has not started.

V25 command reference (macOS, from the repository root; output directories
are write-once, so these identify the frozen runs rather than overwrite them):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_place_pad_release validate --config configs/level5/place_policy_pad_release_v25.yaml --checkpoint outputs/level5/place_repair/policy_v19/seed_5001/validation_candidates/epoch_0020.pt --output-dir outputs/level5/place_repair/policy_v25_validation/validation_seed5001/epoch_0020
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_place_pad_release validate --config configs/level5/place_policy_pad_release_v25.yaml --checkpoint outputs/level5/place_repair/policy_v19/seed_5002/validation_candidates/epoch_0020.pt --output-dir outputs/level5/place_repair/policy_v25_validation/validation_seed5002/epoch_0020
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_place_pad_release validate --config configs/level5/place_policy_pad_release_v25.yaml --checkpoint outputs/level5/place_repair/policy_v19/seed_5003/validation_candidates/epoch_0200.pt --output-dir outputs/level5/place_repair/policy_v25_validation/validation_seed5003/epoch_0200
for seed in 5001 5002 5003; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_place_pad_release screen --config configs/level5/place_pad_release_selection_v25.yaml --seed "$seed"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_place_pad_release select --config configs/level5/place_pad_release_selection_v25.yaml --seed "$seed"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_place_pad_release confirm --config configs/level5/place_policy_pad_release_v25.yaml --checkpoint "outputs/level5/place_repair/policy_v25_selection/seed_${seed}/selected.pt" --output-dir "outputs/level5/place_repair/policy_v25_selection/confirmation_seed${seed}"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.audit_place_repair --config configs/level5/place_policy_pad_release_v25.yaml --selection-config configs/level5/place_pad_release_selection_v25.yaml --seed "$seed"
done
conda run --no-capture-output -n dexvision python -m pytest -q tests/test_place_pad_release_policy.py tests/test_place_wrist_hold_policy.py tests/test_place_bin_opening_policy.py tests/test_level5_learning_plan.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m pytest -q
conda run --no-capture-output -n dexvision python -m ruff check .
git diff --check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision mjpython -m dexvision.apps.review_place_repair --config configs/level5/place_policy_pad_release_v25.yaml --selection-config configs/level5/place_pad_release_selection_v25.yaml --seed 5001
```

All three passing confirmations and complete physical audits now exist, and
the viewer prerequisites were verified before owner review. The last
command above identifies that review. It replays saved states and shows one
success for each planar-zone, placement-slot and receptacle target.

**Pass:** all three clips finish; the correct initially held object visibly
travels to the target, releases inside the frozen tolerance, remains stable
at the end, and agrees with the saved result. **Fail:** wrong object, missing
placement, pre-opening hand-table contact, unstable or bounced final placement,
or scene/report mismatch. Cancelling before all clips finish is not a pass.
Owner confirmation required by AGENTS.md has been received and recorded.
