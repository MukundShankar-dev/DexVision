# 5.6 separate development repair

Checkpoint 5.6 completed on October 16, 2026 after owner manual acceptance;
Last Completed is 5.6; Next Target is 5.7, unstarted. See
`docs/level5_perception_acceptance.json`. Earlier sections below preserve their
then-current development status. The historical baseline V2 test, fresh
confirmation V2 and fresh confirmation V3 are consumed and permanently excluded from fitting, validation selection and repeat testing.
Their source data, protocols, results and failed attempts remain preserved.

The owner explicitly authorized a separately frozen train/validation dataset,
repair of failing gates, and another fresh confirmation, stopping before 5.7.
The new development protocol freezes 128 TRAIN and 32 validation whole simulator
sessions, with separate seeds, episode ids and session ids, before generation.
It uses the same exposed four source cells and four calibrated render conditions;
this is not unseen-family, unseen-cell or real-camera generalization.

All generated frames and source failures remain counted. No saved trajectory
replay, human control, object pose write after reset, outcome filtering or
replacement is permitted. TRAIN masks/poses provide supervision only; validation
labels select candidates after RGB-only inference. Consumed indexes are read only
for seed and pixel-digest collision audits, never to decode their images/labels.

The collector preserves explicit actuator order and measures evaluation-only
truth on copied post-forward MuJoCo data, leaving live dynamics unchanged.
Old frozen collector/trainer/evaluator modules are preserved; new modules provide
the independently versioned continuation. The shared partial typed estimate
schema and fail-closed accepted-control boundary remain in force. Missing full
orientation, semantic identity, contact/relations, fixture depth and declared
robot telemetry remain unsupported, not filled with truth or neutral values.
No accepted control bundle or runtime is modified or qualified.

## Procedure and commands

Commands use the `dexvision` environment and repository root. Paths are write-once;
commands identify the actual runs and must not overwrite existing evidence.

```bash
conda activate dexvision
python -m dexvision.apps.repair_perception freeze-development --config configs/level5/perception_development_v1.yaml --output-dir outputs/level5/perception_development_v1/protocol
python -m dexvision.apps.repair_perception collect --protocol outputs/level5/perception_development_v1/protocol/protocol.json --output-dir outputs/level5/perception_development_v1/dataset
```

The initial freeze/collection were invoked through the corresponding core Python
functions before the CLI was added. On macOS, collection needs a graphics context.

Freeze the recipe, input/code digests, full candidate budget, feature and target
scales, primary weighting, decoder, ranking and fixed qualification limits before
fitting. Preserve every candidate report. Deterministic fits require no stochastic
training seeds. Select on development validation only. A failing validation method
must not consume a confirmation set.

After passing validation, freeze the selected model, recipe and selection report
into another independent test-only protocol before generating confirmation pixels.
Use new disjoint reset seeds; preserve every scheduled reset/frame. Evaluate this
protocol exactly once, reserving the attempt before image/label decoding. Publish
new immutable LFS archives with checksums/manifests/protocols, never replacements.

Original aggregate and per-core-class recall at IoU 0.5 must remain >=95%; median
position <=10 mm, p95 <=25 mm, identity <=1/1,000, and synthetic stale/ambiguity
rejection 100%. The supplemental primary-target gates and primary-class recall
must also pass. Missing/rejected instances stay in eligible denominators;
conditional hypothesis errors, uncertified coverage, availability and latency
remain separately disclosed. Generic visual gates do not qualify millimetre
fixture predicates or accepted 34 ms skill-control loops.

## Manual acceptance

Fresh contact-sheet and temporal-video owner review is required only after all
numerical gates pass. Earlier V8 validation review is already accepted and will
not be requested again. The owner confirmed the new held-out review passed
on October 16, 2026; no manual gate remains for 5.6. Do not start 5.7.

## Development V1 and pixel-center amendment

V1 contains 8,324 RGB frames: 6,703 TRAIN and 1,621 validation. All 160 source
attempts remain counted. The immutable archive's 17,133 members and external
checksum were read back successfully. The separate source/schema audit covers
41,214 typed source snapshots and 66,592 rendered body origins: maximum source
versus render position mismatch is exactly 0 m; rotation matrix difference is
2.34e-15 and calibrated camera/world roundtrip error is 5.07e-16 m.

V9 freezes 24 detector/tracker candidates and 21 metric candidates on V1.
The completed detector matrix fails primary-cylinder recall (best worst-primary
recall 87.68%). These results and the full running experiment are preserved;
failing detection cannot be rescued by conditional pose accuracy.

A deterministic TRAIN-only color/miss audit found that RGB multisampling blends
small visible object pixels with the black hand, while the integer instance
mask describes pixel-center occupancy. An A/B render of the same saved TRAIN
state, camera and lighting changes a marked cylinder pixel from RGB [7,7,7] to
[13,45,21] when offscreen samples change from 4 to 0. The cylinder instance mask,
box [240,242,263,259] and 73-pixel visibility count remain identical. This is a
raster/input alignment amendment, not a reason to modify/re-score either consumed
confirmation or weaken a gate.

`perception_development_v2.yaml` freezes a re-render of the **same** V1 TRAIN/
validation source sessions at zero RGB multisampling, matching integer mask
pixel centers. It creates a new release and preserves V1. It does not claim
another 160 independent physics sessions. Source physics, calibration, splits,
visible-instance labels, source failures and eligible denominators are retained;
no learned controller, resampling or consumed-test image/label is used.

Before any new V2 validation inference, `training_plan_v10.json` freezes four
small-component/grouping/tracker candidates and all 21 original metric fits.
On 48 previously missed TRAIN diagnostic examples, the unchanged detector now
matches 40 at IoU >=0.5; median IoU rises from 0.375 to 1.0. Every examined mask
file is byte-identical. The new method's numerical gates still require complete
validation and a new independent confirmation, using the identical pixel-center
raster convention. This does not establish performance on antialiased or real
camera inputs. Full audit sources/results are preserved under the development
output directories.

```bash
python -m dexvision.apps.render_perception_amendment freeze --config configs/level5/perception_development_v2.yaml --output-dir outputs/level5/perception_development_v2/protocol
python -m dexvision.apps.render_perception_amendment collect --protocol outputs/level5/perception_development_v2/protocol/protocol.json --output-dir outputs/level5/perception_development_v2/dataset
```

## Prior-stage adjustments carried forward

The 5.4A place repair and 5.5 push/press results were reviewed alongside the
training guidance and 5.2A/5.6 contracts. The place saved-state audit requires
physically recomputed post-integration evidence and correct body origins;
this collector reads copied post-forward data without modifying live dynamics.
Robot/mocap replay parity is not asserted from qpos/qvel alone, and no inferred
robot telemetry is fabricated. Object-origin parity is independently measured.
The push repair demonstrates that a consumed failed confirmation cannot select
its repair and that a full budget/final epoch does not itself qualify a model.
Here new TRAIN/validation evidence, exact feature/scaling/decoder bindings,
complete candidate matrices, retained failures and a separate once-only fresh
confirmation enforce the same separation.

Their wrist holds, collision previews, contact phases and skill-specific entry
controllers are not imported into RGB inference. Accepted pick/place later-source
geometry and standalone reach/push control entries remain unsupported coverage
strata. All five accepted control bundles retain their original state-grounded
qualification and block inferred execution on missing fields. This work adds
no runtime rollout, registry admission or 5.7 integration.

V9 completed its entire 24-detector/21-metric budget and exits 2 as a preserved
negative result: aggregate recall 98.11%, primary recall 91.27%, best conditional
primary position p95 3.84 mm, zero identity switches. Primary-cylinder recall
87.68% still fails. Exact bindings are in
`docs/level5_perception_development_v9_failed.json`; no V9 confirmation ran.

The V2 render amendment's full archive readback verifies every one of 17,133
members. All 8,324 annotation records, mask hashes, conditions/parameters,
source episode/session/split/frame/timestamps and every source payload match
V1 exactly. The archive checksum is
`630244ccb8b38437bc43f49a6a8f49db0d228ce447330feb1c0d67aefa869072`.

The final software suite passes **1,061 tests, one skipped in 1,592.94 s**;
focused checks pass 157. Ruff, whitespace, CLI help, frozen V8/consumed V2 pins,
and completed/future progress-section preservation checks pass. Software checks
are distinct from numerical method qualification and owner manual acceptance.


## Expanded development V3 and context repair (October 15)

Following V11's failed fresh V3 confirmation, all three consumed sets stay closed.
Development V3 freezes 256 TRAIN and 128 validation independent simulator sessions
before generation, using disjoint 96560000-/96570000-series seeds. It retains all
384 attempts and 19,557 frames (12,984 TRAIN, 6,573 validation), with no outcome
filtering. There are 186 expert-error endings. The raw final task flags differ
(282 false) because, for example, a completed press-and-release can end released;
neither diagnostic is a skill qualification rate. No source failure is replaced.
All old/cross-split pixel-digest collisions are zero. These remain exposed families
and cells and pixel-center simulator RGB, with no new real-camera claim.

V12 freezes context370 (causal component34 + global144 + local192 RGB-derived
intensity/occupancy features) and predictive causal image association. All four
full weighted CART fits pass the original aggregate/primary position gates but
fail stronger validation selection: one primary switch and large rare cuboid
place errors. `level5_perception_context_v12_failed.json` preserves the result.
Selection additionally requires zero primary switches, >=99% pose coverage,
>=8 visible instances each for cuboid/cylinder transport/place, and original
position/recall limits in each supported primary class/phase. This prevents
static-object abundance from masking late manipulation failures.

V13 freezes equal primary/nonprimary role × phase totals and equal episode
weight within each group. Phase supervises loss only; it never enters inference.
Widened fragment union fixes a missed cuboid and identity switch, but merges
nearby pucks when applied to every class. The complete candidate matrix is kept.

V14's neural attempt stops at startup because the legacy detector ancestor lacks
the context-engine key. No fit or confirmation occurred; the exact failed source
snapshot and startup receipt are preserved. V15 corrects only that loader and
freezes two full deterministic 1,200-epoch CPU fits (seeds 3/7, widths 128/64,
ReLU, TRAIN-only input/target normalization, explicit UV/height decoder). It
shares visual context across classes, with a visual-class one-hot, and uses
24-pixel fragment union for cuboids only, retaining 12 pixels for other classes.
All weights, exact transforms and full histories are packaged; selection remains
complete validation under unchanged strict gates. No new confirmation is created
until those gates pass. This work stays in 5.6; all five accepted controls still
reject the incomplete inferred inputs.

```bash
python -m dexvision.apps.repair_perception freeze-development --config configs/level5/perception_development_v3.yaml --output-dir outputs/level5/perception_development_v3/protocol
python -m dexvision.apps.render_perception_amendment collect --protocol outputs/level5/perception_development_v3/protocol/protocol.json --output-dir outputs/level5/perception_development_v3/dataset
python -m dexvision.apps.context_perception train --config configs/level5/perception_repair_v12.yaml
python -m dexvision.apps.context_phase_perception --config configs/level5/perception_repair_v13.yaml
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m dexvision.apps.neural_perception_v15 train --config configs/level5/perception_repair_v15.yaml
```


Both V15 candidates pass the original aggregate/core/primary gates, with 100%
recall and zero primary identity switches, but fail the stronger rare cuboid
transport/place selection limits (p95 31–40 mm). Both full 1,200-epoch runs,
histories, matrices and exact source are preserved in
`level5_perception_neural_v15_failed.json` and the V15 output/source directories.
No confirmation is opened for V15.

V16 freezes `level5/rgb-motion706-v1`: current context370 plus 336 global/local
RGB context rates computed from the previous acquisition on the same visual
track and divided by the actual elapsed seconds. Availability remains explicit;
missing/expired history has numeric padding and no claim of measured zero
physical velocity. Resets clear history. No future pixels, phase, task/source id,
semantic identity, control state or truth enters inference. The same features,
TRAIN-only transforms, learned UV/height targets, camera/ray decoder and causal
tracker are used in fitting, validation and any future confirmation.
Two full deterministic 1,800-epoch fits (seeds 3/7) are frozen before training;
complete strict validation selects only their final epochs. Portable arrays are
compiled once without modifying the hash-bound JSON model. Exact TRAIN arrays
are retained for reproducibility; they contain no validation/test examples.

The expanded development source audit validates all 96,898 typed snapshots and
156,456 rendered body origins across all 19,557 frames. Source/render position
mismatch is exactly zero; maximum rotation matrix delta is 2.44e-15 and camera/
world roundtrip error is 2.30e-16 m. The full 40,271-file archive readback,
archive checksum and LFS attribute pass. These are evaluation-only oracles, not
RGB inference inputs. Source-audit receipt/code are preserved in
`outputs/level5/perception_development_v3/schema_audit/`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m dexvision.apps.motion_perception train --config configs/level5/perception_repair_v16.yaml
```


### Selection metadata and prospective confirmation freeze

The frozen neural report writer omitted the redundant `recipe_input_digests`
field required by the strict common model loader. The recipe SHA already binds
those exact inputs. `bind_perception_selection` adds that field after checking
the recipe, code and selected-weight bindings. It retains the original report
as `selection_before_binding.json`, verifies unchanged weights/ranking/metrics,
records both hashes and its own code hash, and refreshes the artifact index.
It performs no fitting, re-selection or test evaluation. Confirmation freezes
this finalizer, original/final reports and binding receipt as additional pins.

```bash
python -m dexvision.apps.bind_perception_selection --config configs/level5/perception_repair_v16.yaml
```

Before any V4 source/pixels are generated, `confirmation_v4_plan.json` replaces
the earlier provisional 128-case plan with 256 cases (16 resets per source cell /
condition). This reduces unsupported rare visible cuboid-phase coverage risk.
Every scheduled attempt counts, and zero primary switches and all strict gates
remain unchanged. The owner reviews fixed rep0/1 clips across all16 cells and
conditions plus worst primary-pose episodes; all256 videos remain preserved.
The actual selected-model protocol is frozen only after full strict validation
passes. No V4 test is open at the time of this plan.


## Targeted cuboid coverage amendment V4 / V17

Both V16 full 1,800-epoch fits still fail the stronger rare cuboid phase gates,
despite passing original/core/primary gates with 100% recall and zero switches.
`level5_perception_motion_v16_failed.json` preserves the full result and the
unchanged-weight metadata correction. No fresh confirmation was opened.
A TRAIN-only coverage audit found just 26 cuboid place instances from five
source episodes, 27 transport instances from six, and three stabilization
instances from two. Additional neural complexity did not resolve this scarcity.

Development V4 therefore freezes 256 new TRAIN and 128 validation cuboid source
sessions across all four original conditions (98560000-/98570000-series seeds),
with all attempts, failures and frames retained. This is targeted allocation,
not outcome filtering or replacement. Existing releases and every consumed test
remain unchanged. The V17 plan is frozen before opening the new validation data.
It preserves motion706, detector/tracker, decoder and loss, uses both complete
validation parents, and tests added TRAIN coverage with two complete 3,000-epoch
fits (seeds3/7), selecting only final epochs under unchanged strict gates.

A derived union copies immutable parent RGB/masks/annotations and preserves
source/split ownership; it does not claim more new sessions than its parents.
Raw sources stay in their parent releases. The exact V16 TRAIN feature cache
is reused only after checking feature/code/scales/detector equality. New cuboid
TRAIN examples add supervision; other classes keep original TRAIN arrays.
Cuboid parent masses follow primary TRAIN allocation256:new /64:old (80%/20%),
retaining role/phase/episode weights within each parent. Source outcomes never
weight or remove examples. All new validation annotations remain scored.

```bash
python -m dexvision.apps.repair_perception freeze-development --config configs/level5/perception_development_v4.yaml --output-dir outputs/level5/perception_development_v4/protocol
python -m dexvision.apps.render_perception_amendment collect --protocol outputs/level5/perception_development_v4/protocol/protocol.json --output-dir outputs/level5/perception_development_v4/dataset
python -m dexvision.apps.union_perception_development --parent outputs/level5/perception_development_v3/protocol/protocol.json outputs/level5/perception_development_v3/dataset --parent outputs/level5/perception_development_v4/protocol/protocol.json outputs/level5/perception_development_v4/dataset --output-dir outputs/level5/perception_training_union_v17
```


### TRAIN scale audit and pre-fit V18 amendment

Before V17 fitting or opening the new validation data, a TRAIN-cache audit found
global raw-target standard deviations of [0.0332, 0.1167, 0.6738]. Empirical
normalization gives world-height changes substantially less loss than comparable
pixel-offset metric errors. Class rest heights and parked nonprimary objects
contribute to that variation; negative parked-body heights are preserved data,
not clipped or relabelled. The unstarted V17 plan/code remain unchanged, with a
supersession receipt. No V17 model was fitted and no confirmation was opened.

V18 preserves the RGB-motion706 input transform, architecture, causal tracker
and inference decoder. It freezes per-class weighted TRAIN target centers and
fixed output scales [0.25,0.25,0.1] in raw UVoffset/32px and worldZ/.1m units
(8px,8px,10mm). These exact inverses are packaged per class in selected JSON.
Training minimizes weighted world-body-origin XYZ squared error /0.01m, using
a differentiable equivalent of the same calibrated ray decoder. This fixes the
objective's physical units without changing the labels, camera or inference
inputs. The network receives neither phase nor truth at inference; phase remains
TRAIN weight supervision. Both full3,000-epoch fits and complete union validation are
still required, with unchanged strict gates and no early stopping.

Unit tests compare metric decoding with the frozen NumPy inference decoder,
verify gradients/parallel-ray rejection, and check deterministic full-budget
fitting, weighted class centers and fixed physical output scales. All pass.
`training_plan_v18.json` and its checksum freeze the objective and exact code
before the new validation set is opened. This applies the prior-stage lesson
to audit frames/scales and the engineered decoder before another test attempt.

```bash
python -m dexvision.apps.union_perception_development --parent outputs/level5/perception_development_v3/protocol/protocol.json outputs/level5/perception_development_v3/dataset --parent outputs/level5/perception_development_v4/protocol/protocol.json outputs/level5/perception_development_v4/dataset --output-dir outputs/level5/perception_training_union_v18
python -m dexvision.apps.coverage_metric_perception --config configs/level5/perception_repair_v18.yaml
```


Development V4 collection is complete: all384 cases, 23,688 RGB frames (15,890
TRAIN /7,798 validation), zero historical/cross-split pixel collisions, all352
expert-error endings retained. The archive's48,533 files and checksum read back
successfully; its LFS attribute and external protocol checksum pass. V4 source/
render position mismatch is exactly0m across189,504 body origins; all117,668
typed source snapshots validate. The derived union has43,245 frames and768
sessions (512 TRAIN /256 validation), with no new session claim or test parent.
The V18 recipe is now frozen before fitting both full3,000-epoch runs.


### V19 cuboid-only harvest contract correction

V18 stopped before any model epoch: its shared harvester required all four
TRAIN classes, while the amendment intentionally supplies cuboid supervision.
No weights, candidate score or confirmation was produced. The exact startup
failure and source snapshot are preserved. V19 adds a dedicated cuboid-only
harvester and retains the other three classes' exact prior TRAIN arrays.
Source data, features, objective, architecture, 80%/20% parent weighting,
full two-run3,000-epoch budget and strict gates are unchanged. The original
V18 recipe/code remain immutable; V19 is separately frozen before fitting.

The new regression test passes with cube-only input. A real-data smoke uses
only the first16 TRAIN frames, with no fitting, validation or test selection:
15 visible cube examples yield motion706 features, positive weights, and
maximum float32 training /float64 inference decoder discrepancy of7.71e-8m.
Its receipt is `perception_development_v4/cuboid_metric_smoke_v19.json`.

```bash
python -m dexvision.apps.coverage_metric19_perception --config configs/level5/perception_repair_v19.yaml
```

## V4 confirmation failure and V20 stabilization coverage repair

V19 passed strict validation but its single fresh V4 evaluation fails the
unchanged cuboid stabilization median gate (74visible,13.801mm>10mm). Primary
recall100%,median0.784mm,p953.137mm,zeroID; all original/core/primary gates pass.
The test is consumed, preserved, and excluded from all fitting/selection/retest.
No test images or individual coordinates were inspected to tune this repair.

Independent TRAIN/validation source coverage audit finds only8TRAIN/4VAL visible
stabilization frames in the sparse stride5 export. Frozen derived phase-windowV5
retains every recorded stabilization query, regardless of visibility/outcome,
with15/10/5 preceding control-frame RGB context. 5,131 queries,20,524 rows;
3,408TRAIN/1,723VAL queries preserve original whole-source split ownership.
This is denser sampling of developmentV3/V4, not new independent sessions.

V20 freezes one complete deterministic seed3/3,000-epoch128/64 fit, the exact
metric loss and same RGB706 inference/tracker/decoder. Cuboid TRAIN mass is
50%prior cache/50%new queries; noncuboid arrays unchanged. Only query labels
fit; physical-source weighting prevents virtual-window count from reweighting
episodes. Original full-sequence strict validation plus additional query
coverage/recall/pose gates must pass before another fresh confirmation.
Exact plan:`outputs/level5/perception_development_v5/training_plan_v20.json`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.phase_window_perception collect --protocol outputs/level5/perception_development_v5/protocol/protocol.json --output-dir outputs/level5/perception_development_v5/dataset
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.phase_repair_perception --config configs/level5/perception_repair_v20.yaml
```

No5.6 completion or5.7 work is claimed.

V5 collection/archive/source checks pass:20,524rows,5,131queries,40TRAIN/20VAL
visible primary stabilization samples from10/6physical sources. All145,061
typed source snapshots parse;164,192body-origin comparisons have zero position
delta,2.22e-15rotation-matrix and2.29e-16camera/world roundtrip differences.
Full archive readback verifies24,247payload files; SHA-256
`156093685697d9b447cb9d9cb16dbae8cc2f188c8db5ae15f3d0b5a4220a9eb1`.

Before V20 validation or any new confirmation, planV21 freezes the checkpoint's
three-seed requirement: seed3's exact V20 full-budget fit is retained once;
seeds7/11 each receive3,000epochs on the exact same already-harvested TRAIN
arrays, without another data merge or change to weights/transforms. Every final
epoch receives unchanged full-sequence plus dense-query validation. Selection
requires both strict gates, then worst supported phase p95/primary p95/frozen
order. Exact cache/recipe/code pins and a full-budget reuse guard bind seed3;
invalid cache/transform, shortened budget, test exposure or duplicate seeds
reject in tests. This selects a baseline among three fits and does not claim
that every initialization or all accepted controls are qualified.

V20 completes its full3,000epochs. Dense stabilization validation passes all20
visible queries /6physical validation sources:median2.820mm,p957.441mm,100%
recall/pose coverage. Original/core/primary gates pass, but full strict validation
fails cuboid/lift p9527.651mm>25mm on23visible instances. No confirmation
opens. `docs/level5_perception_phase_v20_failed.json` preserves the negative
receipt; all data/weights/validation remain immutable. Pre-frozen V21 completes
seeds7/11 on the exact same TRAIN cache and retains seed3 once, with unchanged
full-sequence/query gates and no test data.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.phase_multiseed_perception --config configs/level5/perception_repair_v21.yaml
```

## V21 negative budget and V22 phase-balanced loss

All three V21 fits complete3,000epochs and all sequence/query validation.
Dense stabilization passes for every candidate; lift p95 is27.65/30.02/32.69mm,
so none passes25mm. No fresh confirmation opens. All weights/receipts remain
immutable;`docs/level5_perception_phase_v21_failed.json` binds the negatives.

TRAIN-only mass audit finds40new stabilization queries carrying50%cuboid mass
against15,074old examples. V22 corrects this allocation by balancing each
role/phase (primary4:other1) and each physical source inside the group across
original+query examples. Complete TRAIN visible-label counts equal cached row
counts; every cached target decodes to its corresponding original body position
in frozen harvest order. No label skipping/reassignment, new RGB inference or
feature/target alteration occurs. All four classes' features/targets match the
prior cache exactly. New query mass is7.5329%; primary phases each8.6957%.
Noncuboid arrays and all inference transforms/tracker/decoder remain unchanged.

Frozen harvest/config/source pins bind recovered source/phase metadata before
new fitting. All three seeds3/7/11 receive3,000epochs and both unchanged full
sequence and dense-query gates. This is TRAIN loss weighting only; physical
phase never enters RGB inference. Original development/consumed releases remain
immutable. The phase-window release is unchanged; no new physical sessions.

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.balanced_phase_perception harvest --config configs/level5/perception_harvest_v22.yaml --stream base --output outputs/level5/perception_development_v5/balanced_v22/base.npz
conda run --no-capture-output -n dexvision python -m dexvision.apps.balanced_phase_perception harvest --config configs/level5/perception_harvest_v22.yaml --stream phase --output outputs/level5/perception_development_v5/balanced_v22/phase.npz
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.balanced_phase_perception train --config configs/level5/perception_repair_v22.yaml
```

Existing paths are immutable; do not rerun into them. No5.6 completion, fresh
confirmation acceptance or5.7 work is claimed.

Metadata erratum:`docs/level5_perception_recipe_metadata_errata.json` binds the
inherited obsolete CART seed-description field in frozen neural recipes to
the actual executable `network_candidates` declarations. V22 runs full MLP
seeds3/7/11 at3,000epochs each; all neural weights store exact seed/epoch metadata.
Frozen historical recipes/weights are preserved. The independently locked
harvest recipe is included in config-lock discovery; no5.0locks are changed.

V22 completes all3full3,000epoch budgets; all3candidates pass both strict
validation checks. Frozen ranking selects seed11: primary recall99.983%,median
0.745mm,p953.199mm,zeroID; worst supported phase (cuboid/lift)p9520.402mm.
Dense stabilization20/20poses/6physical sources:median3.023mm,p954.671mm.
Exactreceipt:`docs/level5_perception_phase_v22_validation.json`. Full regression
passes1,115tests/1skip(785.68s),14focusedrecipe/balancechecks pass,Ruff passes.
Actual selected-model64-query truth-isolation/resource audit and accepted-bundle
inventory/transform oracles pass. FreshV5protocol binds that exact model before
256new99580000-series source cases are generated. Collection is in progress;
no fresh outcome,owner acceptance or5.6completion is claimed.

FreshV5 is consumed and negative. All256sources /13,227frames are retained,
zero historical collisions. Archive2,249,469,632bytes/27,227files,complete
memberreadbackpasses,SHA7d6a8168a71f32a85123fbebdbfd43eaf90973fd5cdd398bd447809aa97385e4.
Original/core/primary position/identity gates pass:primaryrecall99.959%,median
0.830mm,p953.432mm,zeroID. Strict cube release17/18(94.44%) andsettle7/9(77.78%)
fail95%recall; pose gates pass. All5consumed sets closed; actual alternate-output
retryguardrefusesbeforemissingdatasetaccess,withunchangedreservationandno2nd
inference. Noownerreviewtriggered. Receipt:`docs/level5_perception_confirmation_v5_failed.json`.
FurtherrepairisTRAIN/validation-onlylatephasecoverage/detection; no testimages
orindividualcoordinatesaretuninginputs. Gatesunchanged; no5.7work.

## V23 TRAIN hue refinement with exact pose-weight reuse

Independent VALdiagnostic shows a3pixel cube(H11,S223) joined to3reddish
background pixels(H3/4,S168–175), expanding a3x1GTbox to5x2(IoU.3). The test
images/coordinates are not used. Freeze a TRAIN-only signed-hue histogram fit
(.0001/.9999quantiles,one-bin padding) using eligible cuboid GTmask pixels from
original TRAIN plus late-window TRAIN. Other classes/saturation/value/confidence
limits remain frozen. No hue parameter is fitted on validation/test pixels.

Separately frozen developmentV6contains1,066release/settle query windows,4,264
rows,retaining480parents/441zero-query parents and39rendered physical sources.
Visible TRAINrelease473/settle243each27sources;VAL195/99each11sources. All
queries are retained independent of visibility/outcome; no new physics sessions.
Archive/memberreadbackpasses3,289files;SHA
a3f4d3ba9774dff059a5e27aa2b2d72fce4d762e6157eb3797b1fd675c697854.

V23preserves exact V22MLP parameters/transforms/decoder/cadence for all3seeds;
no neural refitting or new stochastic budget. It changes runtime cube segmentation
with the fitted hue interval and fully revalidates all3bundles. V22was trained
under the original segmentation frontend; this distinction is explicit. All
original full-sequence plus stabilization and individualrelease/settle query
gates must pass before any fresh confirmation. Neural parameter digests are
asserted unchanged. Selected weights and actual revised frontend are packaged
together; no accepted control pipeline is altered.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n dexvision python -m dexvision.apps.hue_repair_perception --config configs/level5/perception_repair_v23.yaml
```

Existing paths remain immutable; this records the actual run,not a retry command.
No5.6completion/newmanualacceptance/5.7work is claimed.

V23finishes all3candidate validations and allpass. The frozen ranking selects
V22seed11's exact original3,000epoch pose parameters with runtime cube hue9–19.
Primary VALrecall100%,median0.745mm,p953.208mm,zeroID; original worst supported
phase p9520.402mm. Late release195/195poses/detections (median2.983mm,p9510.523mm),
settle99/99(median1.094mm,p959.325mm),each11physical sources; stabilization20/20
from6sourcespasses. Neural parameter digest exactlymatches V22parent. Full
regression1,123passed/1skip(790.22s),18focusedchecks,Ruff pass. Actualselected
64-query typed truth-isolation/resourceaudit and accepted-bundle oracles pass.
Exactreceipt:`docs/level5_perception_hue_v23_validation.json`.

FreshV6plan freezes512new101580000-series physical resets(32per exposed cell/
condition) before generation. Stronger mandatory fresh release/settle support
requires8visible from4physical sources each,withalloriginal limitsunchanged.
No unsupported late stratum can qualify by omission. Ownerreviews32fixed
nominal/boundaryclipsplusworst/coreclass;all512videosretained. Protocol creation
and collection are active,not an acceptance claim.

FreshV6consumednegative:512sources/26,848frames,zerohistoricalcollisions,
archive4,566,326,484bytes/55,237files fullreadbackPASS,
SHA3bc8546fc00f35c40914bf3e0f602f9b05f4d63e04950c73e78da2843f63c7bf.
Global/primarygatesPASS(primary14209/14210,median.814mm,p953.021mm,zeroID).
Cube transportp9537.212mm>25 and release position-hypothesis coverage50/51
(98.04%)<99 FAILstrict. Release51/52recall andsettle30/30pass; missingpose
aggregatecauseidentityambiguity. No testimages/coordinatesusedfortuning.
All6consumedsetsclosed;actualalternate-output/missingdatasetretryguardpasses
withunchangedreservation/no2ndinfer. Noownerreviewtriggered.
`docs/level5_perception_confirmation_v6_failed.json` bindsnegativeevidence.
FurtherrepairusesONLYTRAIN/VAL:exactHuefrontendpose refit,dense allmotion
phases,andseparatelytrainedcurrent-RGBlocalization withidentityambiguity
retained/invalidforcontrol/no ambiguous motion. No metric/ambiguitythreshold
waiver,no5.7work.

## V24 dual localization repair (in progress)

V6 consumed confirmation fails cuboid transport p95 and release localization
coverage; exact negative receipt and all512sources/26,848frames are preserved.
A separate current-frame head will retain position hypotheses under identity
ambiguity without certifying identity, position validity or motion/control.
Temporal pose is retrained under the exact V23 hue frontend. Six independent
full3,000epoch fits (two heads x seeds3/7/11) use weighted TRAIN XYZ loss and the
same fixed output scales and calibrated ray decoder. No inherited pose weights.

DevelopmentV7 all-control-frame motion-window collection was interrupted for
resource volume before fitting/evaluation; its frozen protocol and partial
staging remain. DevelopmentV8 independently freezes original frame-index stride5,
three causal contexts, six motion phases, all480physical parents/outcomes and
whole-source split ownership. It is a derived RGB amendment, no new sessions.
No visibility/outcome-based query selection. Query features can be cached using
current RGB only; exact runtime parity, scalar/batch parity, ambiguous invalidity,
head-switch/gap/reset motion handling and current-only independence are tested.
All six consumed sets remain closed; no5.7work.

Commands executed/prepared (write-once outputs; never overwrite/retest):
```bash
conda activate dexvision
python -m dexvision.apps.cadence_window_perception freeze --config configs/level5/perception_development_v8.yaml --output-dir outputs/level5/perception_development_v8/protocol
python -m dexvision.apps.cadence_window_perception collect --protocol outputs/level5/perception_development_v8/protocol/protocol.json --output-dir outputs/level5/perception_development_v8/dataset
python -m dexvision.apps.dual_repair_perception harvest --config configs/level5/perception_harvest_v24.yaml
python -m dexvision.apps.dual_repair_perception train --config configs/level5/perception_repair_v24.yaml
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -q tests/test_perception_dual.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -q
ruff check .
```
The harvest/training recipes bind actual dataset/cache digests before use.
Current focused checks:10passed. Full regression and collection in progress.
No fresh protocol/pixels or owner acceptance for V24 are claimed.

Cadence correction before any V24 fit: original D3/D4 query stride is5, not20.
V8 retains that cadence and adds causal-window reset contexts; no4x density or
new sessions is claimed. The immutable V8 reason-string error is bound in
`docs/level5_perception_cadence_metadata_errata.json`. All dense stabilizationV5
and lateV6 windows are reharvested too; validation duplicates count once.

V8 collection/archive/schema audit passes:19,448queries/77,792rows,
320TRAIN/160VALphysical parents including2zero-query sources, no new sessions,
zero consumed/cross-split collisions. Archive3,307,189,778bytes/41,769files has
full checksum/member readback and LFSfilter. Source/render position delta0 on
622,336body origins from146,776typed snapshots. TRAIN/VAL primary visible
queries546/294from99/46physical sources; these are label coverage, not model
performance. V24feature harvest is active under its frozen harvest recipe.

Final V24 software suite:1,136passed/1skipped in795.48s; focused13passed;
Ruff and whitespace pass. Immutable local evidence is saved under
`outputs/level5/perception_training_union_v18/software_v24/` with a SHA index.
This software result precedes model fitting and does not claim learned visual
validation, fresh confirmation or owner acceptance. Feature harvest is active.

Before fresh confirmation, a display-only interface gap was found: the frozen
DualContextPose has no projection method used by the old review overlay.
Frozen training/feature/inference code and weights are preserved. New versioned
`perception_projection.py` supplies a calibrated projection view ONLY to the
renderer after inference; `perception_dual_review.py` keeps the scoring function
body and actual inference class identical. New confirmation/export code is
pinned before fresh generation. Two new headless tests cover projection without
state mutation and complete video/contact-sheet/typed export. Focused15pass;
full1,138-test regression is in progress. The waiting old post-validation job
was stopped before any fresh protocol/pixels; the replacement waits for these
checks too. This changes no learned parameter, numerical gate, or control rule.

## V25 frozen current-only comparison

V25 now freezes a simpler current-only comparison before any new fresh pixels.
V24seed3 current-only stress passes every phase (lift p95 19.59mm), while its
temporal route fails lift p95 34.74mm. All V24source seeds still complete both
3,000epoch fits. V25reuses each exact completed current head, with zero optimizer
updates, and scores every candidate on complete sequences and all six phase
queries. Localization receives current RGB only; the visual tracker stays
causal. Physical motion is unavailable (None), not a zero measurement. A new
engine identifier rejects accidental legacy temporal loading. A pure optional
VALfeature cache is keyed by current RGB/mask/box/class, never labels/history or
model output; fresh inference is uncached. Focused20pass and full1,143-test
regression is running. No V25full-main result or fresh acceptance is claimed.

Plan: `configs/level5/perception_current_plan_v25.yaml` and SHA sidecar.
Fresh7case/seed/coverage plan is unchanged; its pre-pixel binding addendum is
`outputs/level5/perception_development_v8/confirmation_v7_binding_v25.json`.
V24temporal/current source fits and all prior negative sets stay preserved.

## V25selected validation and pre-confirmation audits

All six V24source fits finish; dual candidates fail and remain preserved in
`docs/level5_perception_dual_v24_source_results.json`. V25reuses exact current
banks without updates. Current seeds3/11 pass and seed7fails worst p95 25.32mm;
selected seed11 has primary100%recall,0.904mmmedian/3.414mmp95,0ID; worst phase
p95 14.082mm. Each545phasequery matches+haspose. No motion is fabricated.
Fullsoftware1,143pass/1skip780.40s; current20focused andcombined30pass. The
config-lock test now discovers independently checksummed current-plan YAMLs;
original5.0lock unchanged. Source87files and data/active-weight digests are frozen.
Actual64query truth isolation and48cached/uncached source-query replay have
zero position delta; all five inferred-control dispositions stay unsupported.
Model14,528,857bytes/Pythontracedpeak32,430,134bytes(scope64queries after imports;
not native/full-media peak; RTXuntested). Portable bundle relocation parity and
complete99-member readback pass. A temporary publication filename variable
shadowing error was fixed before the first archive; original staging/source are
preserved and publication changes no parameter/transform/calibration. Resume
starts at the package boundary, with no earlier fit/evaluation repeated.
FreshV7config/protocol freezes512disjoint103580000-series cases against this
exact model before any pixel generation. All6consumed tests excluded. Collection
is active; no held-out/manual acceptance or5.6completion is claimed. No5.7work.

## V7fresh collection and one-time evaluation

All512source attempts and26,555frames are preserved, with zero historical and
cross-split pixel collisions. Immutable test-only release has complete54,651-file
checksum/member readback. One current-only inference reservation now consumes
V7permanently, regardless of outcome; all earlier six sets remain closed too.
No numerical/manual pass is claimed before the final report/audits.

Executed under dexvision (write-once outputs; consumed tests cannot be rerun):
```bash
python -m dexvision.apps.current_perception select --config configs/level5/perception_repair_v25.yaml
python -m dexvision.apps.current_perception freeze --config configs/level5/perception_confirmation_v7.yaml --output-dir outputs/level5/perception_confirmation_v7/protocol
python -m dexvision.apps.render_perception_amendment collect --protocol outputs/level5/perception_confirmation_v7/protocol/protocol.json --output-dir outputs/level5/perception_confirmation_v7/dataset
python -m dexvision.apps.repair_perception package --protocol outputs/level5/perception_confirmation_v7/protocol/protocol.json --dataset-dir outputs/level5/perception_confirmation_v7/dataset --release datasets/level5-perception-confirmation-v7.tar.gz
python -m dexvision.apps.current_confirmation_perception --protocol outputs/level5/perception_confirmation_v7/protocol/protocol.json --dataset-dir outputs/level5/perception_confirmation_v7/dataset --output-dir outputs/level5/perception_confirmation_v7/evaluation
```

V7 final result: strict confirmation fails one primary cuboid/release identity
switch; position/recall and late coverage pass. Receipt:
`docs/level5_perception_confirmation_v7_failed.json`. V7 is consumed, archived,
and closed to repair/selection/repeat evaluation. V26 uses only existing frozen
TRAIN/validation data and exact V25 position weights; no5.7 work.

V26 and V27 fail independent validation's added non-vacuous identity coverage
because broad singleton/new-birth rejection discards legitimate puck evidence.
Both frozen negative attempts and full software logs remain preserved. V28
instead changes only cuboid association coordinates to the calibrated projection
of the fixed current-RGB body-center prediction; original visible masks/boxes
and all position outputs remain unchanged. V29 adds a cube-only uncertain
replacement-ID guard from an independent synthetic position-outlier audit.
Other classes retain original birth and visible-box association. Tracker limits
and added99%primary/95%supported-phase unambiguous coverage remain unchanged.
`docs/level5_perception_body_tracking_v29_guard_plan.json` records the choice
before any fresh V8 pixels and without V28 validation outcomes. Frozen recipe:
`configs/level5/perception_repair_v29.yaml` (SHA256
`c272d1adeddcd2a1c627e06623efcb7efd9d0b7b885e0bdb8d694227edc898ad`).
All source/data/fit artifacts remain immutable; this is checkpoint5.6 only.

## Final V29 automated result and owner handoff

**Current outcome: 5.6 complete; V29/fresh V8 automated gates and owner manual acceptance pass.**
The owner confirmed the held-out review passed on October 16, 2026.
Last Completed is **5.6**; Next Target is **5.7 (unstarted)**, on branch `main`.
`docs/level5_perception_acceptance.json` binds this owner-reported acceptance to
the exact frozen artifacts; no agent visual observation is claimed. Historical
automated receipts retain their then-pending manual status. No 5.7 work is
authorized by this completion request. The earlier V8 validation diagnostic
review remains separately accepted.

V29 retains exact V25 selected seed 11/source epoch 3,000 current-RGB MLP weights,
TRAIN transforms, fixed8px/8px/10 mm output scales and calibrated ray decoder,
with zero new optimizer updates. All three original current candidates and all
six source fits remain preserved. Cuboid association uses the calibrated
projection of the current learned body center, with a cuboid-only uncertain
replacement-ID guard and history rollback. Other classes keep original birth/
visible-box association. Published masks, boxes and positions are unchanged.
Original tracker limits and all original visual gates remain unchanged.
Added identity coverage requires99% unambiguous primary matches overall and 95%
in every supported primary class/phase; V29 passes both.

Validation: 5,923/5,923 primary matches and positions, zero identity switches, median
0.904 mm/p95 3.414 mm;
all 545 six-phase queries pass (worst phase p95 14.082 mm).
Fresh V8: 512 ordinary-physics source resets/26,687 RGBframes,
primary recall100.0000%, median 0.954 mm,
p95 3.675 mm, 14,244 positions and
0 primary ID switches. All original/global/core/primary, supported phase,
position coverage, added identity coverage, and mandatory release/settle gates
pass independently. Release: 46 visible from
13 physical sources;
settle: 24 visible from
13 physical sources.

Full software: 1,177 passed/1 skipped; focused 64, Ruff and
whitespace pass. Actual64VAL truth-isolation queries,48cached/uncached runtime
paths, exact accepted-transform oracles, portable RGB relocation, archive/member
readbacks, source/body-origin/schema audit, all 512 video decodes, typed/raw
parity and actual consumed-protocol retry rejection pass. The review has 16
panels and 36 fixed-plus-worst temporal clips.
Evidence:`docs/level5_perception_verification_v29.json`.

Model semantic SHA:`3a25ea7b49cc05ce7d67e89226a7e8c231a7feaea945d4863e7ed56a5cf5cadd`.
Active current-weight SHA:`0ba58825982fbfd714f028c3b8a9c4195e9f4ec5befe678e5273025863bbdb85`.
The baselineV2 and freshV2–V7 consumed tests stay permanently closed, with all
archives, data, failed/positive candidates, reports and reservations preserved.
V7failed one primary cuboid/release IDswitch; V26/V27failed independent identity
coverage. V28positive validation and the independently frozen V29guard plan
are preserved. Fresh V8 is now consumed too and cannot be retried or used for
training/selection. All source outcomes were retained; no test tuning or reuse.

The availability-aware schema is shared by truth and inference. Physical
linear/angular motion, full orientation, semantic identity, robot/fixture
telemetry and contacts remain explicitly unavailable and fail closed. All five
accepted inferred-control tracks remain unsupported. Qualification scope is
calibrated simulator pixel-center images on exposed task families/cells;
no real-camera, unseen-family, hardware, full-feature or closed-loop control
qualification is claimed. Python traced peak: 32,454,158 bytes for
model construction and 64 queries after imports; native/full-media peak remains
unmeasured and RTX 5070 Ti is untested.

Manual command (dexvision Conda environment, repository root):
`python -m dexvision.apps.repair_perception review --evaluation-dir outputs/level5/perception_confirmation_v8/evaluation`.
The owner confirmed all prescribed review criteria passed on October 16, 2026;
`docs/level5_perception_acceptance.json` records acceptance of the 16 panels and
36 scheduled clips. The command above is retained for saved-media reproduction;
no manual check remains for 5.6. Stop before 5.7.


```bash
conda activate dexvision
python -m dexvision.apps.guarded_body_perception select --config configs/level5/perception_repair_v29.yaml
python -m dexvision.apps.guarded_body_perception freeze --config configs/level5/perception_confirmation_v8.yaml --output-dir outputs/level5/perception_confirmation_v8/protocol
python -m dexvision.apps.render_perception_amendment collect --protocol outputs/level5/perception_confirmation_v8/protocol/protocol.json --output-dir outputs/level5/perception_confirmation_v8/dataset
python -m dexvision.apps.repair_perception package --protocol outputs/level5/perception_confirmation_v8/protocol/protocol.json --dataset-dir outputs/level5/perception_confirmation_v8/dataset --release datasets/level5-perception-confirmation-v8.tar.gz
python -m dexvision.apps.guarded_body_confirmation_perception --protocol outputs/level5/perception_confirmation_v8/protocol/protocol.json --dataset-dir outputs/level5/perception_confirmation_v8/dataset --output-dir outputs/level5/perception_confirmation_v8/evaluation
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n dexvision pytest -q
conda run --no-capture-output -n dexvision pytest -q tests/test_perception_guarded_body.py tests/test_perception_body_tracking.py tests/test_perception_continuity.py tests/test_perception_identity.py tests/test_perception_dual.py tests/test_level5_learning_plan.py
conda run --no-capture-output -n dexvision ruff check .
git diff --check
```

Completed saved-media review command (repository root; no repeat required):

```bash
conda activate dexvision
python -m dexvision.apps.repair_perception review --evaluation-dir outputs/level5/perception_confirmation_v8/evaluation
```

SPACE starts after the contact sheet, pauses/resumes a clip, and advances after
its end. Q, ESC or closing the window means incomplete review, not acceptance.
Pass only after all 16 scheduled panels and all 36 fixed-plus-worst clips are
present, readable and reviewed; visible original masks/boxes and current body
positions align with geometry; certified visual IDs stay with their objects;
gaps, resets and reacquisition are declared; uncertain replacement IDs and
ambiguous/stale/low-confidence evidence visibly reject. Unsupported full pose,
semantic identity, motion and sensor fields must stay unavailable, with no
control admission. Diagnostic position hypotheses may remain visible while
identity is explicitly invalid. Source expert manipulation failure alone is
not a perception failure. Fail for missing/corrupt media, unexplained valid-ID
swaps, unexplained pose jumps, misaligned overlays, hidden truth filling, or
accepting invalid/stale/ambiguous evidence. Report pass/fail and any failing
clip/panel. The owner has confirmed this review passed; acceptance is recorded
in `docs/level5_perception_acceptance.json`.
The earlier V8 validation diagnostic review is accepted and need not be repeated.
