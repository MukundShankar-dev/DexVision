# Detector, tracker and metric pose baseline (5.6)

Checkpoint **5.6 remains open**. This implements a conventional CPU baseline,
its independent recipe, temporal evaluator, observability audit and fail-closed
schema boundary. V1 and V2 fail validation; the frozen V2 held-out evaluation
is a one-time negative experiment, not a qualified perception bundle. Last
Completed remains 5.5 and Next Target remains 5.6. No 5.7 rollout, registry or
accepted-policy modification is included.

## Prior-stage evidence and changes

Reviewed the accepted 5.4A place V25/V19 and 5.5 push V8/V7/press V1 receipts,
their repair/results reports, the reach repair/training guidance, the Level 4
visual export and the relevant module contracts. Their required adjustments
are carried forward as follows:

- Measured forearm and commanded mocap origins differ. The offline audit calls
  the exact accepted weld calibration and feature transforms; inferred object
  position is not substituted for robot telemetry.
- Physical retention/contact and causal completion/dwell were essential to
  pick/place/push/press. No saved expert phase, retrospective success, empty
  contact list or default held bit supplies inferred history.
- Place V25 uses copied live MuJoCo state to preview contacts and select actions.
  Its accepted preview remains unchanged and explicitly unsupported for vision.
- Source geometry and qualification geometry differed in earlier repairs.
  Nominal XML primitive dimensions and upright silhouette hypotheses here are
  **not certified instance dimensions or full 6D body poses**. Their errors
  remain counted, including parked off-board objects and partial visibility.
- Visual samples are every fifth source frame, with cross-split duplicates
  excluded. Tracking resets per episode, uses actual elapsed timestamps and
  expires after the independently frozen 0.51-second gap. The typical 0.17-second
  image spacing cannot provide the inherited 0.1-second freshness limit or the
  accepted 0.034-second control cadence between images.
- Earlier result artifacts and failed attempts stay immutable. CPU thread limits
  from the prior training commands are also used for the final regression run.

## Method and frozen artifacts

`configs/level5/perception.yaml` / `perception.sha256` freeze V2 before fitting:
camera `level4/visual-camera-v1`, training-only supervision, three deterministic
smoothing candidates, selection ranking, geometry, timestamp/gap rules,
thresholds, diagnostics and 248 input pins plus implementation/schema pins.
The original 5.0 YAMLs, `SHA256SUMS`, `evaluation_common.yaml`, datasets and
selected policy weights are unchanged. The config-lock test is extended to
recognize this new independent sidecar; it does not rewrite historical locks.

V1 fits median HSV hue per loose-object class from training RGB/masks, extracts
colour components, tracks by image-space assignment and fits primitive projected
box edges. Its complete validation recall is 92.87%, median position error
39.18 mm and p95 123.15 mm. All three hue-margin candidates are retained under
`outputs/level5/perception_model_v1/`; its exact recipe and implementation are
preserved in `outputs/level5/perception_model_v1_source/`. V1 never opens test.

V2 adds a deterministic training-only hue/saturation histogram for foreground
classes and background, and fits continuous silhouette centroid/covariance to
projected primitives. All candidates receive the complete 531-frame validation
stream. Selection maximizes worst core-class recall, then total recall, fewer
false positives and smaller smoothing margin. Margin 8 is selected. V2 validation
recall improves to 93.33%, but median position error regresses to 93.49 mm and
p95 to 453.55 mm. The regression is retained, not described as an improvement
in metric pose. False positive components and uncertified hypotheses remain
visible. The first failed full-suite lock check is also recorded in the task
history; the independent-recipe enumeration extension repairs it.

There is no neural training, stochastic sampling, optimizer, pretrained model,
VLM or actuator output. One complete deterministic histogram/median fit replaces
training epochs, so three random seeds are inapplicable to this conventional
method. Torch is used only to read the exact accepted state-bundle schemas in
the audit. RGB inference accepts only pixels and acquisition timestamps.

V2 selected parameters, candidate reports, full validation and inventories are
under `outputs/level5/perception_model_v2/`. The selected JSON binds the exact
recipe, code and data. `perception_model_v2_heldout_attempt.json` reserves one
test evaluation independent of output path; changing the output directory
does not permit another evaluation. Existing evidence directories are refused.
The held-out report, every inferred frame, every core truth instance metric,
contact sheet and episode-local videos are under `outputs/level5/perception/`.
Artifact indexes and the tracked verification receipt bind these local outputs.
They are not a published model/data release.

## Field-source inventory and compatibility

`field_source_inventory.json` in the model/evaluation directories enumerates
every exact selected neural input and additional runtime dependencies. Its
normalized-feature descriptors must be read with the physical-unit map below
and the pinned accepted recipes; they do not establish deployed sensors.

| Input family | Physical units/body origin and required producer | Disposition |
| --- | --- | --- |
| Object/goal/command offsets, height, entry reference | metres at object body centre, world or accepted command-hand/task basis; position divisor 0.1 m in accepted transforms | RGB offers only uncertified nominal-size/upright hypotheses; entity-derived goals blocked |
| Robot pose, tracking lag, prior commands, joints | named forearm pose calibrated through pinned weld into command origin; metres, wxyz/rotation6d, radians; prior applied command at pre-action time | Conditional on actual declared encoder/command telemetry; simulator availability is not a deployed sensor |
| Object linear motion | m/s, world or accepted hand frame; exact elapsed-time differences; 0.1 physical divisor where used by accepted transform | Hypothesis only; missing entry/gap velocity is `None`, never zero |
| Object orientation/angular motion | wxyz or rotation6d, rad/s; accepted pick orientation error uses its pinned 0.1 numerical scale | Full tilt/yaw/angular motion unsupported; only upright cuboid symmetry-yaw diagnostics |
| Semantic class/instance | class indicators are unitless; IDs must name stable requested workcell entities | RGB class hypotheses/episode-local `visual/NNNN` IDs do not certify semantic instance IDs |
| Fixture depth/rest/axis | metres from fixture body-centre rest along frozen world +X; depth divisor 0.01 m | Rest/axis calibration may be fixed; current displacement and 2-mm release rule need a fixture sensor |
| Contact, supported/held state | measured booleans or explicit measured contacts at aligned time | Unsupported; no default false or empty truth substitute |
| Phase, closure/contact/pressed/release dwell, completion latch | causal current/past physical predicates; sample counts, unitless bits; pick closed-contact count divided by 10 | Unsupported without measured contact/retention/fixture/terminal producers; reset every independent request |
| Static target geometry, bounds, fixed posture/cadence | frozen world geometry, physical limits and named command channels | Engineered inputs/controls; they are not learned from vision |
| Place action preview | full live qpos/qvel plus copied-state predicted contacts/object positions | Evaluation-only privileged dependency; unsupported for perception |

All five accepted bundles have an explicit **unsupported** perception
compatibility disposition. Reach needs certified entity-derived goals and
declared robot telemetry; pick needs orientation/retention/contact and entry
context; place additionally needs a nonprivileged preview replacement in a
separately authorized version; push needs measured support/contact, task basis,
physical phases and settling predicates; press needs measured fixture depth and
contact/dwell. No consumer is enabled by this audit.

The existing `ObjectObservation` and `WorldState` types remain unchanged. The
new boundary constructs a world state only from explicit available inferred
entities and declared measured robot/relation/fixture/contact sources. Missing
orientation/identity prevents these RGB hypotheses becoming valid typed policy
inputs. Evaluation-only truth uses the same existing object schema, separately
from inference. Schema parity unit tests are not full inferred feature parity.

The original compact visual release has no RGB for later unassisted pick V2
and place V3 source entries/geometry. These missing strata are explicit, not
repaired using test scenes or source-truth pose labels. Historical source
skill/phase labels are used only for evaluation breakdowns after inference.
Supported baseline claims are limited to the frozen simulated camera/conditions;
there is no real-world, unseen-family or cross-camera claim.

## Metrics and diagnostics

Every visible core truth instance is eligible, including tiny and partially
occluded instances. Invisible labels never create detections. Matching is
one-to-one same-class IoU >=0.5, strictly after RGB inference. Report missing
detections, false positives, conditional position/velocity hypothesis counts,
certified pose availability (zero), rejected observations and identity matched
coverage alongside errors. No empty group passes a gate.

Identity switches use episode-local truth association only for scoring, within
the frozen gap; long gaps expire tracks and reacquisitions are reported. Motion
uses timestamp differences. Cuboid yaw diagnostic scoring is modulo pi/2 for
the square block and pi for the rectangular block when truth tilt satisfies the
predeclared upright diagnostic limit. Round-body yaw/full orientation/angular
velocity and fixture depth are unavailable, not assigned zero error.

The exact accepted reach/pick/place/push/press feature transforms are called
in separate synthetic offline oracles. Scalar/batched parity errors are at
most 8.7e-19. No unavailable physical field is filled to produce inferred policy
features. Bias sweeps at 1/10/25 mm and explicit press 2-mm, push tracking 10-mm
and speed 20-mm/s boundary sweeps illustrate sensitivity; they do not measure
sensor reliability or closed-loop success. Real inferred feature/phase parity
and calibrated uncertainty at these boundaries remain blocked.

Synthetic stale/future, low-confidence and ambiguous delivery rejection is
100%. Additional tests cover unavailable orientation/velocity/contact/fixture
fields, explicit invalidity, gap/reset/reacquisition, immutable selection,
one-time test reservation and withheld truth-label isolation. Inference cannot
consume annotation poses, IDs, masks, task/phase labels or terminal outcomes.

Hardware/latency/model size and Python allocation peak are recorded in the
held-out report. Native OpenCV/SciPy process peak is not measured, and RTX 5070
Ti is explicitly untested/unavailable. Held-out latency includes Python memory
instrumentation and concurrent regression work; it is an offline diagnostic,
not a control-cadence qualification. Released generated data uses CC-BY-4.0;
custom workcell/source and Shadow Hand assets retain Apache-2.0 attribution.
Historical export-time unspecified-license notes are not edited.

## Commands and current acceptance boundary

Commands run from the repository root in `dexvision`:

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.train_perception --config configs/level5/perception.yaml
conda run --no-capture-output -n dexvision python -m dexvision.apps.evaluate_perception --config configs/level5/perception.yaml --output-dir outputs/level5/perception
conda run --no-capture-output -n dexvision pytest -q tests/test_object_perception.py tests/test_perception_metrics.py
conda run --no-capture-output -n dexvision pytest -q tests/test_object_perception.py tests/test_perception_metrics.py tests/test_level5_learning_plan.py
conda run --no-capture-output -n dexvision ruff check dexvision tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run --no-capture-output -n dexvision pytest -q
git diff --check
```

The first two identify consumed runs, not commands to overwrite evidence.
Intentional reproduction requires a separately versioned recipe/model directory;
changing just the evaluation output path cannot repeat consumed test. Evaluation
returns exit 2 for an unqualified baseline after writing its complete report;
this differs from a code/dependency failure (exit 1). Failed visual gates cannot
be fixed by lowering thresholds, owner viewer acceptance or repeating test.

No acceptance review is triggered while automated visual gates fail. The
generated media is diagnostic evidence only. For inspection on macOS:

```bash
open outputs/level5/perception/heldout_contact_sheet.png outputs/level5/perception/sequence_*.avi
```

On Windows PowerShell:

```powershell
Start-Process outputs/level5/perception/heldout_contact_sheet.png
Get-ChildItem outputs/level5/perception/sequence_*.avi | ForEach-Object { Start-Process $_.FullName }
```

The eventual manual gate requires every scheduled contact-sheet panel and all
16 held-out episode videos in `manual_review.json`, with original episode,
frame, acquisition timestamp, inferred ID/confidence/age, contour/box,
hypothesis axes and rejection reason. Missing later skill/fixture strata must
remain labelled unsupported.

**Pass:** all panels/sequences are present; IDs stay with the correct visible
objects; pose/box/mask overlays align; timestamps, units and source frames agree;
missing/stale/ambiguous observations visibly reject. **Fail:** identity swaps,
unexplained jumps, misaligned overlays, hidden truth-filled estimates, omitted
strata, silent stale acceptance, cancellation or review errors. Owner
confirmation is required only after the automated baseline gates pass; neither
this diagnostic media nor agent inspection completes 5.6.

## One-time held-out V2 result

All 811 frames from 16 test-owned episodes were processed once. There are
4,391 visible eligible core instances, 2,304 IoU-matched detections and 2,087
misses. The remaining 9,951 predicted components are false positives. One
matched ambiguous observation has no pose hypothesis: 2,303 conditional pose
errors remain in the denominator, alongside zero certified typed poses.

| Metric | Result | Original gate |
| --- | ---: | ---: |
| Core detection recall | 52.47% | >=95% — failed |
| Median position hypothesis error | 81.35 mm | <=10 mm — failed |
| P95 position hypothesis error | 453.55 mm | <=25 mm — failed |
| Identity switches | 0 | <=1 per 1,000 track frames — passed under frozen disclosed denominator |
| Synthetic stale/ambiguous rejection | 100% | 100% — passed |
| Certified full body pose / semantic instance input | 0 | Unavailable; consumers reject |

Per-class recalls are button 100%, cuboid 52.34%, cylinder 49.57% and puck
54.09%. This conventional appearance model does not meet the held-out class
gate. V2 pose fitting also fails substantially; its position numbers are
uncertified hypotheses, not accurate valid measurements. No test result is
used to change the recipe, thresholds or selection, and no consumed test is
repeated. The separate negative receipt preserves all input/output bindings.

The model JSON is 266,456 bytes. On Darwin arm64, instrumented offline inference
p95 is 0.724 seconds and peak Python allocation is 118,048,149 bytes. Native
process peak and RTX 5070 Ti performance remain unmeasured/untested. The original
image intervals are approximately 0.17 seconds, above the 0.1-second parent age
limit and accepted control cadence. These measurements do not qualify runtime.

The manual schedule contains all 16 condition/class panel slots and all 16
episode videos. Independent file/video read-back checks are in the verification
receipt. A single extracted video frame was inspected by the agent and shows
heavy false-positive/label clutter, consistent with the negative experiment;
this is neither a complete visual review nor owner acceptance. Further work
must remain in checkpoint 5.6 under a separately frozen method/data amendment,
preserving these failed runs. The baseline and all skill perception tracks
remain unqualified. No next checkpoint is recommended or started.

## Changed files

Final software verification: **32 focused perception tests passed**, **42 passed**
with the config-lock regression, **946 passed and one skipped** repository-wide,
repository-wide Ruff passed, and `git diff --check` passed. The final full-suite
run took 741.90 seconds. Artifact read-back verified all 811 frame/timestamp
records, all 16 panels, complete decoding of all 16 videos, every artifact
index, current source pins and the retained V1 source snapshot. These software
and integrity passes do not change the negative visual qualification.

```text
configs/level5/perception.yaml
configs/level5/perception.sha256
dexvision/perception/object_detector.py
dexvision/perception/object_tracker.py
dexvision/perception/object_pose.py
dexvision/perception/perception_baseline.py
dexvision/learning/perception_baseline.py
dexvision/evaluation/evaluate_perception.py
dexvision/evaluation/perception_metrics.py
dexvision/evaluation/perception_audit.py
dexvision/apps/train_perception.py
dexvision/apps/evaluate_perception.py
tests/test_object_perception.py
tests/test_perception_metrics.py
tests/test_level5_learning_plan.py
docs/level5_perception_results.md
docs/level5_perception_verification.json
docs/CURRENT_STATUS.md
docs/progress_level_5.md
docs/module_contracts.md
```

The completed checkpoint prose, original model/data releases and accepted
runtime modules are unchanged. Documentation edits touch current staging,
the active 5.6 section/checklist and its new results only. Local generated
evidence remains ignored under `outputs/level5/`.
