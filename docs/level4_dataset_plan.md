# Level 4 Workcell and Dataset Requirements Freeze

Specification version: `level4/workcell-dataset-plan-v1`

Status: frozen for the Level 4.1–4.3 implementation and pilot. Level 4.3 may
replace provisional counts only by publishing a new configuration version and
recording pilot evidence. No Level 4 collection has started under this schema.

## Decision and evidence boundary

Level 3 established that the saved-data pipeline is reproducible, but none of
the validation-selected full-action policies passed its frozen closed-loop
gate. The failures do not support a single-cause claim or an immediate model
change. Level 4 therefore freezes a combined data and control contract:

- broaden task, object, goal, reset, and outcome coverage;
- isolate genuine recording sessions across train, validation, and test;
- record typed world state, causal online phase, and aligned fixed-camera RGB;
- retain requested, commanded, and applied actions with deterministic safety
  masks and reasons; and
- keep expert successes, ordinary failures, policy rollouts, and corrections
  distinguishable.

The evidence sources are:

- `docs/level3_results.md` and
  `outputs/level3/feasibility_v1/summary.json`;
- `docs/level3_evaluation_protocol.md` and the three frozen Level 3 evaluation
  YAML files;
- `datasets/dexvision_level2_v1_manifest.json`, its archive checksum, and the
  released quality/summary reports; and
- `docs/task_environment.md` and `docs/module_contracts.md`.

The immutable Level 2 release is legacy seed data only. Its 55 reach, 55
button, and 101 push clean successes do not count toward Level 4 minimums, and
its episodes will not be rewritten to invent session provenance.

## Frozen workcell scope

The machine-readable authority is `configs/level4_dataset.yaml`. It freezes one
bounded MuJoCo tabletop workcell with three rigid-object families and two
instances per family:

| Family | Training instance | Test-only held-out instance |
|---|---|---|
| Cuboid | `block_small` | `block_large` |
| Cylinder | `cylinder_short` | `cylinder_tall` |
| Flat puck | `puck_light` | `puck_heavy` |

The required fixture is `start_button`. The frozen targets are
`return_bin_left`, `return_bin_right`, `inspection_pad`, `setup_slot_a`, and
`setup_slot_b`. `return_bin_right` is a held-out goal region. The optional
`mode_dial` is deferred and contributes zero required episodes unless Level
4.3 promotes it in a new version after a successful pilot.

The primary claim is limited to these named rigid bodies, the fixed camera,
and the configured tabletop ranges. It does not imply arbitrary-object,
cross-camera, real-world visual, cross-operator, or open-world generalization.

## Required skills and executable terminal metrics

All goals declare types, shapes, units, frames, allowed ids or numeric ranges,
and required/optional status in the YAML.

| Skill | Preconditions | Successful terminal metric |
|---|---|---|
| `reach_object` | Safe neutral hand; named entity present and not stale | Approach distance at most 25 mm, orientation error at most 15 degrees, scene disturbance at most 5 mm, for five consecutive control samples |
| `pick_object` | Correct object supported, not held, and inside its approach envelope | Correct object held and lifted at least 40 mm above support for ten samples |
| `place_held_object` | Exactly the requested object held | Object within 25 mm of target, released, and moving at most 20 mm/s for ten samples |
| `push_object_to_target` | Push-compatible object supported and hand inside its approach envelope | Planar target distance at most 35 mm, object still on board, and speed at most 20 mm/s for five samples |
| `press_button` | Named fixture present and hand inside its approach envelope | Correct button reaches requested depth/state, other buttons remain below 2 mm, for three samples |

Timeouts, failure predicates, and retryability are frozen separately from the
success predicates. Workspace and joint-limit violations are abort-only.
Ordinary retryable failures permit at most one deterministic move-to-safe-pose
retry; there is no learned retry or regrasp behavior in Level 4.

## Causal phase contract

The frozen phase vocabulary is:

```text
approach, acquire, lift, stabilize, transport, place, release, settle,
push_contact, fixture_contact, retract
```

The YAML contains ordered per-skill transitions. A transition may use only the
current typed world state, prior typed world state, task request, current
terminal metrics, and the previous online phase. Future frames and audited
annotations are forbidden. The online phase and separately audited annotation
are both stored so disagreement can be reported.

Every phase has a versioned relevance mask for base position, base
orientation, wrist, and fingers. The complete named action remains stored in
every phase. An irrelevant group holds its previous applied value; it is never
dropped from the array or silently trained as an active target.

## Action and safety record

Every control sample stores, in order:

1. requested action and request source (`operator`, `script`, or `policy`);
2. commanded action before safety handling;
3. applied action after clipping, rate limiting, rejection, or fallback;
4. prior commanded and prior applied actions;
5. one safety mask and stable reason code per named field; and
6. request, command, application, and resulting-state timestamps plus the
   measured control interval.

All three action arrays use the complete 27-field Level 1.13 layout: three
absolute world-frame base-position targets, a four-component MuJoCo `wxyz`
absolute base quaternion, and the 20 ordered Shadow Hand wrist/finger actuator
targets. Bounds and rate limits are explicit in the YAML. Applied quaternions
are normalized and sign-continuous with the previous applied quaternion; the
first sample uses the canonical rule `qw >= 0`, followed by the first nonzero
component when `qw == 0`.

This record makes both absolute actions and bounded residual targets
deterministically derivable. Unsafe requested or commanded motion may be kept
as failure evidence, but only the safe applied action may be an expert target.

## Coverage and split ownership

The required minimum is exactly 250 new accepted episodes, with a planning
maximum of 350:

| Data group | Required minimum | Planning range |
|---|---:|---:|
| Reach/object-or-fixture approach | 30 | 30–40 |
| Complete pick/place sequences | 120 | 120–150 |
| Push-to-zone | 40 | 40–55 |
| Button press | 30 | 30–40 |
| Ordinary failures and safe corrections | 30 | 30–50 |
| **Required total** | **250** | **250–350** |

The YAML enumerates 79 required coverage cells: 10 reach, 30 complete
pick/place, 20 push, 10 press, and nine failure/correction cells. Every cell
has exactly one split owner and a `minimum_accepted_by_split` map. Global
surplus cannot repair a missing cell. A complete pick/place sequence is one
episode even though it can produce reach, pick, and place segments; reports
must publish episode and segment counts separately.

Sessions `session_a` and `session_b` are training-owned, `session_c` is
validation-only, and `session_d` is untouched test. These are collection-slot
labels, not permission to fabricate provenance: every actual id must identify
a fresh process start with a new reset/calibration record. A session belongs
to one split for its entire lifetime. Additional sessions must be assigned to
one split before any episode from them is inspected or accepted.

The held-out instances (`block_large`, `cylinder_tall`, `puck_heavy`) and the
held-out goal region (`return_bin_right`) have zero training ownership. Test
data cannot influence tuning, normalization, checkpoint selection, thresholds,
or the Level 4.3 count revision.

## Streams and fixed-camera claim

State, task metrics, all action stages, phase/safety state, and timestamps are
recorded at the control rate. RGB must be available for the final release,
either captured synchronously or reproduced deterministically from saved
MuJoCo state and the frozen render configuration. Each accepted image maps to
one source episode/frame and state timestamp.

The only visual claim uses `workcell_fixed_v1`, a fixed 640×480 camera with
frozen pose and intrinsics. The matrix covers nominal rendering, mild
illumination, partial occlusion, and bounded distractors. Each condition has
train, validation, and test minima and explicit entity coverage. These are
source-episode requirements within the 250-episode budget, not additional
episodes and not duplicate samples across splits.

## Acceptance workflow

Accepted expert episodes must pass schema validation, timestamp alignment,
headless replay, terminal-metric recomputation, operator/recomputed label
agreement, quality thresholds, coverage assignment, and split/session leakage
checks. Accepted episode directories are immutable and append-only. Failed or
rejected attempts remain auditable outside the expert set.

Corrections preserve their triggering episode, failure class, retryability,
intervention interval, trigger source, and final result. A policy checkpoint is
required only when the trigger source is a policy rollout; teleoperation and
scripted sources store a stable not-applicable reason. Unsafe intervals are
never promoted into expert targets.

## Level 3 failure traceability

| Level 3 category | Disposition | Frozen Level 4 response |
|---|---|---|
| Immediate workspace/joint-limit terminations and action-group sensitivity | Accepted | Requested/commanded/applied action stages, previous actions, per-field masks/reasons, phase relevance, explicit bounds/rates, and safe-boundary coverage |
| Contact-only/narrow task set | Accepted | Five typed skills with executable preconditions, goals, phases, terminal metrics, and complete pick/place episodes |
| No genuine session ids or cross-session evidence | Accepted | Mandatory genuine session id; two train sessions, one validation session, one untouched test session; whole-session ownership |
| One cube and narrow fixture/object coverage | Accepted | Three rigid-object families, two instances per family, named fixtures/targets, held-out instances and goal region |
| Fixed starts and narrow goals | Accepted | Interior/near-boundary reset bands, multiple targets/approaches, mass/friction/geometry variation, per-cell minima |
| Success-only learning sets and no correction provenance | Accepted | Separate expert, failure, policy-rollout, and correction streams with nine failure classes and conditional provenance |
| Missing world relations, previous action, safety, phase, and RGB state | Accepted | Typed entity/support/held/contact state, prior actions, causal phase, safety state, and aligned single-camera RGB/annotations |
| Compounding error, temporal ambiguity, and distribution shift | Deferred pending evidence | Preserve timestamps and prior actions; do not add a sequence model until the Level 3.7 causal trigger is met |
| Cross-operator, cross-camera, real-world, arbitrary-object, or learned-recovery claims | Intentionally unsupported | Record operator id but make no claim; fixed camera only; rigid named objects only; deterministic abort/retry only |

Every measured Level 3 gap is therefore mapped, deferred with a trigger, or
explicitly unsupported. No full-scale policy training, qualification, workcell
scene implementation, collection, VLM, planner, or language orchestration is
part of checkpoint 4.0.

## Change control

The specification is immutable by convention once collection begins. Changes
to ids, layouts, bounds, phase transitions, safety codes, coverage ownership,
quality thresholds, or visual conditions require a new config version and a
migration/compatibility note. Level 4.3 may freeze revised counts only from
reported pilot timing, rejection, replay, phase-agreement, coverage, storage,
and safety evidence. Existing accepted episodes are never edited in place.
