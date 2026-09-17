# Training guidance after the reach repair

Planning amendment, September 17, 2026. Applies prospectively to Level 5.3+
and the Level 6/7 handoffs. It does not implement or complete a checkpoint.
`CURRENT_STATUS.md` remains the checkpoint-selection authority.

## Which recipe to follow

`level5_learning_plan.md` and `configs/level5/SHA256SUMS` preserve the completed
5.0 protocol. They are historical, checksum-locked evidence, not the default
recipe for a new skill experiment. In particular, do not silently inherit its
full-action head, near-zero empirical standard deviations, 100-epoch cap or
20-epoch offline-loss patience. Do not edit those files to adopt this guidance.

Before training each future skill, create an independently versioned and
checksum-locked recipe that names its parent protocol and every changed
observation, action, normalization, model and training decision. Retain original
split ownership, safety limits, qualification thresholds and evaluation rules.
Extend recipe-loading and lock checks in that skill's checkpoint, then document
the actual runnable commands. Prose here does not change the existing trainer
or make it accept a new recipe. Frozen v1 runs remain valid reproductions only.

The evidence is in [the reach repair report](level5_reach_repair.md). V5's
selected models passed across three seeds after representation and calibration
repairs. More epochs alone did not fix V4; V5's final epoch was not always its
best controller. These results justify the process below, not a guarantee that
five inputs, two outputs or 1,000 epochs will solve every skill.

## Required pre-training decisions

1. **Audit the state/action interface.** Align observations before action t
   with its target action; verify reset handling, measured versus commanded
   body origins, transforms, units, quaternion conventions and the actual
   control interval. Reconstruct causal phase from current/prior observations,
   never from future success or saved post-action labels. Replayed prefixes
   may establish declared preconditions but cannot control a learned rollout.
2. **Audit what must be learned.** Tabulate action ranges, near-constant inputs,
   variation by phase/cell and correlations between outputs on training data.
   Prefer task/object-relative geometry with calibrated tracking feedback,
   while retaining height, contact, velocity, phase or other available state
   needed to distinguish different expert actions. Adding a field requires a
   versioned causal schema and an equivalent deployment source. Do not feed
   retrospective held/success labels, hidden expert stage or instance ordinals.
3. **Make scales explicit.** Use fixed physical scales, or training-fitted
   statistics with documented physical floors and constant-feature handling.
   Validation checks the transform but never fits its statistics. Inspect
   normalized magnitudes on validation rollouts to catch distribution shifts.
4. **Choose an action representation per skill.** Start with bounded deltas
   from the previous applied command and a small feed-forward network when
   current state is sufficient. Explicitly hold a channel only when training
   and task semantics support that invariant across relevant phases. Preserve
   needed finger closure, release, rotation and bidirectional corrections.
   Geometric decoders must reject incompatible targets and expose their units,
   assumptions and zero-output behaviour. Do not impose reach's straight-line,
   fixed-posture decoder on contact skills.
5. **Freeze the experiment.** Record inputs, active outputs, decoder, loss,
   optimizer, physical scales, sampling, three seeds, compute budget, early/late
   candidate schedule and selection ranking before runs. A two-layer 128-unit
   MLP remains a comparison starting point, not a qualification guarantee.
   Use balanced cell/episode sampling; measure phase coverage before deciding
   whether phase balancing or loss weights are necessary. Separate the effect
   of representation changes from optimizer/model/data changes.

## Skill-specific questions to resolve

These are proposed audits and representation hypotheses, not implemented or
qualified controllers. Exact fields must be supported by the released data and
the eventual state/perception interface.

| Skill | State/action information to test | Failure that must remain visible |
|---|---|---|
| Pick (5.3) | Object-relative hand pose, measured tracking error, finger aperture/joints, available contact/retention evidence, lift height and causal acquire/lift/hold phase. Learn varying closure and any required wrist/orientation motion. | Closing before contact, lifting without the object, excessive closure, slip/drop; a moving hand is not proof of acquisition. |
| Place (5.4) | Held-object-to-target geometry, grasp offset, clearance/height, measured retention, object motion and causal transport/lower/release/settle phase. Keep release channels active. | Correct hand endpoint with misplaced object, early release, collision, bounce or motion after release. |
| Push (5.5) | Object-to-zone error plus hand-to-object contact geometry, object motion and causal approach/contact/push state. Investigate lateral error/contact loss before restricting direction. | Hand reaches the zone while the object stalls, slips sideways or leaves the board. The old expert-plus-residual probe is not standalone-policy evidence. |
| Press (5.5) | Fixture-frame alignment, measured depth/motion, tracking lag and causal approach/press/dwell/retract state. Represent pressing and release directions. | Wrong-button contact, shallow press, repeated pushing after the limit, or failure to release. |

Optional dial work still requires promotion and its own evidence; reach success
does not authorize it. If fields or coverage are absent, record the deficiency
and request a versioned data/schema amendment rather than inventing labels or
editing the immutable release.

## Training, selection and failure diagnosis

- Complete the declared training budget; do not stop solely on an offline-loss
  plateau before measuring control behaviour. A safety/integrity or numerical
  failure may stop a run, but it stays a failed attempt with a recorded reason.
  Additional budget or a new stopping rule is a new validation-only experiment.
  Reach's 1,000 epochs are evidence for reach, not a universal minimum.
- Retain early, late, best-offline and final candidates. Use the full frozen
  closed-loop validation matrix for selections and compare every seed under
  the same rule. Smoke screens cannot qualify a policy. Select safety first,
  then task/worst-group performance using the declared ranking; never select
  by lowest loss, final epoch, pooled seed average or best seed alone.
- Diagnose the first divergence: feature scale/frame error, ambiguous phase,
  unsupported action channel, contact transition, feedback accumulation or
  missing coverage. Use training/validation-only comparisons and preserve
  negative outcomes. Fix an identified interface/representation problem before
  concluding that more epochs, a larger network or more data is the remedy.
- A history model requires evidence that similar current observations need
  different actions and that causal history resolves the ambiguity. Action
  chunking requires its own measured hypothesis and must account for feedback,
  contact changes, interruption and latency; open-loop chunks are not an
  automatic improvement. Preserve the existing approval boundary for RL,
  diffusion and foundation policies.
- Corrective learning uses only eligible scripted correction intervals with
  preserved lineage/splits. Compare matched nominal/correction arms with
  declared mixture and update budgets; an abort-only failure is not a valid
  action target. Expert-assisted residuals need expert/zero-residual ablations
  and cannot be relabeled standalone policies.
- Reserve test/confirmation until method selection is frozen. If a test has
  already informed development, retain that disclosure and preregister new
  confirmation; fresh reset seeds do not make an exposed family unseen.
  Never rerun a failing confirmation until it passes or drop failed seeds.

## Perception, physical verification and composition

For 5.6, validate metric pose, body origins and timestamps as well as image
metrics. Image augmentation cannot change metric labels without a corresponding
geometric transform; keep episode ownership across all render variants. Select
confidence/staleness handling from training/validation, not policy test cases.

For 5.7, run the exact selected state-policy bundle first, including transforms,
normalization, action decoder and cadence. Replace only the declared entity
observation source; do not leak truth through derived goals, features or phase.
Missing deployment observability is a schema problem, not permission for a
truth fallback. Assess pose noise, bias, dropout and latency on validation with
versioned, physically justified conditions. Any robustness retraining is a
separate experiment, not an edit to the paired comparison or a test-driven fix.

For every skill, recompute terminal results from measured physical pose, object
motion/contact, retention and required dwell, as applicable. If an existing
metric checks a commanded target, retain its historical meaning and add an
independent physical audit rather than silently redefining it. Numerical and
viewer gates both remain required where specified.

For 5.11, validate skill transitions before pilot test runs using actual
predecessor terminal states: a scripted standalone start does not establish
composability. Preserve object identity, held state, coordinate frame and prior
applied command; initialize causal phase/history from real state. Reject an
unsupported entry state rather than resetting, teleporting or consulting an
expert to rescue the sequence. These checks do not authorize learned recovery.

## Release and runtime handoff

A policy is its selected weights **plus** its named feature transform, physical
calibration, scales, active action mask, decoder, causal phase/history rules,
control cadence and safety/terminal contracts. Bind all of these to versions,
code/config digests and selection evidence. Verify offline/live feature parity,
zero-output behaviour and physical outcomes before a released executor loads
the bundle; loading weights alone is insufficient.

Level 6 diagrams and results must distinguish learned behaviour, engineered
geometry, optional expert assistance and supervision. Report each seed, actual
unique reset counts, reused family exposure and negative results. Optional
dataset exports preserve state-before-action alignment and the decoder/target
semantics; normalized network outputs are not physical actuator commands.
Level 7 consumes these qualified bundles through typed skill requests. It
cannot replace the decoder, tune weights online or convert a state-only pass
into perception qualification.
