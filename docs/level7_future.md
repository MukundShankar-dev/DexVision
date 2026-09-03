# Future Level 7 — Language-Guided Skill Orchestration

Level 7 is intentionally outside the current implementation sequence. It uses
the artifacts produced by the earlier levels; it does not compensate for
missing skills by asking a language model to generate raw controls.

DexVision / Hand2Bot should first complete:

```text
Level 2: resettable task environments, demonstration data, replay, and quality
Level 3: learning-feasibility baselines on the existing narrow datasets
Level 4: comprehensive multi-session dataset and frozen training splits
Level 5: full-scale skill learning, qualification, runtime, and scripted pilots
Level 6: robustness, reproducibility, results, and portfolio packaging
```

## Intended boundary

```text
user request
  -> LLM or deterministic task planner
  -> typed skill plan
  -> deterministic orchestration supervisor
  -> Level 5 supervised skill executor
  -> closed-loop learned policy
  -> MuJoCo, and only later validated hardware
```

The planner chooses skills and parameters. It must not output joint angles,
actuator targets, or high-rate base/wrist/finger commands. The deterministic
supervisor owns:

```text
parameter and schema validation
world-state and precondition checks
safety limits
timeouts and cancellation
success/failure monitoring
retry and transition bookkeeping
idempotency
```

The first Level 7 prototype should execute scripted plans against mock or
deterministic skills. Qualified learned policies should replace the mocks
behind the same interface one at a time. An in-process Python/tool interface is
enough initially; an HTTP API is not a prerequisite.

## Workcell pilot requests

Level 7 must demonstrate more than one memorized showcase. Requests should map
to the coherent tabletop-workcell pilots already validated in Level 5:

```text
"Clear the loose parts from the workspace into their return bins."
"Put this part on the inspection pad and press Start."
"Set up the workspace for job B using the marked target positions."
"Prepare inspection job B: put the blue cylinder on the inspection pad,
 place the red block in the left tray slot, optionally set the dial to
 45 degrees, then press Start."
```

These requests reuse five required operational skills: `reach_object`,
`pick_object`, `place_held_object`, `push_object_to_target`, and
`press_button`. `rotate_dial` is optional. Kitchen tasks, cutting, pouring,
tools, deformables, arbitrary household scenes, and end-to-end VLM control are
deferred beyond the first complete project.

## Planner inputs and outputs

The planner receives symbolic world state and structured results, not the
continuous control stream. The API may expose:

```python
list_skills() -> list[SkillSummary]
get_world_state() -> WorldState
start_skill(name, parameters, request_id) -> ExecutionHandle
get_skill_status(execution_id) -> SkillResult
cancel_skill(execution_id) -> SkillResult
```

Each request identifies the skill and compatible version, a unique caller
request id/idempotency key, typed parameters, object/target identifiers, units,
coordinate frames, and an optional timeout. Results distinguish:

```text
running
succeeded
failed
timed_out
cancelled
rejected before execution
```

The planner must respect skill preconditions and terminal-state envelopes.
For example, `pick_object(object_id)` requires a reachable rigid object, and
`place_held_object(target_pose_or_receptacle)` must verify that an object
remains held before execution.

## Required Level 5 exports

```text
qualified policy checkpoints and digests
policy input/output and compatibility schemas
typed skill parameter schemas
task and world-state metadata
precondition and terminal-state schemas
structured SkillResult examples
held-out rollout metrics and known limitations
versioned skill cards and registry data
```

The detailed runtime contract is in `docs/skill_orchestration_future.md`.

Do not implement Level 7 orchestration until the user explicitly advances the
project beyond Levels 3–6.
