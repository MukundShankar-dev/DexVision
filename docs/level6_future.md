# Future Level 6 — Language-Guided Skill Orchestration

Level 6 is intentionally outside the current implementation sequence. It uses
the artifacts produced by the earlier levels; it does not compensate for
missing skills by asking a language model to generate raw controls.

DexVision / Hand2Bot should first complete:

```text
Level 2: resettable task environments, demonstration data, replay, and quality
Level 3: learning-feasibility baselines on the existing narrow datasets
Level 4: comprehensive multi-session dataset and qualified skill library
Level 5: robustness, reproducibility, results, and portfolio packaging
```

## Intended boundary

```text
user request
  -> LLM or deterministic task planner
  -> typed skill plan
  -> deterministic orchestration supervisor
  -> Level 4 supervised skill executor
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

The first Level 6 prototype should execute scripted plans against mock or
deterministic skills. Qualified learned policies should replace the mocks
behind the same interface one at a time. An in-process Python/tool interface is
enough initially; an HTTP API is not a prerequisite.

## Diverse pilot requests

Level 6 must demonstrate more than one memorized showcase. Candidate requests
map to the scripted pilots already validated in Level 4:

```text
"Sort these objects into the matching bins."
"Clear the workspace into the tray."
"Press the blue button, then turn the dial to 90 degrees."
"Assemble a cheese sandwich on the plate."
"Set a place with the cup, plate, and utensil."
```

The sandwich uses rigid ingredient proxies and is one pilot, not the final
definition of the project. Real tomato slicing is not an acceptance task for
the present setup. A guided, pre-segmented rigid cutting proxy may be explored
later, but it must not be represented as safe deformable-food cutting.

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
For example, `grasp_object(object_id)` cannot start from arbitrary hand state,
and `place_object(object_id, target_pose)` must verify that the object remains
held.

## Required Level 4 exports

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

Do not implement Level 6 orchestration until the user explicitly advances the
project beyond Levels 3–5.
