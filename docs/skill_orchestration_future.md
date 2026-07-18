# Future Skill Orchestration

Level 5 is future skill-library and orchestration work. It is intentionally not
part of the current DexVision / Hand2Bot implementation plan.

The current repository should first produce reliable building blocks:

```text
Level 2 task demonstrations
Level 2 replay and quality reports
Level 3 learned policy checkpoints
Level 3 rollout metrics
Level 4 documentation and reproducibility artifacts
```

A future Level 5 system can consume trained Level 3 policies and skill cards as
reusable skills. Example skills could include:

```text
free_space_gesture, calibration only unless explicitly promoted
reach_object
reach_touch_target
button_press
push_object_to_target
grasp_object
pinch_lift_object
place_object
release_object
rotate_dial
spread_over_surface, optional later
tool_use_simple, optional later
```

That orchestration layer can choose which skill to run, chain skills across a
longer task, monitor success/failure, retry failed subtasks, and expose a
higher-level task interface. This is separate from behavior cloning itself:
Level 3 learns individual policies, while Level 5 would decide how to compose
them.

The LLM/planner should select and parameterize learned skills. It should not
output raw joint actions, raw actuator targets, or per-timestep base/finger
commands.

The intended future runtime boundary is:

```text
LLM/planner
  -> typed skill request
  -> deterministic orchestration supervisor
  -> DexVision supervised skill executor
  -> closed-loop learned policy
  -> MuJoCo or robot
```

The LLM should receive symbolic world state and structured skill results, not
the high-rate continuous control stream. The deterministic supervisor should
own parameter validation, precondition checks, timeouts, cancellation, safety
limits, success/failure monitoring, retry policy, and transition bookkeeping.

A future orchestration API can start as an in-process Python/tool interface; it
does not need to begin as an HTTP service:

```python
list_skills() -> list[SkillSummary]
get_world_state() -> WorldState
start_skill(name, parameters, request_id) -> ExecutionHandle
get_skill_status(execution_id) -> SkillResult
cancel_skill(execution_id) -> SkillResult
```

Each request must identify the skill name and compatible version, a unique
caller request id/idempotency key, typed parameters, object/target identifiers,
units and coordinate frames, and an optional execution timeout. Repeating the
same idempotency key must not start a duplicate physical action.

Each result must distinguish:

```text
running
succeeded
failed
timed_out
cancelled
rejected before execution
```

The orchestrator must plan from skill preconditions and terminal-state
envelopes. For example, `grasp_object(object_id)` cannot be assumed to start
from arbitrary robot state, and `place_object(object_id, target_pose)` must
verify that the object is still held before execution.

Example future request:

```text
I want a sandwich.
```

Example future decomposition in a constrained environment:

```text
reach_object(bread) -> grasp_object(bread) -> place_object(plate) -> release_object()
```

That decomposition is illustrative, not a claim that the current repo solves
sandwich-making. Object perception, symbolic world state, ingredient tracking,
long-horizon planning, and robust recovery are Level 5 concerns.

Before connecting an LLM, Level 5 should validate the orchestration contract
with scripted plans and mock or deterministic skills. Learned policies should
then be substituted behind the same supervised executor one skill at a time.

Level 5 can live in a separate repository. This repo should eventually export
the artifacts that such a repo would need:

```text
policy checkpoints
policy input/output schema
task metadata
task success metrics
skill cards and descriptions
example rollout metrics
typed parameter schemas
precondition and terminal-state schemas
checkpoint digests and compatibility metadata
structured SkillResult examples
```

Do not implement Level 5 orchestration in this repository yet.
