# Future Level 5 Skill Orchestration

Level 5 is intentionally out of scope for this repository's current staged
plan.

DexVision / Hand2Bot should first finish:

```text
Level 2: resettable task environments, skill-demo recording, replay, validation, and quality filtering
Level 3: skill policies trained from saved demos, plus skill cards
Level 4: polish, robustness, reproducibility, and portfolio packaging
```

A future Level 5 project can live in a separate repository. That repo can treat
trained Level 3 policies and exported skill cards as reusable skills, for
example:

```text
reach_object policy
reach_touch_target policy
button_press policy
push_object_to_target policy
grasp_object policy
pinch_lift_object policy
place_object policy
release_object policy
rotate_dial policy
```

The future orchestration layer can chain learned skills, choose task order,
monitor success/failure, retry failed subtasks, and expose higher-level task
interfaces. The LLM/planner should choose and parameterize skills, not output
raw robot joint or actuator actions.

The LLM must call a typed skill interface through a deterministic supervisor.
The supervisor, not the LLM, owns parameter validation, precondition checks,
safety limits, success/failure monitoring, timeouts, cancellation, and retry
bookkeeping. It should invoke the supervised Level 3 skill executor and return
a structured result with distinct succeeded, failed, timed-out, cancelled, and
rejected states.

The first Level 5 prototype should use scripted plans plus mock or
deterministic skills to validate world-state updates and skill transitions.
Learned skills should then replace those implementations behind the same
interface one at a time.

Example future request:

```text
I want a sandwich.
```

Example future decomposition in a constrained sandwich environment:

```text
reach_object(bread) -> grasp_object(bread) -> place_object(plate) -> release_object()
```

This example is only a future orchestration sketch. The current Level 2/3 plan
does not solve sandwich-making. Object perception, symbolic world state,
ingredient/state tracking, long-horizon planning, and recovery policies are
Level 5 concerns, not Level 2/3 requirements.

Level 5 should consume exported policy checkpoints, typed parameter schemas,
task metadata, checkpoint compatibility data, structured skill-result
contracts, and skill cards from this repo instead of adding
planner/orchestrator code here now.

The detailed interface and artifact boundary is documented in
`docs/skill_orchestration_future.md`.

Do not implement Level 5 orchestration in DexVision at this stage.
