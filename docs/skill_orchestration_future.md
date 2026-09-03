# Future Level 7 Skill Orchestration Contract

Level 7 is future language-guided orchestration work. It is intentionally not
part of the current DexVision / Hand2Bot implementation sequence.

The current repository should first produce reliable building blocks:

```text
Level 2 task demonstrations
Level 2 replay and quality reports
Level 3 learning-feasibility results
Level 4 comprehensive multi-session dataset releases
Level 5 qualified policy checkpoints, skill cards, runtime, and rollout metrics
Level 6 documentation, robustness, and reproducibility artifacts
```

A future Level 7 system can consume qualified Level 5 policies and skill cards
as reusable skills. The first complete system exposes this bounded set:

```text
reach_object(object_id, approach_pose)
pick_object(object_id)
place_held_object(target_pose_or_receptacle)
push_object_to_target(object_id, target_zone)
press_button(button_id)
rotate_dial(dial_id, target_angle), optional
```

Grasp, lift, hold, transport, release, and slide remain internal policy phases
or measurements rather than separate planner-visible calls. Learned regrasp,
dropped-object recovery, arbitrary-object grasping, hinged-lid operation, tool
use, cutting, pouring, and deformables are deferred.

That orchestration layer can choose which skill to run, chain skills across a
longer task, monitor success/failure, retry failed subtasks, and expose a
higher-level task interface. This is separate from behavior cloning itself:
Level 3 proves feasibility, Level 4 supplies comprehensive data, and Level 5
learns and qualifies individual policies; Level 7 decides how to compose them.

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
envelopes. For example, `pick_object(object_id)` requires a reachable rigid
object, and `place_held_object(target_pose_or_receptacle)` must verify that an
object is still held before execution.

Example future requests:

```text
Clear the loose parts from the workspace into their return bins.
Put the blue cylinder on the inspection pad and press Start.
Set up the workspace for job B using the marked target positions.
Prepare inspection job B: place the blue cylinder on the inspection pad,
place the red block in the left tray slot, optionally set the dial to 45
degrees, then press Start.
```

Example inspection-job decomposition in a constrained environment:

```text
pick_object(blue_cylinder)
-> place_held_object(inspection_pad)
-> pick_object(red_block)
-> place_held_object(left_tray_slot)
-> rotate_dial(dial, 45 degrees), if qualified
-> press_button(start)
```

That decomposition is illustrative. Level 5 must first validate workspace
clearing, inspection-station operation, and workspace setup as deterministic
scripted pilots. The supervisor handles failure by moving to a safe pose,
retrying once when allowed, then aborting with a structured reason; a learned
recovery policy is not required.

Before connecting an LLM, Level 7 should validate the orchestration contract
with scripted plans and mock or deterministic skills. Learned policies should
then be substituted behind the same supervised executor one skill at a time.

Level 7 can live in a separate repository. This repo should eventually export
the Level 5/6 artifacts that such a repo would need:

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

Do not implement Level 7 orchestration in this repository yet.
