from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_current_status_matches_active_progress_file() -> None:
    status = read("docs/CURRENT_STATUS.md")
    path = re.search(r"## Current Progress File\n\n`([^`]+)`", status).group(1)
    progress = read(path)
    current_level = re.search(r"## Current Level\n\nLevel (\d+)", status).group(1)
    assert path == f"docs/progress_level_{current_level}.md"
    active_positions = []
    for field in ("Last Completed Checkpoint", "Next Target Checkpoint"):
        title = re.search(rf"## {field}\n\n([^\n]+)", status).group(1)
        if title.startswith("None"):
            continue
        level = re.match(r"Level (\d+)\.", title).group(1)
        owner = progress if level == current_level else read(f"docs/progress_level_{level}.md")
        assert f"## {title}" in owner
        if level == current_level:
            active_positions.append(progress.index(f"## {title}"))
    if len(active_positions) == 2:
        assert active_positions[0] < active_positions[1]
    assert "hammer-curl" in status


def test_seven_level_roadmap_has_distinct_responsibilities() -> None:
    agents = read("AGENTS.md")
    level3 = read("docs/progress_level_3.md")
    level4 = read("docs/progress_level_4.md")
    level5 = read("docs/progress_level_5.md")
    level6 = read("docs/progress_level_6.md")
    level7 = read("docs/level7_future.md")

    assert "organized into seven levels" in agents
    assert "# Progress Level 3 — Learning Feasibility on Existing Data" in level3
    assert "# Progress Level 4 — Comprehensive Workcell Skill Dataset" in level4
    assert "# Progress Level 5 — Workcell Skill Learning and Qualification" in level5
    assert "# Progress Level 6 — Portfolio Polish" in level6
    assert "# Future Level 7 — Language-Guided Skill Orchestration" in level7


def test_level5_requires_bounded_workcell_skills_and_pilots() -> None:
    level5 = read("docs/progress_level_5.md")

    assert "`reach_object`" in level5
    assert "`pick_object`" in level5
    assert "`place_held_object`" in level5
    assert "`push_object_to_target`" in level5
    assert "`press_button`" in level5
    assert "Workspace clearing" in level5
    assert "Inspection-station operation" in level5
    assert "Workspace setup" in level5
    assert "20 frozen reset seeds per core pilot" in level5
    assert "combined work-order success rate >= 0.50" in level5
    assert "learned regrasp/drop recovery" in level5
    assert "tools, cutting, pouring, liquids, or deformables" in level5


def test_level4_uses_realistic_collection_envelope() -> None:
    level4 = read("docs/progress_level_4.md")

    assert "Required total | 250 | 250–350" in level4
    assert "at least 4 genuine sessions" in level4
    assert "session C: validation only" in level4
    assert "session D: untouched cross-session test" in level4
    assert "Complete pick/place sequences | 120 | 120–150" in level4
    assert "counts and segment counts" in level4
    assert "renaming" in level4
    assert "does not create multiple sessions" in level4


def test_level4_freezes_action_phase_correction_and_visual_contracts() -> None:
    level4 = read("docs/progress_level_4.md")
    contracts = read("docs/module_contracts.md")

    for phrase in (
        "requested action and request source",
        "commanded action before safety handling",
        "applied action after clipping",
        "minimum_accepted_by_split",
        "online phase state machine",
        "mild illumination variation",
        "partial occlusion",
        "bounded workcell distractors",
    ):
        assert phrase in level4
    assert (
        "require source policy/checkpoint only when source category is policy_rollout"
        in level4
    )
    assert "source_policy_checkpoint: required only when trigger_source" in contracts
    assert "Quaternions are normalized and sign-continuous" in contracts


def test_level4_checkpoints_are_execution_ready() -> None:
    level4 = read("docs/progress_level_4.md")
    checkpoints = re.findall(
        r"^## Level 4\.(\d+(?:[A-Z])?) —", level4, flags=re.MULTILINE
    )

    assert checkpoints == ["0", "1", "2", "3", "4", "5A", "5B", "6", "7", "8", "9"]
    assert len(re.findall(r"^### Commands$", level4, flags=re.MULTILINE)) == 11
    assert len(re.findall(r"^### Pass criteria$", level4, flags=re.MULTILINE)) == 11
    assert "Level 3 failure -> Level 4 requirement traceability table" in level4
    assert "Stop until the user confirms" in level4
    assert "Clean-clone retrieval and SHA-256 verification" in level4


def test_level43_pivot_is_incremental_and_learning_gated() -> None:
    level4 = read("docs/progress_level_4.md")
    status = read("docs/CURRENT_STATUS.md")
    contracts = read("docs/module_contracts.md")

    subcheckpoints = re.findall(
        r"^#### Level 4\.3([A-I]) —", level4, flags=re.MULTILINE
    )
    assert subcheckpoints == list("ABCDEFGHI")
    assert "Level 4.3I is complete" in status
    assert "expert.reset(task, world_state)" in level4
    assert "applied_action - requested_action" in level4
    assert "20 scripted button successes" in level4
    assert "small MLP" in level4
    assert "simulator state only" in level4
    assert "Do not add OMPL, RRT" in level4
    assert "Do not add an LLM, VLM" in level4
    assert "requested_action is the nominal scripted action" in contracts
    assert "114 required accepted episodes" in " ".join(level4.split())
    assert "111 scripted nominal episodes" in level4
    assert "zero live-control episodes" in level4
    assert "zero policy-rollout" in level4
    assert "Level 4.4 is complete" in status
    assert "Last Completed is now 4.4" in status
    assert "Next Target is 4.5A" in status
    assert "1,112-episode release-candidate floor" in contracts
    assert "All 62 nominal cells" in level4
    assert "4/8/16-per-cell probes" in level4
    assert "Level 4.4 cannot start until the user accepts" in contracts


def test_level5_checkpoints_freeze_training_and_qualification() -> None:
    level5 = read("docs/progress_level_5.md")
    checkpoints = re.findall(r"^## Level 5\.(\d+) —", level5, flags=re.MULTILINE)

    assert checkpoints == [str(index) for index in range(13)]
    assert level5.count("### Commands") == 13
    assert level5.count("### Pass criteria") == 13
    assert "3 independent training seeds" in level5
    assert "at least 30 held-out state-grounded rollouts" in level5
    assert "workspace violations = 0" in level5
    assert "joint-limit safety violations = 0" in level5
    assert "Level 3.4 result must be explicitly handled" in level5
    assert "Do not continue to pilots" in level5


def test_level35b_repairs_checkpoint_selection_without_rewriting_v1() -> None:
    level3 = read("docs/progress_level_3.md")
    normalized_level3 = " ".join(level3.split())

    assert "Level 3.5B — Checkpoint-Selection Repair and Cross-Task Baselines" in level3
    assert "lowest offline validation loss" in level3
    assert "break an exact tie" in level3
    assert "unchanged 35-run matrix" in normalized_level3
    assert "Preserve and compare the Level 3.4 v1 checkpoint/report" in level3
    assert "must not choose an epoch" in level3
    assert "Do not add an unplanned hyperparameter sweep" in level3


def test_project_authored_markdown_has_no_old_orchestration_number_or_path() -> None:
    authored_paths = [ROOT / "AGENTS.md", ROOT / "README.md"]
    authored_paths.extend((ROOT / "datasets").glob("*.md"))
    authored_paths.extend((ROOT / "docs").glob("*.md"))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in authored_paths)

    assert "docs/level5_future.md" not in combined
    assert "docs/level6_future.md" not in combined
    assert "Future Level 5" not in combined
    assert "Level 5 orchestration" not in combined
    assert "Level 6 orchestration" not in combined


def test_project_overview_source_and_pdf_exist() -> None:
    overview = read("docs/project_overview.md")
    pdf_path = ROOT / "DexVision Project Overview.pdf"

    assert "Levels 1 through 3 are complete" in overview
    assert "Level 4 is complete through checkpoint 4.9" in overview
    assert "Level 4 is complete" in read("README.md")
    assert "60 scripted core episodes" in overview
    assert "Level 3 — Learning feasibility" in overview
    assert "Level 4 — Comprehensive scripted skill dataset" in overview
    assert "Level 5 — Full-scale skill learning and qualification" in overview
    assert "Level 7 — Language-guided orchestration" in overview
    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 1_000


def test_active_roadmap_retires_live_hand_control() -> None:
    agents = read("AGENTS.md")
    level4 = read("docs/progress_level_4.md")
    level5 = read("docs/progress_level_5.md")
    dataset_plan = read("docs/level4_dataset_plan.md")
    normalized_agents = " ".join(agents.split())
    normalized_level5 = " ".join(level5.split())
    normalized_dataset_plan = " ".join(dataset_plan.split())

    assert "not an active Level 4+ data or control path" in normalized_agents
    assert "Do not use live hand-pose control for required data" in level4
    assert "never require a camera, hand tracker, or human control input" in level4
    assert "outside the Level 5 system boundary" in normalized_level5
    assert "only active Level 4 data path" in normalized_dataset_plan
