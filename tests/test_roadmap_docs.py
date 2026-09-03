from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_current_status_advances_to_level41_after_dataset_plan_freeze() -> None:
    status = read("docs/CURRENT_STATUS.md")

    assert (
        "Level 4 — Comprehensive Multi-Session Dataset Collection and "
        "Versioned Release"
    ) in status
    assert "`docs/progress_level_4.md`" in status
    assert "## Last Completed Checkpoint\n\nLevel 4.0" in status
    assert "## Next Target Checkpoint\n\nLevel 4.1" in status


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
    checkpoints = re.findall(r"^## Level 4\.(\d+) —", level4, flags=re.MULTILINE)

    assert checkpoints == [str(index) for index in range(10)]
    assert level4.count("### Commands") == 10
    assert level4.count("### Pass criteria") == 10
    assert "Level 3 failure -> Level 4 requirement traceability table" in level4
    assert "Stop until the user confirms" in level4
    assert "Clean-clone retrieval and SHA-256 verification" in level4


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
    assert "Level 4 is active at checkpoint 4.0" in overview
    assert "at least four genuine sessions" in overview
    assert "Level 3 — Learning feasibility" in overview
    assert "Level 4 — Comprehensive skill dataset" in overview
    assert "Level 5 — Full-scale skill learning and qualification" in overview
    assert "Level 7 — Language-guided orchestration" in overview
    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 1_000
