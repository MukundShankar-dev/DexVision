from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_current_status_keeps_level3_as_active_roadmap() -> None:
    status = read("docs/CURRENT_STATUS.md")

    assert "Level 3 — Learning Feasibility on Existing Data" in status
    assert "`docs/progress_level_3.md`" in status
    assert "## Last Completed Checkpoint" in status
    assert "## Next Target Checkpoint" in status


def test_seven_level_roadmap_has_distinct_responsibilities() -> None:
    agents = read("AGENTS.md")
    level3 = read("docs/progress_level_3.md")
    level4 = read("docs/progress_level_4.md")
    level5 = read("docs/progress_level_5.md")
    level6 = read("docs/progress_level_6.md")
    level7 = read("docs/level7_future.md")

    assert "organized into seven levels" in agents
    assert "# Progress Level 3 — Learning Feasibility on Existing Data" in level3
    assert "# Progress Level 4 — Comprehensive Skill Dataset" in level4
    assert "# Progress Level 5 — Full-Scale Skill Learning and Qualification" in level5
    assert "# Progress Level 6 — Portfolio Polish" in level6
    assert "# Future Level 7 — Language-Guided Skill Orchestration" in level7


def test_level5_requires_bounded_workcell_skills_and_pilots() -> None:
    level5 = read("docs/progress_level_5.md")

    assert "reach_object(object_id, approach_pose)" in level5
    assert "pick_object(object_id)" in level5
    assert "place_held_object(target_pose_or_receptacle)" in level5
    assert "Workspace clearing" in level5
    assert "Inspection-station operation" in level5
    assert "Workspace setup" in level5
    assert "All three pilots must pass" in level5
    assert "learned regrasp or dropped-object recovery policy" in level5
    assert "Kitchen tasks, cutting, pouring" in level5


def test_level4_uses_realistic_collection_envelope() -> None:
    level4 = read("docs/progress_level_4.md")

    assert "250–350 new" in level4
    assert "accepted episodes in total" in level4
    assert "at least three genuine sessions" in level4
    assert "100–150 complete pick/place sequences" in level4
    assert "do not demand a separate 100 episodes" in level4


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

    assert "Level 3 — Learning feasibility" in overview
    assert "Level 4 — Comprehensive skill dataset" in overview
    assert "Level 5 — Full-scale skill learning and qualification" in overview
    assert "Level 7 — Language-guided orchestration" in overview
    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 1_000
