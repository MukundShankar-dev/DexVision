from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_current_status_keeps_level3_loader_as_next_target() -> None:
    status = read("docs/CURRENT_STATUS.md")

    assert "Level 3 — Learning Feasibility on Existing Data" in status
    assert "`docs/progress_level_3.md`" in status
    assert "Level 3.0 — Roadmap Rebaseline" in status
    assert "Level 3.1 — Goal-Conditioned Per-Skill Dataset Loader" in status


def test_six_level_roadmap_has_distinct_responsibilities() -> None:
    agents = read("AGENTS.md")
    level3 = read("docs/progress_level_3.md")
    level4 = read("docs/progress_level_4.md")
    level5 = read("docs/progress_level_5.md")
    level6 = read("docs/level6_future.md")

    assert "organized into six levels" in agents
    assert "# Progress Level 3 — Learning Feasibility on Existing Data" in level3
    assert "# Progress Level 4 — Comprehensive Skill Dataset and Skill Library" in level4
    assert "# Progress Level 5 — Portfolio Polish" in level5
    assert "# Future Level 6 — Language-Guided Skill Orchestration" in level6


def test_level4_requires_diverse_pilots_and_bounds_cutting_claim() -> None:
    level4 = read("docs/progress_level_4.md")

    assert "Sort and pack" in level4
    assert "Clear the workspace" in level4
    assert "Operate a control panel" in level4
    assert "Assemble a simple sandwich proxy" in level4
    assert "At least three materially different pilots must pass" in level4
    assert "Real tomato cutting is not a core pilot" in level4


def test_project_authored_markdown_has_no_old_orchestration_number_or_path() -> None:
    authored_paths = [ROOT / "AGENTS.md", ROOT / "README.md"]
    authored_paths.extend((ROOT / "datasets").glob("*.md"))
    authored_paths.extend((ROOT / "docs").glob("*.md"))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in authored_paths)

    assert "docs/level5_future.md" not in combined
    assert "Future Level 5" not in combined
    assert "Level 5 orchestration" not in combined


def test_project_overview_source_and_pdf_exist() -> None:
    overview = read("docs/project_overview.md")
    pdf_path = ROOT / "DexVision Project Overview.pdf"

    assert "Level 3 — Learning feasibility" in overview
    assert "Level 4 — Comprehensive dataset and qualified skills" in overview
    assert "Level 6 — Language-guided orchestration" in overview
    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 1_000
