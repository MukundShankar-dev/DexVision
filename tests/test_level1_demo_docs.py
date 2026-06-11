from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_level1_demo_command_and_limitations() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Level 1 Demo" in readme
    assert (
        "mjpython -m dexvision.apps.run_level1_teleop --camera-id 0 "
        "--show-camera-window --print-interval 10"
    ) in readme
    assert "Short demo video or GIF" in readme or "short demo video or GIF" in readme
    assert "## Known Limitations" in readme
    assert "--assume-mirrored-input" in readme
