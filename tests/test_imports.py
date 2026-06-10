from __future__ import annotations

import subprocess
import sys

from dexvision.apps import health_check


def test_package_imports() -> None:
    import dexvision
    import dexvision.apps

    assert dexvision is not None
    assert dexvision.apps is not None


def test_health_check_reports_all_dependencies_without_crashing() -> None:
    checks = health_check.run_checks()

    assert [check.name for check in checks] == [
        "Python",
        "NumPy",
        "OpenCV",
        "MediaPipe",
        "MuJoCo",
        "PyTorch",
    ]
    assert checks[0].ok
    assert checks[-1].optional


def test_health_check_formats_ok_and_missing_lines() -> None:
    ok_line = health_check.format_check(
        health_check.DependencyCheck("Example", "example", True, version="1.2.3")
    )
    missing_line = health_check.format_check(
        health_check.DependencyCheck(
            "Future Package",
            "future_package",
            False,
            message="Install package providing 'future_package'.",
            optional=True,
        )
    )

    assert ok_line.startswith("OK")
    assert "Example 1.2.3" in ok_line
    assert missing_line.startswith("MISSING")
    assert "Future Package optional" in missing_line
    assert "Install package providing 'future_package'." in missing_line


def test_health_check_runs_as_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.health_check"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "DexVision health check" in result.stdout
    assert "No camera, MuJoCo model, or GUI was opened." in result.stdout
