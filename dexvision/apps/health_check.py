"""Dependency health check for DexVision."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class DependencyCheck:
    """Result for one import or runtime dependency check."""

    name: str
    import_name: str | None
    ok: bool
    version: str | None = None
    message: str | None = None
    optional: bool = False


DEPENDENCIES: tuple[tuple[str, str | None, bool], ...] = (
    ("Python", None, False),
    ("NumPy", "numpy", False),
    ("OpenCV", "cv2", False),
    ("MediaPipe", "mediapipe", False),
    ("MuJoCo", "mujoco", False),
    ("PyTorch", "torch", True),
)


def check_python() -> DependencyCheck:
    """Check the Python runtime version without importing external modules."""

    version = ".".join(str(part) for part in sys.version_info[:3])
    ok = sys.version_info >= (3, 11)
    message = None if ok else "Python 3.11 or newer is required."
    return DependencyCheck("Python", None, ok, version=version, message=message)


def check_import(name: str, import_name: str, *, optional: bool = False) -> DependencyCheck:
    """Import one dependency and capture a clear result instead of raising."""

    try:
        module = importlib.import_module(import_name)
    except ImportError as exc:
        return DependencyCheck(
            name=name,
            import_name=import_name,
            ok=False,
            message=f"Install package providing '{import_name}' ({exc}).",
            optional=optional,
        )

    version = getattr(module, "__version__", None)
    return DependencyCheck(
        name=name,
        import_name=import_name,
        ok=True,
        version=str(version) if version is not None else None,
        optional=optional,
    )


def run_checks() -> list[DependencyCheck]:
    """Run all health checks without opening cameras or simulator windows."""

    checks: list[DependencyCheck] = []
    for name, import_name, optional in DEPENDENCIES:
        if import_name is None:
            checks.append(check_python())
        else:
            checks.append(check_import(name, import_name, optional=optional))
    return checks


def format_check(check: DependencyCheck) -> str:
    """Format one result line for terminal output."""

    status = "OK" if check.ok else "MISSING"
    optional = " optional" if check.optional else ""
    version = f" {check.version}" if check.version else ""
    detail = f" - {check.message}" if check.message else ""
    return f"{status:7} {check.name}{optional}{version}{detail}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check DexVision runtime imports.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)

    print("DexVision health check")
    for check in run_checks():
        print(format_check(check))

    print("No camera, MuJoCo model, or GUI was opened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
