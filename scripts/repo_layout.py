"""Discover the repository's projects and labs from the filesystem."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _directories(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted(
        path
        for path in parent.iterdir()
        if path.is_dir() and not path.name.startswith(".") and not path.name.startswith("_")
    )


def project_dirs() -> list[Path]:
    """Return directories under projects/ that look like a project."""
    return [path for path in _directories(ROOT / "projects") if (path / "pyproject.toml").is_file()]


def lab_dirs() -> list[Path]:
    """Return directories under labs/ that look like a lab."""
    return [path for path in _directories(ROOT / "labs") if (path / "README.md").is_file()]


def testable_dirs(group: str) -> list[Path]:
    """Return project or lab directories that declare a tests/ directory."""
    if group == "projects":
        candidates = project_dirs()
    elif group == "labs":
        candidates = lab_dirs()
    else:
        candidates = project_dirs() + lab_dirs()
    return [path for path in candidates if (path / "tests").is_dir()]


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()
