# Path: projects/pplx-embed-project/tests/test_dependencies.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Keep the Docker requirements file aligned with the packaging metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _requirement_lines(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_api_requirements_match_pyproject_dependencies() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(metadata["project"]["dependencies"])
    docker = _requirement_lines(PROJECT_ROOT / "api" / "requirements.txt")

    assert docker == declared
