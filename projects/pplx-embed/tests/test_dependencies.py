"""Guards for the single-source dependency policy."""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRECTORIES = {".venv", "build", ".pytest_cache", "__pycache__"}


def test_no_requirements_files() -> None:
    found = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in PROJECT_ROOT.rglob("requirements*.txt")
        if not IGNORED_DIRECTORIES.intersection(path.parts)
    )
    assert found == []


def test_pyproject_is_the_dependency_source() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert project["dependencies"]
    assert project["optional-dependencies"]["ui"]
    assert project["scripts"]["pplx-embed-download"]
    assert project["scripts"]["pplx-embed-projector"]
    assert metadata["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]


def test_dockerfiles_install_the_package() -> None:
    for name in ("Dockerfile.api", "Dockerfile.streamlit"):
        text = (PROJECT_ROOT / "docker" / name).read_text(encoding="utf-8")
        assert "pyproject.toml" in text
        assert "requirements.txt" not in text
