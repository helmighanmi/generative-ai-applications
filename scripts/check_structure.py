"""Enforce the repository layout contract.

The rules are documented in docs/architecture/repo-layout.md. This script keeps
them honest: it fails when a project drifts from the skeleton, when a dependency
file reappears, or when a removed convention (per-file author headers, orphaned
image directories) comes back.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from repo_layout import ROOT, lab_dirs, project_dirs, relative

REQUIRED_PROJECT_ENTRIES = ("README.md", "pyproject.toml", "src", "tests")
DOCS_SECTIONS = ("architecture", "decisions", "guides", "operations", "history")
HEADER_MARKERS = ("Research Profile:", "Current Role:", "Author: GHANMI")
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
    ".sh",
    ".cfg",
    ".ini",
    ".example",
}


def tracked_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    return [ROOT / name for name in output.decode().split("\0") if name]


def check_projects() -> list[str]:
    problems: list[str] = []
    for project in project_dirs():
        for entry in REQUIRED_PROJECT_ENTRIES:
            if not (project / entry).exists():
                problems.append(f"{relative(project)}: missing {entry}")

        src = project / "src"
        if src.is_dir() and not any(src.glob("*/__init__.py")):
            problems.append(f"{relative(project)}: src/ contains no package with __init__.py")

        metadata_path = project / "pyproject.toml"
        if metadata_path.is_file():
            metadata = tomllib.loads(metadata_path.read_text(encoding="utf-8"))
            if not metadata.get("project", {}).get("name"):
                problems.append(f"{relative(project)}: pyproject.toml has no project.name")
    return problems


def check_labs() -> list[str]:
    problems: list[str] = []
    for lab in lab_dirs():
        if not (lab / "README.md").is_file():
            problems.append(f"{relative(lab)}: missing README.md")
        if (lab / "pyproject.toml").is_file():
            problems.append(f"{relative(lab)}: labs are not packaged; remove pyproject.toml")
    return problems


def check_no_requirements() -> list[str]:
    return [
        f"{relative(path)}: requirements files are not allowed; use pyproject.toml"
        for path in tracked_files()
        if path.name.startswith("requirements") and path.suffix == ".txt"
    ]


def check_no_orphaned_images() -> list[str]:
    return [
        f"{relative(path)}: image directories are not part of the layout"
        for path in tracked_files()
        if "images" in path.parts
    ]


def check_no_author_headers() -> list[str]:
    problems: list[str] = []
    for path in tracked_files():
        if path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            head = path.read_text(encoding="utf-8").splitlines()[:20]
        except UnicodeDecodeError:
            continue
        if any(marker in line for line in head for marker in HEADER_MARKERS):
            problems.append(f"{relative(path)}: per-file author header is not allowed")
    return problems


def check_docs() -> list[str]:
    problems: list[str] = []
    docs = ROOT / "docs"
    if not (docs / "README.md").is_file():
        problems.append("docs/README.md: documentation index is missing")
    for entry in sorted(docs.iterdir()):
        if entry.is_file() and entry.name != "README.md":
            problems.append(f"docs/{entry.name}: loose documentation file; move it into a section")
        if entry.is_dir() and entry.name not in DOCS_SECTIONS and not entry.name.startswith("."):
            problems.append(f"docs/{entry.name}/: unknown section")
    for section in DOCS_SECTIONS:
        if not (docs / section).is_dir():
            problems.append(f"docs/{section}/: missing section")
    return problems


def check_root_evals() -> list[str]:
    if (ROOT / "evals").exists():
        return ["evals/: project evaluation belongs to the project that owns it"]
    return []


def check_no_tracked_build_artifacts() -> list[str]:
    problems: list[str] = []
    for path in tracked_files():
        if "build" in path.parts or path.name.endswith(".egg-info"):
            problems.append(f"{relative(path)}: build artifact should not be tracked")
    return problems


def main() -> int:
    problems = []
    problems.extend(check_projects())
    problems.extend(check_labs())
    problems.extend(check_no_requirements())
    problems.extend(check_no_orphaned_images())
    problems.extend(check_no_author_headers())
    problems.extend(check_docs())
    problems.extend(check_root_evals())
    problems.extend(check_no_tracked_build_artifacts())

    if problems:
        print("Repository structure check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(
        f"Structure check passed: {len(project_dirs())} projects, "
        f"{len(lab_dirs())} labs, {len(DOCS_SECTIONS)} documentation sections."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
