# Path: scripts/run_tests.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Run fast, offline repository tests with project-local import paths."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = [
    ROOT / "projects" / "document-parsing",
    ROOT / "projects" / "langchain",
    ROOT / "projects" / "multimodal-rag-pipeline",
    ROOT / "projects" / "pplx-embed-project",
    ROOT / "projects" / "rag-qa-bot-langchain",
]


def run_project(project: Path) -> int:
    tests = project / "tests"
    if not tests.is_dir():
        return 0
    env = os.environ.copy()
    src = project / "src"
    if src.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else str(src)
    print(f"\n==> {project.name}", flush=True)
    return subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=project, env=env)


def main() -> int:
    failed = [project.name for project in PROJECTS if run_project(project) != 0]
    if failed:
        print(f"\nFailed projects: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nAll lightweight test suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
