"""Run offline tests for every project and lab in the repository."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from repo_layout import testable_dirs


def run_project(project: Path) -> int:
    env = os.environ.copy()
    src = project / "src"
    if src.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else str(src)
    print(f"\n==> {project.name}", flush=True)
    return subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=project, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=("all", "projects", "labs"),
        default="all",
        help="Which part of the repository to test (default: all)",
    )
    args = parser.parse_args()

    projects = testable_dirs(args.group)
    if not projects:
        print(f"No testable directories found for group {args.group!r}.", file=sys.stderr)
        return 1

    failed = [project.name for project in projects if run_project(project) != 0]
    if failed:
        print(f"\nFailed suites: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nAll lightweight test suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
