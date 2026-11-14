"""Emit a GitHub Actions matrix for the project directories.

Each project contributes its directory name, distribution name and the install
target used by the dependency audit. The matrix is derived from pyproject.toml,
so adding a project requires no workflow edit.
"""

from __future__ import annotations

import json
import tomllib

from repo_layout import project_dirs, relative


def audit_target(metadata: dict) -> str:
    extras = metadata.get("project", {}).get("optional-dependencies", {})
    return ".[ai]" if "ai" in extras else "."


def main() -> int:
    include = []
    for project in project_dirs():
        metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
        name = metadata.get("project", {}).get("name")
        if not name:
            raise SystemExit(f"{relative(project)}: pyproject.toml is missing project.name")
        include.append(
            {
                "project": project.name,
                "package": name,
                "audit": audit_target(metadata),
            }
        )
    print(json.dumps({"include": include}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
