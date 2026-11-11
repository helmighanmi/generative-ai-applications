# Path: scripts/check_docs_links.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Verify that relative Markdown links resolve inside the repository."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "#")


def markdown_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "SECURITY.md"]
    files.extend(sorted((ROOT / "docs").rglob("*.md")))
    files.extend(sorted((ROOT / "projects").glob("*/README.md")))
    return [path for path in files if path.is_file()]


def check_file(path: Path) -> list[str]:
    problems: list[str] = []
    text = path.read_text(encoding="utf-8")
    for raw_target in LINK_PATTERN.findall(text):
        target = raw_target.strip()
        if target.startswith(SKIP_PREFIXES):
            continue
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        target = unquote(urlsplit(target).path)
        if not target:
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            problems.append(f"{path.relative_to(ROOT)}: missing target {target!r}")
    return problems


def main() -> int:
    problems = [problem for path in markdown_files() for problem in check_file(path)]
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"Checked {len(markdown_files())} Markdown files; all relative links resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
