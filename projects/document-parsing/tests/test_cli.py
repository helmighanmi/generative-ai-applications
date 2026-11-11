# Path: projects/document-parsing/tests/test_cli.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_emits_one_json_object_per_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "two-pages.pdf"
    document = pymupdf.open()
    for index in range(2):
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 750),
            f"Applied mathematics and artificial intelligence engineering, page {index + 1}. " * 40,
            fontsize=10,
        )
    document.save(pdf_path)
    document.close()

    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pdf_page_classifier.cli",
            str(pdf_path),
            "--calibration-pages",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payloads = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert [payload["page"] for payload in payloads] == [1, 2]
    for payload in payloads:
        assert payload["hybrid"] in {"document", "diagram"}
        assert payload["adaptive"] in {"document", "diagram"}
        assert 0.0 <= payload["confidence"] <= 1.0
