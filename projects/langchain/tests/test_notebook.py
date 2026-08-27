# Path: projects/langchain/tests/test_notebook.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

import json
from pathlib import Path


def test_document_loader_notebook_is_valid_and_has_code_cells() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "LangChain document loader-v1.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert any(cell.get("cell_type") == "code" for cell in notebook["cells"])
