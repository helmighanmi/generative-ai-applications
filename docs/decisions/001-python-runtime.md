<!--
Path: docs/decisions/001-python-runtime.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# ADR-001: Python 3.11 Baseline

**Status:** Accepted

## Context

The monorepo combines scientific Python, PyMuPDF/OpenCV, FAISS, sentence-transformers/PyTorch, FastAPI, Streamlit, and LangChain. AI/ML projects often depend on compiled wheels whose support can lag the newest CPython release.

## Decision

Use **Python 3.11** as the default development, Docker, and CI runtime. Project metadata expresses `>=3.11,<3.12` until each project has an explicit compatibility matrix.

## Consequences

- Reproducibility is prioritized over chasing the newest interpreter.
- Binary-wheel availability is strong across the repository's dependency classes.
- Moving to Python 3.12+ becomes an explicit tested change rather than an accidental environment difference.
