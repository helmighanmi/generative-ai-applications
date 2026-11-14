# Changelog

## Unreleased — Repository structure

- Removed the five-line author header from every tracked file and from notebook
  metadata; authorship lives in git history, CODEOWNERS and CONTRIBUTING.
- Split `labs/` from `projects/`: projects are installed, tested and run; labs
  are read and never imported. The rule is recorded in ADR-004.
- Renamed `pplx-embed-project` to `pplx-embed` and `rag-qa-bot-langchain` to
  `rag-qa-bot` so directory and distribution names agree.
- Moved pplx-embed to the `src/` layout and removed its `sys.path` and
  `pythonpath` workarounds.
- Reorganised `docs/` into architecture, decisions, guides, operations and
  history, with an index at `docs/README.md`.
- Made `pyproject.toml` the only dependency source and deleted every
  `requirements.txt`; the root `dev` extra is tooling-only.
- Replaced hardcoded project lists with filesystem discovery in the test runner
  and in both CI matrices, and added `scripts/check_structure.py`.
- Adopted mypy in `make check` and CI; fixed the three issues it surfaced.

## Unreleased — CI and end-to-end recovery

- Restored `.gitignore` and `.python-version` after a refactor renamed them.
- Fixed documentation links by renaming `docs/descisions` to `docs/decisions`.
- Removed duplicate root-level GitHub configuration files and activated `.pre-commit-config.yaml`.
- Consolidated Ruff configuration and pinned the Ruff version used by CI and pre-commit.
- Made the root development environment installable with `pip install -e '.[dev]'`;
  `make install` also installs every project in editable mode.
- Updated the embedding dependency stack and dependency-audit policy.
- Replaced the vulnerable ChromaDB dependency in `rag-qa-bot` with a local FAISS retriever.
- Made offline stub embeddings deterministic across processes.
- Added readiness handling and end-to-end smoke tests.
- Audited the multimodal provider stack by installing its `ai` extra, and documented the
  `.[dev,ai]` install for local provider use.
- Raised the `pymupdf` floor to `1.24.3`, the first release that ships the `pymupdf` module name.
- Fixed the embedding projector for current scikit-learn: t-SNE uses `max_iter`, small datasets
  fall back to PCA, and exported HTML bundles Plotly without external font/CDN requests.
- Moved runtime dependencies out of the root extras; `make install` resolves them from each
  project's `pyproject.toml`.
- Reject query embeddings whose dimension does not match the persisted FAISS index instead of
  letting FAISS abort the process.
- Added a reproducible page-classification evaluation: deterministic synthetic
  fixtures, six curated pages, per-model metrics with bootstrap intervals,
  confusion matrices, and per-page error analysis (`make eval-doc-parsing`).

## 2.0.0 — Engineering hardening

- Reframed the repository as an AI engineering monorepo.
- Standardized Python 3.11 as the compatibility baseline.
- Replaced permissive CI behavior with real quality gates.
- Added architecture, testing, security, runbook, and ADR documentation.
- Added root lint/format/test/security tooling.
- Hardened service Dockerfiles and runtime configuration.
- Replaced placeholder tests with meaningful smoke/unit tests where practical.
- Added consistent author and role attribution to text/code files that support comments.
