<!--
Path: CHANGELOG.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Changelog

## Unreleased — CI and end-to-end recovery

- Restored `.gitignore` and `.python-version` after a refactor renamed them.
- Fixed documentation links by renaming `docs/descisions` to `docs/decisions`.
- Removed duplicate root-level GitHub configuration files and activated `.pre-commit-config.yaml`.
- Consolidated Ruff configuration and pinned the Ruff version used by CI and pre-commit.
- Made the root development environment installable with `pip install -e '.[dev]'`.
- Updated the embedding dependency stack and dependency-audit policy.
- Replaced the vulnerable ChromaDB dependency in `rag-qa-bot-langchain` with a local FAISS retriever.
- Made offline stub embeddings deterministic across processes.
- Added readiness handling and end-to-end smoke tests.
- Audited the multimodal provider stack by installing its `ai` extra, and documented the
  `.[dev,ai]` install for local provider use.
- Raised the `pymupdf` floor to `1.24.3`, the first release that ships the `pymupdf` module name.
- Fixed the embedding projector for current scikit-learn: t-SNE uses `max_iter`, small datasets
  fall back to PCA, and exported HTML bundles Plotly without external font/CDN requests.
- Added `faiss-cpu` to the root dev extras so the documented `make test` quick start works from a
  clean checkout.
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
