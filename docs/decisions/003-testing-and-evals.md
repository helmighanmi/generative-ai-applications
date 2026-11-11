<!--
Path: docs/decisions/003-testing-and-evals.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# ADR-003: Separate Software Tests from AI Evaluations

**Status:** Accepted

## Decision

Deterministic correctness checks live under each project's `tests/`. AI quality benchmarks live under `evals/` (project-local when domain-specific, root-level for conventions/templates).

CI runs offline deterministic tests by default. Real-model/API evaluation jobs must be explicitly enabled and must publish measurable thresholds.
