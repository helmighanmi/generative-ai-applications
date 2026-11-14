# ADR-003: Separate Software Tests from AI Evaluations

**Status:** Accepted

## Decision

Deterministic correctness checks live under each project's `tests/`. AI quality benchmarks live under `evals/` (project-local when domain-specific, root-level for conventions/templates).

CI runs offline deterministic tests by default. Real-model/API evaluation jobs must be explicitly enabled and must publish measurable thresholds.
