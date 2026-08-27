<!--
Path: docs/decisions/002-monorepo-structure.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# ADR-002: Keep AI Projects Independently Runnable in a Monorepo

**Status:** Accepted

## Context

The repository contains experiments, pipelines, and services with different dependency footprints. A single shared runtime would create unnecessary conflicts and couple unrelated work.

## Decision

Keep projects under `projects/` with project-local dependencies and commands. Root tooling enforces shared engineering standards only.

## Consequences

- A reviewer sees one coherent engineering portfolio.
- Each application remains reproducible in isolation.
- Cross-project reuse should become a real shared package only after repeated, stable duplication justifies it.
