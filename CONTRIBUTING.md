<!--
Path: CONTRIBUTING.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Contributing

Contributions should keep each project independently runnable and should not introduce heavyweight infrastructure without a demonstrated need.

## Development flow

1. Create a focused branch.
2. Add or update tests before changing behavior.
3. Run `make lint`, `make format-check`, `make compile`, and `make test`.
4. Document architectural changes with an ADR when the decision affects project boundaries, runtime choices, external providers, persistence, or deployment.
5. Never commit credentials, generated model weights, local caches, or private data.

Pull requests should explain the problem, the chosen approach, trade-offs, test evidence, and rollback considerations for operational changes.
