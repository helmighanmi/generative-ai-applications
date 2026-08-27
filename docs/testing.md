<!--
Path: docs/testing.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Testing and AI Evaluation

Software testing and AI evaluation answer different questions.

- **Unit tests:** deterministic logic, validation, transformations, error handling.
- **Integration tests:** service boundaries, vector stores, file parsers, framework adapters.
- **Smoke tests:** the smallest end-to-end path proving an application starts or a pipeline executes.
- **AI evaluations:** answer quality, retrieval relevance, faithfulness, latency/cost regressions, and adversarial cases.

CI should run deterministic tests on every pull request. Expensive model/API evaluations should use a controlled dataset and run only when credentials/resources are available, with explicit thresholds rather than subjective inspection.
