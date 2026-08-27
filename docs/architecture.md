<!--
Path: docs/architecture.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Architecture

## Monorepo boundary

This repository is a portfolio of independent AI systems and experiments. The root owns engineering policy (CI, security, documentation, developer commands); each folder under `projects/` owns its runtime dependencies and execution model.

```text
repository
├── .github/workflows/      CI and security automation
├── docs/                   architecture, ADRs, runbooks
├── evals/                  cross-project evaluation conventions
├── projects/
│   ├── document-parsing/   applied document intelligence
│   ├── langchain/          framework exploration
│   ├── multimodal-rag-pipeline/
│   ├── pplx-embed-project/
│   └── rag-qa-bot-langchain/
└── scripts/                repository-level automation
```

## Design principles

1. **Independent deployability:** service projects do not import code from sibling projects.
2. **Explicit provider boundaries:** external LLM/model/vector-store dependencies are wrapped behind small interfaces where practical.
3. **Offline-first tests:** unit tests do not require paid APIs or multi-gigabyte model downloads.
4. **Configuration over source edits:** environment-specific values are injected through environment variables/config files.
5. **Failure is visible:** CI fails on failed tests; runtime errors are logged and returned without leaking unnecessary internals.
6. **Complexity must earn its place:** no Kubernetes, queues, or distributed components are added unless a project actually needs them.
