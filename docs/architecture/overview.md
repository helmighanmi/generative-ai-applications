# Architecture

## Monorepo boundary

This repository is a portfolio of independent AI systems and experiments. The root owns engineering policy (CI, security, documentation, developer commands); each folder under `projects/` owns its runtime dependencies and execution model.

```text
repository
├── .github/workflows/      CI and security automation
├── docs/                   architecture, ADRs, guides, runbooks
├── projects/               installed, tested, runnable
│   ├── document-parsing/   applied document intelligence
│   ├── multimodal-rag-pipeline/
│   ├── pplx-embed/
│   └── rag-qa-bot/
├── labs/                   exploratory notebooks, never imported
│   └── langchain-foundations/
└── scripts/                repository-level automation
```

The distinction matters: `projects/` code is installed, tested, and run; `labs/` code is read.
A lab that becomes load-bearing graduates into `projects/` with a package, tests, and a run command.

## Design principles

1. **Independent deployability:** service projects do not import code from sibling projects.
2. **Explicit provider boundaries:** external LLM/model/vector-store dependencies are wrapped behind small interfaces where practical.
3. **Offline-first tests:** unit tests do not require paid APIs or multi-gigabyte model downloads.
4. **Configuration over source edits:** environment-specific values are injected through environment variables/config files.
5. **Failure is visible:** CI fails on failed tests; runtime errors are logged and returned without leaking unnecessary internals.
6. **Complexity must earn its place:** no Kubernetes, queues, or distributed components are added unless a project actually needs them.
