<!--
Path: README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Generative AI Applications — Engineering Portfolio

Production-minded AI engineering monorepo covering document intelligence, retrieval-augmented generation (RAG), local embedding services, multimodal retrieval, testing, containerization, CI/CD, and operational practices.

> **Engineering goal:** make every project understandable, testable, reproducible, and runnable by another engineer from a clean checkout.

## Engineering lifecycle demonstrated

```text
Problem / experiment
      │
      ▼
Architecture & ADRs
      │
      ▼
Implementation ──► Unit tests
      │                 │
      ▼                 ▼
AI evaluation ───► Integration / smoke tests
      │
      ▼
Container build
      │
      ▼
CI quality gates ──► Security checks
      │
      ▼
Runnable service / reproducible experiment
      │
      ▼
Logging, health checks, runbooks
```

## Repository map

| Project | Focus | Primary engineering signal |
|---|---|---|
| [`projects/document-parsing`](projects/document-parsing/) | PDF page classification | Applied ML + deterministic heuristics + calibration |
| [`projects/langchain`](projects/langchain/) | LangChain foundations | Framework experiments kept isolated from production services |
| [`projects/multimodal-rag-pipeline`](projects/multimodal-rag-pipeline/) | Multimodal PDF RAG | Provider abstraction, FAISS retrieval, configurable pipeline |
| [`projects/pplx-embed-project`](projects/pplx-embed-project/) | Local embedding API + UI | FastAPI, Streamlit, Docker Compose, health checks |
| [`projects/rag-qa-bot-langchain`](projects/rag-qa-bot-langchain/) | PDF QA / RAG | Retrieval pipeline with modern LangChain package boundaries |

Cross-cutting engineering material lives in [`docs/`](docs/), root automation in [`Makefile`](Makefile), and GitHub automation in [`.github/workflows`](.github/workflows/).

## Python baseline

**Python 3.11** is the repository baseline. The projects combine scientific Python, PyTorch/sentence-transformers, FAISS, OpenCV/PyMuPDF, and LangChain. Python 3.11 is intentionally conservative for AI/ML binary-wheel compatibility while remaining actively supported and modern.

See [`docs/decisions/001-python-runtime.md`](docs/decisions/001-python-runtime.md) for the decision record.

## Quick start

```bash
# Create a root development environment for repository tooling
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'

# Repository-wide static checks
make lint
make format-check
make compile

# Run lightweight tests that do not download large models
make test
```

Each subproject has its own installation and run instructions because their runtime dependency sets differ significantly.

## Quality gates

Pull requests are expected to pass:

- Python syntax compilation
- Ruff lint and formatting checks
- project unit tests
- dependency metadata validation
- secret-pattern checks
- Docker build validation for service projects
- CodeQL security analysis

CI never converts a failing test into a successful build.

## Public repository / secrets policy

No credentials are required in source control. Copy `.env.example` files where provided and inject secrets at runtime. Never commit `.env`, API keys, cloud credentials, model access tokens, or private datasets.

The example values in this repository are placeholders only.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — monorepo architecture and boundaries
- [`docs/development.md`](docs/development.md) — local development workflow
- [`docs/testing.md`](docs/testing.md) — software tests vs AI evaluation
- [`docs/security.md`](docs/security.md) — public-repository security practices
- [`docs/runbook.md`](docs/runbook.md) — operational checks and troubleshooting
- [`docs/decisions/`](docs/decisions/) — architecture decision records (ADRs)

## Author

**GHANMI Helmi**  
Current Role: **AI Engineer**  
Past Role: **Researcher in Applied Mathematics**  
Research profile: <https://www.researchgate.net/profile/Ghanmi-Helmi>

## License

MIT — see [`LICENSE`](LICENSE).
