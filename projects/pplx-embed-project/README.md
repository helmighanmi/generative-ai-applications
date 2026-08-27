<!--
Path: projects/pplx-embed-project/README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# pplx-embed — Local Embedding Service

Production-minded local embedding stack around **Perplexity `pplx-embed-v1-0.6b`** with a FastAPI service, Streamlit exploration UI, embedding projector, persistent model cache, Docker Compose, health checks, and offline unit tests.

## Architecture

```text
Client / Streamlit
       │
       ▼
FastAPI
├── /health
├── /embed
└── /similarity
       │
       ▼
EmbeddingModel
       │
       ▼
MODEL_CACHE_DIR  ── persistent Docker volume / local model directory
       │
       ▼
pplx-embed-v1-0.6b (CPU)
```

## Engineering decisions

- **Python 3.11** is the supported runtime for deterministic ML dependency compatibility.
- The API is created through `create_app()`, enabling model-free unit tests through dependency injection.
- Model storage is configured once through `MODEL_CACHE_DIR`; local and Docker execution use the same contract.
- Docker runs as a non-root user and exposes health checks.
- CORS is explicit and environment-configurable instead of `*`.
- Internal exception details are logged server-side without being echoed directly to clients.
- Model auto-download is opt-in locally and enabled for the Docker demo so the named volume can populate on first run.
- `trust_remote_code=True` is treated as a documented security boundary; production deployments should pin a reviewed `MODEL_REVISION`.

## Quick start — Docker

```bash
docker compose up --build
```

Services:

- FastAPI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

The first run downloads a multi-gigabyte model into the `model_cache` volume. Subsequent restarts reuse it.

## Local development

```bash
./launch.sh install
source .venv/bin/activate
./launch.sh download
make test
./launch.sh api
```

In another shell:

```bash
source .venv/bin/activate
./launch.sh ui
```

## Configuration

Copy `.env.example` or export variables in your runtime environment:

```bash
MODEL_CACHE_DIR=./model
MODEL_AUTO_DOWNLOAD=false
MODEL_REVISION=
CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
API_URL=http://localhost:8000
```

For production, pin `MODEL_REVISION` to a reviewed model commit rather than tracking a mutable remote revision.

## Tests

```bash
make test
```

API tests inject a deterministic fake embedder, so pull requests do not require network access or model downloads. Projector tests exercise PCA and HTML export independently of the model-serving path.

## Operational notes

`/health` reports whether the application lifespan successfully loaded a model. If Docker remains unhealthy, inspect the API logs and verify model download access, free disk space, memory, and write permissions on the model volume.
