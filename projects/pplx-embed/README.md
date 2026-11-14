# pplx-embed — Local Embedding Service

Production-minded local embedding stack around **Perplexity `pplx-embed-v1-0.6b`** with a FastAPI service, Streamlit exploration UI, embedding projector, persistent model cache, Docker Compose, health checks, and offline unit tests.

## Architecture

```text
Client / Streamlit
       │
       ▼
FastAPI
├── /health   (liveness)
├── /ready    (readiness; 503 until the model loads)
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

## Layout

```text
src/pplx_embed/
├── api/        FastAPI app and embedder adapter
├── ui/         Streamlit exploration UI
├── scripts/    model download and projector export
└── paths.py    project and model-cache resolution
```

`pyproject.toml` is the only dependency declaration. The Docker images install the
package with `pip install .` and `pip install '.[ui]'`; there are no separate
requirements files to keep in sync.

## Engineering decisions

- **Python 3.11** is the supported runtime for deterministic ML dependency compatibility.
- The API is created through `create_app()`, enabling model-free unit tests through dependency injection.
- Model storage is configured once through `MODEL_CACHE_DIR`; local and Docker execution use the same contract.
- Docker runs as a non-root user and exposes health checks.
- CORS is explicit and environment-configurable instead of `*`.
- Internal exception details are logged server-side without being echoed directly to clients.
- Model auto-download is opt-in locally and enabled for the Docker demo so the named volume can populate on first run.
- `trust_remote_code=True` is treated as a documented security boundary; production deployments should pin a reviewed `MODEL_REVISION`.
- `/ready` gates Docker health checks and Compose startup; `/health` remains a liveness endpoint that does not claim readiness.
- `EMBEDDER_BACKEND=fake` loads a deterministic, dependency-free embedder for CI container smoke tests and UI development without a model download.
- The dependency metadata keeps Intel macOS on an installable `torch` line; Linux CI resolves and audits the fixed line. Intel macOS is a local-development limitation, not a CI exception.

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

API tests inject a deterministic fake embedder, so pull requests do not require network access or
model downloads. Projector tests exercise PCA, t-SNE, tiny-dataset fallbacks, and offline HTML
export independently of the model-serving path. Exported HTML bundles Plotly JS and makes no
external CDN or font requests.

To run the API without a model locally or in a container:

```bash
EMBEDDER_BACKEND=fake uvicorn pplx_embed.api.main:app --reload
curl --fail http://localhost:8000/ready
curl --fail -X POST http://localhost:8000/embed \
  -H 'Content-Type: application/json' -d '{"texts":["hello"]}'
```

## Intel macOS note

PyTorch no longer publishes wheels for Intel macOS. The project metadata keeps that platform on
`torch>=2.2,<2.3` so local development remains possible; Linux and ARM resolve the fixed
`torch>=2.15,<3` line. Dependency audits run on Linux and are authoritative.

## Operational notes

`/health` is a liveness check. `/ready` reports whether the application lifespan successfully loaded a model and returns HTTP 503 until it has. If Docker remains unhealthy, inspect the API logs and verify model download access, free disk space, memory, and write permissions on the model volume.
