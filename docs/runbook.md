<!--
Path: docs/runbook.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Runbook

Last verified: 2026-11-08 with Python 3.11.

## CI fails at repository quality

Run the same commands locally:

```bash
make compile
make lint
make format-check
```

Syntax and import-layout failures are intentionally detected before the expensive test jobs.
Ruff is pinned to `0.17.4` in the root `pyproject.toml`, `.pre-commit-config.yaml`, and CI.
If a nested project needs a different rule, add `extend = "../../pyproject.toml"` instead of a
standalone `[tool.ruff]` block.

## CI fails at project tests

```bash
python scripts/run_tests.py
```

Or run a single project from its directory:

```bash
cd projects/multimodal-rag-pipeline
python -m pytest -q
```

Each project's `pyproject.toml` sets `pythonpath`, so tests do not depend on the caller's
`PYTHONPATH`. Root `pip install -e '.[dev]'` installs repository tooling only. The
`multimodal-rag-pipeline` provider stack is an extra: use `pip install -e '.[dev,ai]'` for a full
local environment. CI installs `.[ai]` explicitly in the audit and integration jobs.

## CI fails at integration-smoke

The job proves that the offline pipeline works across process boundaries:

```bash
cd projects/multimodal-rag-pipeline
python -m pytest -q tests/test_pipeline.py
```

It builds a FAISS index with stub embeddings, saves it, loads it in a subprocess, and asserts
the expected document is retrieved. A failure here usually means stub embeddings changed
between processes or the persisted index/metadata are out of sync.

## Dependency audit failure

The `dependency-audit` matrix installs each project and runs `pip-audit --strict` on Linux.
Linux CI is authoritative; Intel macOS cannot resolve the same `torch` line.

1. Reproduce in a clean Linux environment or container.
2. Identify the vulnerable package and the first fixed version.
3. Upgrade the dependency range in the owning `pyproject.toml` (and `api/requirements.txt`
   for the pplx API).
4. Re-run `pip-audit --strict`.
5. If no fixed release exists, follow [`security-audit-policy.md`](security-audit-policy.md).
   Add an expiring `--ignore-vuln` entry with an owner, reason, and mitigation. Never add a
   blanket ignore.

## Model-serving project is unhealthy

1. Check liveness with `/health`.
2. Check readiness with `/ready`. The Docker health check and Compose `depends_on` condition use
   `/ready`, which returns HTTP 503 until a model is loaded.
3. Check container logs for model download/load failures.
4. Confirm adequate memory and a writable model cache volume.
5. Confirm the model revision/custom code is compatible with the pinned transformer stack.
6. For a model-free smoke test, start the API with `EMBEDDER_BACKEND=fake`.

## RAG results regress

1. Reproduce with the same input document and query set.
2. Compare chunking parameters and retrieval top-k.
3. Confirm embedding model/provider did not change.
4. Run deterministic retrieval evals before changing the generation model.

## Stub embeddings differ between runs

Stub vectors are seeded with SHA-256, not Python's salted `hash()`. If vectors differ:

1. Check that the same `text`/`image_b64` payload is used (image stubs hash the full payload).
2. Confirm `EmbeddingService.dim` matches between the build and query processes.
3. Re-run `tests/test_pipeline.py`; the cross-process test fails if determinism regresses.

## Secret accidentally committed

Revoke/rotate the credential first. Removing the file in a later commit is not sufficient; purge
repository history if required and verify downstream caches/artifacts.
