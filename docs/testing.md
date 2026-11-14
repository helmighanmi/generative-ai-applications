# Testing and AI Evaluation

Software testing and AI evaluation answer different questions.

- **Unit tests:** deterministic logic, validation, transformations, error handling.
- **Integration tests:** service boundaries, vector stores, file parsers, framework adapters.
- **Smoke tests:** the smallest end-to-end path proving an application starts or a pipeline executes.
- **AI evaluations:** answer quality, retrieval relevance, faithfulness, latency/cost regressions, and adversarial cases.

CI should run deterministic tests on every pull request. Expensive model/API evaluations should use a controlled dataset and run only when credentials/resources are available, with explicit thresholds rather than subjective inspection.

## Current integration coverage

- Stub embeddings are deterministic across processes (SHA-256 seed, not Python's salted `hash()`).
- Image stubs hash the full payload, so different images produce different vectors.
- A FAISS index built in one process is saved, reloaded in a subprocess, and queried successfully.
- The pplx API container answers `/ready` and `/embed` with `EMBEDDER_BACKEND=fake`.
- The OpenAI generation branch is exercised with a fake response object, without a network call.

`CI / integration-smoke` runs the cross-process pipeline tests and fails on any non-zero exit.
